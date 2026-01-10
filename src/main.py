"""
Paper Translator CLI
ArXiv 논문을 한국어로 번역하는 명령줄 인터페이스
"""
import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich import print as rprint

from src.graph import run_translation, translate_arxiv
from src.state import get_state_summary
from src.db.repositories import TerminologyRepository
from src.feedback import (
    SyncManager,
    SyncResult,
    SyncItem,
    SyncAction,
    get_changed_files,
)
from src.collectors import (
    ArxivCollector,
    SemanticScholarCollector,
    ArxivPaper,
    SemanticScholarPaper,
    RateLimitError,
)

# 콘솔 및 앱 초기화
console = Console()
app = typer.Typer(
    name="paper-translator",
    help="ArXiv 논문을 한국어로 번역하는 AI 도구",
    add_completion=False,
    rich_markup_mode="rich",
)

# 하위 명령어 그룹
terms_app = typer.Typer(
    name="terms",
    help="전문용어 관리",
    add_completion=False,
)
app.add_typer(terms_app, name="terms")


# === 유틸리티 함수 ===

def print_header(title: str):
    """헤더 출력"""
    console.print(Panel(f"[bold blue]{title}[/bold blue]", expand=False))


def print_success(message: str):
    """성공 메시지 출력"""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str):
    """에러 메시지 출력"""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str):
    """경고 메시지 출력"""
    console.print(f"[yellow]![/yellow] {message}")


def print_info(message: str):
    """정보 메시지 출력"""
    console.print(f"[blue]ℹ[/blue] {message}")


def create_stats_table(stats: dict) -> Table:
    """통계 테이블 생성"""
    table = Table(title="번역 통계", show_header=True, header_style="bold magenta")
    table.add_column("항목", style="cyan")
    table.add_column("값", style="green")

    table.add_row("총 청크", str(stats.get("total_chunks", "N/A")))
    table.add_row("완료", str(stats.get("completed_chunks", "N/A")))
    table.add_row("실패", str(stats.get("failed_chunks", "N/A")))
    table.add_row("총 토큰", f"{stats.get('total_tokens', 0):,}")
    table.add_row("소요 시간", f"{stats.get('total_time_sec', 0):.1f}초")
    table.add_row("예상 비용", stats.get("estimated_cost_usd", "N/A"))
    table.add_row("용어 매칭률", stats.get("term_match_rate", "N/A"))

    return table


# === translate 명령어 ===

@app.command("translate", help="논문 번역")
def translate(
    url: Optional[str] = typer.Option(
        None, "--url", "-u",
        help="ArXiv PDF URL"
    ),
    arxiv_id: Optional[str] = typer.Option(
        None, "--arxiv-id", "-a",
        help="ArXiv 논문 ID (예: 1706.03762)"
    ),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f",
        help="로컬 PDF 파일 경로",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    domain: str = typer.Option(
        "General", "--domain", "-d",
        help="논문 도메인 (NLP, CV, RL, General)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="출력 디렉토리 (기본: ./output)"
    ),
    no_references: bool = typer.Option(
        True, "--no-references/--with-references",
        help="References 섹션 제외 여부"
    ),
    no_tables: bool = typer.Option(
        False, "--no-tables/--with-tables",
        help="표 추출 제외 여부"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="실제 번역 없이 예상 비용만 계산"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="상세 로그 출력"
    ),
):
    """
    ArXiv 논문을 한국어로 번역합니다.

    사용 예시:

        # ArXiv ID로 번역
        paper-translator translate --arxiv-id 1706.03762 --domain NLP

        # URL로 번역
        paper-translator translate --url https://arxiv.org/pdf/1706.03762

        # 로컬 PDF 번역
        paper-translator translate --file ./paper.pdf --domain CV
    """
    # 입력 검증
    source = None
    if arxiv_id:
        source = arxiv_id
        print_info(f"ArXiv ID: {arxiv_id}")
    elif url:
        source = url
        print_info(f"URL: {url}")
    elif file:
        source = str(file)
        print_info(f"파일: {file}")
    else:
        print_error("--url, --arxiv-id, 또는 --file 중 하나를 지정해야 합니다.")
        raise typer.Exit(1)

    print_header("Paper Translator")
    console.print(f"도메인: [cyan]{domain}[/cyan]")
    console.print(f"References 제외: [cyan]{no_references}[/cyan]")
    console.print(f"표 추출: [cyan]{not no_tables}[/cyan]")
    console.print()

    # Dry run 모드
    if dry_run:
        print_warning("Dry run 모드: 실제 번역은 수행되지 않습니다.")
        # TODO: 예상 비용 계산 로직
        console.print("[dim]예상 비용 계산 기능은 추후 구현 예정입니다.[/dim]")
        return

    # 번역 실행
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]번역 중...", total=8)

            # 진행 상황 콜백
            node_progress = {
                "fetch_pdf": 1,
                "parse_pdf": 2,
                "chunk_text": 3,
                "pre_process": 4,
                "translate_chunks": 5,
                "post_process": 6,
                "generate_markdown": 7,
                "save_output": 8,
            }

            def progress_callback(node_name: str, status: str):
                if node_name in node_progress:
                    progress.update(task, completed=node_progress[node_name])
                    progress.update(task, description=f"[cyan]{node_name}...")

            # 번역 실행
            final_state = run_translation(
                source=source,
                domain=domain,
                exclude_references=no_references,
                extract_tables=not no_tables,
                progress_callback=progress_callback,
            )

        console.print()

        # 결과 출력
        if final_state.get("status") == "failed":
            print_error(f"번역 실패: {final_state.get('error')}")
            raise typer.Exit(1)

        # 성공
        output_info = final_state.get("output", {})
        stats = final_state.get("stats", {})
        metadata = final_state.get("metadata", {})

        print_success("번역 완료!")
        console.print()

        # 메타데이터
        console.print(f"[bold]제목:[/bold] {metadata.get('title', 'N/A')}")
        if metadata.get("arxiv_id"):
            console.print(f"[bold]ArXiv:[/bold] {metadata.get('arxiv_id')}")
        console.print()

        # 통계 테이블
        console.print(create_stats_table(stats))
        console.print()

        # 출력 파일
        file_path = output_info.get("file_path", "N/A")
        console.print(f"[bold]출력 파일:[/bold] [green]{file_path}[/green]")
        console.print(f"[bold]MD5 해시:[/bold] {output_info.get('md5_hash', 'N/A')}")

    except KeyboardInterrupt:
        print_warning("\n번역이 중단되었습니다.")
        raise typer.Exit(130)
    except Exception as e:
        print_error(f"오류 발생: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


# === terms 명령어 그룹 ===

@terms_app.command("list", help="전문용어 목록 조회")
def terms_list(
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d",
        help="도메인 필터 (NLP, CV, RL, General)"
    ),
    search: Optional[str] = typer.Option(
        None, "--search", "-s",
        help="검색어"
    ),
    limit: int = typer.Option(
        50, "--limit", "-n",
        help="최대 출력 개수"
    ),
    json_output: bool = typer.Option(
        False, "--json",
        help="JSON 형식 출력"
    ),
):
    """
    등록된 전문용어 목록을 조회합니다.

    사용 예시:

        # 전체 목록
        paper-translator terms list

        # NLP 도메인만
        paper-translator terms list --domain NLP

        # 검색
        paper-translator terms list --search attention
    """
    try:
        terms = TerminologyRepository.get_all(domain=domain, limit=limit)

        # 검색어 필터
        if search:
            search_lower = search.lower()
            terms = [
                t for t in terms
                if search_lower in t.get("source_text", "").lower()
                or search_lower in t.get("target_text", "").lower()
            ]

        if json_output:
            console.print_json(data=terms)
            return

        # 테이블 출력
        table = Table(title=f"전문용어 목록 ({len(terms)}개)", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="dim", width=8)
        table.add_column("원문 (Source)", style="cyan")
        table.add_column("번역 (Target)", style="green")
        table.add_column("도메인", style="yellow")
        table.add_column("사용횟수", justify="right")

        for term in terms:
            table.add_row(
                str(term.get("id", ""))[:8],
                term.get("source_text", ""),
                term.get("target_text", ""),
                term.get("domain", ""),
                str(term.get("usage_count", 0)),
            )

        console.print(table)

    except Exception as e:
        print_error(f"조회 실패: {e}")
        raise typer.Exit(1)


@terms_app.command("add", help="전문용어 추가")
def terms_add(
    source: str = typer.Option(
        ..., "--source", "-s",
        help="원문 (영어)"
    ),
    target: str = typer.Option(
        ..., "--target", "-t",
        help="번역어 (한국어)"
    ),
    domain: str = typer.Option(
        "General", "--domain", "-d",
        help="도메인 (NLP, CV, RL, General)"
    ),
    description: Optional[str] = typer.Option(
        None, "--description",
        help="설명"
    ),
):
    """
    새 전문용어를 추가합니다.

    사용 예시:

        paper-translator terms add --source "attention" --target "어텐션" --domain NLP
    """
    try:
        term_data = {
            "source_text": source,
            "target_text": target,
            "domain": domain,
            "description": description or "",
            "usage_count": 0,
        }

        result = TerminologyRepository.create(term_data)

        if result:
            print_success(f"용어 추가 완료: '{source}' → '{target}'")
            console.print(f"  도메인: {domain}")
            if description:
                console.print(f"  설명: {description}")
        else:
            print_error("용어 추가 실패")
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"추가 실패: {e}")
        raise typer.Exit(1)


@terms_app.command("update", help="전문용어 수정")
def terms_update(
    term_id: str = typer.Argument(
        ...,
        help="수정할 용어 ID"
    ),
    source: Optional[str] = typer.Option(
        None, "--source", "-s",
        help="새 원문"
    ),
    target: Optional[str] = typer.Option(
        None, "--target", "-t",
        help="새 번역어"
    ),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d",
        help="새 도메인"
    ),
    description: Optional[str] = typer.Option(
        None, "--description",
        help="새 설명"
    ),
):
    """
    기존 전문용어를 수정합니다.

    사용 예시:

        paper-translator terms update abc123 --target "새번역어"
    """
    try:
        update_data = {}
        if source:
            update_data["source_text"] = source
        if target:
            update_data["target_text"] = target
        if domain:
            update_data["domain"] = domain
        if description is not None:
            update_data["description"] = description

        if not update_data:
            print_warning("수정할 항목이 없습니다.")
            return

        result = TerminologyRepository.update(term_id, update_data)

        if result:
            print_success(f"용어 수정 완료: {term_id}")
            for key, value in update_data.items():
                console.print(f"  {key}: {value}")
        else:
            print_error("용어 수정 실패 (ID를 찾을 수 없음)")
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"수정 실패: {e}")
        raise typer.Exit(1)


@terms_app.command("delete", help="전문용어 삭제")
def terms_delete(
    term_id: str = typer.Argument(
        ...,
        help="삭제할 용어 ID"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="확인 없이 삭제"
    ),
):
    """
    전문용어를 삭제합니다.

    사용 예시:

        paper-translator terms delete abc123
        paper-translator terms delete abc123 --force
    """
    try:
        if not force:
            confirm = typer.confirm(f"정말 '{term_id}' 용어를 삭제하시겠습니까?")
            if not confirm:
                print_info("삭제가 취소되었습니다.")
                return

        result = TerminologyRepository.delete(term_id)

        if result:
            print_success(f"용어 삭제 완료: {term_id}")
        else:
            print_error("용어 삭제 실패 (ID를 찾을 수 없음)")
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"삭제 실패: {e}")
        raise typer.Exit(1)


@terms_app.command("export", help="전문용어 내보내기")
def terms_export(
    output: Path = typer.Option(
        Path("./terminology_export.json"), "--output", "-o",
        help="출력 파일 경로"
    ),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d",
        help="도메인 필터"
    ),
    format: str = typer.Option(
        "json", "--format", "-f",
        help="출력 형식 (json, csv)"
    ),
):
    """
    전문용어를 파일로 내보냅니다.

    사용 예시:

        paper-translator terms export --output terms.json
        paper-translator terms export --domain NLP --format csv
    """
    try:
        terms = TerminologyRepository.get_all(domain=domain, limit=10000)

        if format == "json":
            with open(output, "w", encoding="utf-8") as f:
                json.dump(terms, f, ensure_ascii=False, indent=2)
        elif format == "csv":
            import csv
            with open(output, "w", encoding="utf-8", newline="") as f:
                if terms:
                    writer = csv.DictWriter(f, fieldnames=terms[0].keys())
                    writer.writeheader()
                    writer.writerows(terms)
        else:
            print_error(f"지원하지 않는 형식: {format}")
            raise typer.Exit(1)

        print_success(f"내보내기 완료: {output} ({len(terms)}개 용어)")

    except Exception as e:
        print_error(f"내보내기 실패: {e}")
        raise typer.Exit(1)


@terms_app.command("import", help="전문용어 가져오기")
def terms_import(
    file: Path = typer.Argument(
        ...,
        help="가져올 파일 경로",
        exists=True,
        file_okay=True,
        readable=True,
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite",
        help="기존 용어 덮어쓰기"
    ),
):
    """
    파일에서 전문용어를 가져옵니다.

    사용 예시:

        paper-translator terms import terms.json
        paper-translator terms import terms.json --overwrite
    """
    try:
        # 파일 형식 감지
        suffix = file.suffix.lower()

        if suffix == ".json":
            with open(file, "r", encoding="utf-8") as f:
                terms = json.load(f)
        elif suffix == ".csv":
            import csv
            with open(file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                terms = list(reader)
        else:
            print_error(f"지원하지 않는 형식: {suffix}")
            raise typer.Exit(1)

        # 용어 추가
        added = 0
        skipped = 0
        updated = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]가져오는 중...", total=len(terms))

            for term in terms:
                progress.update(task, advance=1)

                term_data = {
                    "source_text": term.get("source_text", ""),
                    "target_text": term.get("target_text", ""),
                    "domain": term.get("domain", "General"),
                    "description": term.get("description", ""),
                    "usage_count": int(term.get("usage_count", 0)),
                }

                if not term_data["source_text"] or not term_data["target_text"]:
                    skipped += 1
                    continue

                # 기존 용어 확인
                existing = TerminologyRepository.search(term_data["source_text"])
                if existing:
                    if overwrite:
                        TerminologyRepository.update(existing[0]["id"], term_data)
                        updated += 1
                    else:
                        skipped += 1
                else:
                    TerminologyRepository.create(term_data)
                    added += 1

        print_success(f"가져오기 완료!")
        console.print(f"  추가: {added}개")
        console.print(f"  수정: {updated}개")
        console.print(f"  건너뜀: {skipped}개")

    except Exception as e:
        print_error(f"가져오기 실패: {e}")
        raise typer.Exit(1)


# === sync 명령어 ===

def create_sync_items_table(items: list[SyncItem]) -> Table:
    """동기화 항목 테이블 생성"""
    table = Table(title="동기화 항목", show_header=True, header_style="bold cyan")
    table.add_column("액션", style="yellow", width=15)
    table.add_column("설명", style="white")
    table.add_column("확신도", justify="right", width=10)

    action_icons = {
        SyncAction.UPDATE_TERM: "[blue]📝 업데이트[/blue]",
        SyncAction.ADD_TERM: "[green]➕ 추가[/green]",
        SyncAction.LOG_CHANGE: "[dim]📋 로그[/dim]",
        SyncAction.UPDATE_HASH: "[cyan]🔄 해시[/cyan]",
        SyncAction.SKIP: "[dim]⏭️ 건너뜀[/dim]",
    }

    for item in items:
        action_text = action_icons.get(item.action, item.action.value)
        confidence = item.data.get("confidence")
        confidence_str = f"{confidence:.0%}" if confidence else "-"

        table.add_row(action_text, item.description, confidence_str)

    return table


def create_sync_result_table(result: SyncResult) -> Table:
    """동기화 결과 테이블 생성"""
    table = Table(title="동기화 결과", show_header=True, header_style="bold magenta")
    table.add_column("항목", style="cyan")
    table.add_column("값", style="green")

    table.add_row("파일", result.file_path)
    table.add_row("성공 여부", "[green]✓[/green]" if result.success else "[red]✗[/red]")
    table.add_row("용어 업데이트", str(result.terms_updated))
    table.add_row("용어 추가", str(result.terms_added))
    table.add_row("변경 로그", str(result.changes_logged))
    table.add_row("해시 업데이트", "[green]✓[/green]" if result.hash_updated else "[dim]-[/dim]")

    if result.error:
        table.add_row("비고", f"[dim]{result.error}[/dim]")

    return table


@app.command("sync", help="번역 결과 동기화")
def sync(
    file: Optional[Path] = typer.Option(
        None, "--file", "-f",
        help="동기화할 마크다운 파일",
        exists=True,
        file_okay=True,
    ),
    all_files: bool = typer.Option(
        False, "--all", "-a",
        help="모든 변경 파일 동기화"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="실제 변경 없이 변경 사항만 확인"
    ),
    auto: bool = typer.Option(
        False, "--auto",
        help="확인 없이 자동 동기화"
    ),
    output_dir: Path = typer.Option(
        Path("./translations"), "--output-dir", "-o",
        help="번역 파일 디렉토리 (--all 옵션용)"
    ),
    use_llm: bool = typer.Option(
        True, "--llm/--no-llm",
        help="LLM 분석 사용 여부"
    ),
    min_confidence: float = typer.Option(
        0.7, "--min-confidence", "-c",
        help="최소 확신도 (0.0 ~ 1.0)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="상세 로그 출력"
    ),
):
    """
    번역된 마크다운 파일의 변경 사항을 DB에 동기화합니다.

    사용자가 번역 결과(마크다운 파일)를 수정하면,
    변경된 용어를 자동으로 DB에 반영합니다.

    사용 예시:

        # 단일 파일 동기화 (확인 필요)
        paper-translator sync --file output/translated.md

        # 변경 사항만 미리보기 (dry-run)
        paper-translator sync --file output/translated.md --dry-run

        # 자동 동기화 (확인 없이)
        paper-translator sync --file output/translated.md --auto

        # 모든 변경 파일 동기화
        paper-translator sync --all --output-dir ./translations

        # LLM 분석 없이 휴리스틱만 사용
        paper-translator sync --file output/translated.md --no-llm
    """
    print_header("번역 결과 동기화")

    # 입력 검증
    if not file and not all_files:
        print_error("--file 또는 --all 옵션 중 하나를 지정해야 합니다.")
        raise typer.Exit(1)

    # 동기화할 파일 목록 결정
    files_to_sync: list[str] = []

    if file:
        files_to_sync = [str(file)]
        print_info(f"대상 파일: {file}")
    elif all_files:
        print_info(f"변경된 파일 검색 중: {output_dir}")
        files_to_sync = get_changed_files(str(output_dir))

        if not files_to_sync:
            print_success("변경된 파일이 없습니다.")
            return

        print_info(f"변경된 파일 {len(files_to_sync)}개 발견")
        for f in files_to_sync:
            console.print(f"  • {f}")

    console.print()

    # 설정 출력
    if verbose:
        console.print(f"[dim]LLM 분석: {use_llm}[/dim]")
        console.print(f"[dim]최소 확신도: {min_confidence:.0%}[/dim]")
        console.print(f"[dim]자동 동기화: {auto}[/dim]")
        console.print(f"[dim]Dry run: {dry_run}[/dim]")
        console.print()

    # 사용자 확인 콜백
    def confirm_sync(items: list[SyncItem]) -> bool:
        """사용자 확인 콜백"""
        # 동기화 항목 테이블 출력
        console.print(create_sync_items_table(items))
        console.print()

        # 요약
        update_count = sum(1 for i in items if i.action == SyncAction.UPDATE_TERM)
        add_count = sum(1 for i in items if i.action == SyncAction.ADD_TERM)
        skip_count = sum(1 for i in items if i.action == SyncAction.SKIP)

        console.print(f"[bold]요약:[/bold] 업데이트 {update_count}건, 추가 {add_count}건, 건너뜀 {skip_count}건")
        console.print()

        return typer.confirm("동기화를 진행하시겠습니까?")

    # SyncManager 생성
    manager = SyncManager(
        auto_sync=auto,
        confirm_callback=None if dry_run else confirm_sync,
        min_confidence=min_confidence,
        use_llm_analysis=use_llm
    )

    # 동기화 실행
    try:
        results: list[SyncResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]동기화 중...", total=len(files_to_sync))

            for file_path in files_to_sync:
                progress.update(task, description=f"[cyan]{Path(file_path).name}...")

                result = manager.sync_file(file_path, dry_run=dry_run)
                results.append(result)

                progress.update(task, advance=1)

        console.print()

        # 결과 출력
        if dry_run:
            print_warning("Dry run 모드: 실제 변경은 수행되지 않았습니다.")
            console.print()

            for result in results:
                if result.items:
                    console.print(f"[bold]{result.file_path}[/bold]")
                    console.print(create_sync_items_table(result.items))
                    console.print()
                else:
                    console.print(f"[dim]{result.file_path}: {result.error or '변경 사항 없음'}[/dim]")
        else:
            # 전체 결과 요약
            summary = manager.get_sync_summary(results)

            console.print(Panel(
                f"[bold]파일:[/bold] {summary['total_files']}개 "
                f"([green]성공 {summary['success']}[/green], "
                f"[red]실패 {summary['failed']}[/red])\n"
                f"[bold]용어 업데이트:[/bold] {summary['terms_updated']}건\n"
                f"[bold]용어 추가:[/bold] {summary['terms_added']}건\n"
                f"[bold]변경 로그:[/bold] {summary['changes_logged']}건",
                title="동기화 완료",
                expand=False
            ))

            # 개별 결과 (verbose 모드)
            if verbose:
                console.print()
                for result in results:
                    console.print(create_sync_result_table(result))
                    console.print()

            # 성공/실패 메시지
            if summary['failed'] == 0:
                print_success("모든 파일 동기화 완료!")
            else:
                print_warning(f"{summary['failed']}개 파일 동기화 실패")

                for result in results:
                    if not result.success:
                        print_error(f"  {result.file_path}: {result.error}")

    except KeyboardInterrupt:
        print_warning("\n동기화가 중단되었습니다.")
        raise typer.Exit(130)
    except Exception as e:
        print_error(f"동기화 실패: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


# === 버전 및 정보 ===

@app.command("version", help="버전 정보")
def version():
    """버전 정보를 출력합니다."""
    from src import __version__
    console.print(f"[bold]Paper Translator[/bold] v{__version__}")
    console.print("ArXiv 논문을 한국어로 번역하는 AI 도구")


@app.command("info", help="시스템 정보")
def info():
    """시스템 및 설정 정보를 출력합니다."""
    from src.utils import settings

    print_header("시스템 정보")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("항목", style="cyan")
    table.add_column("값", style="green")

    table.add_row("OpenAI 모델", settings.openai_model)
    table.add_row("최대 청크 토큰", str(settings.max_chunk_tokens))
    table.add_row("오버랩 토큰", str(settings.overlap_tokens))
    table.add_row("청킹 전략", settings.chunking_strategy)
    table.add_row("번역 Temperature", str(settings.translation_temperature))
    table.add_row("출력 디렉토리", settings.output_directory)
    table.add_row("파일명 형식", settings.filename_format)

    console.print(table)


# === discover 명령어 ===

def create_arxiv_table(papers: list[ArxivPaper], start_idx: int = 0) -> Table:
    """ArXiv 논문 테이블 생성"""
    table = Table(title=f"ArXiv 논문 ({len(papers)}개)", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("ArXiv ID", style="cyan", width=15)
    table.add_column("제목", style="white", max_width=45)
    table.add_column("저자", style="dim", max_width=20)
    table.add_column("카테고리", style="yellow", width=8)
    table.add_column("날짜", style="green", width=10)

    for idx, paper in enumerate(papers, start_idx + 1):
        authors = ", ".join(paper.authors[:2])
        if len(paper.authors) > 2:
            authors += f" 외 {len(paper.authors) - 2}명"

        table.add_row(
            str(idx),
            paper.arxiv_id,
            paper.title[:45] + "..." if len(paper.title) > 45 else paper.title,
            authors[:20] + "..." if len(authors) > 20 else authors,
            paper.primary_category,
            paper.published.strftime("%Y-%m-%d"),
        )

    return table


def paginate_results(papers: list, page: int, per_page: int) -> tuple[list, int, int]:
    """
    결과를 페이지네이션합니다.

    Returns:
        (현재 페이지 논문 목록, 총 페이지 수, 시작 인덱스)
    """
    total = len(papers)
    total_pages = (total + per_page - 1) // per_page  # ceiling division

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    return papers[start_idx:end_idx], total_pages, start_idx


def create_semantic_scholar_table(papers: list[SemanticScholarPaper], start_idx: int = 0) -> Table:
    """Semantic Scholar 논문 테이블 생성"""
    table = Table(title=f"Semantic Scholar 논문 ({len(papers)}개)", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("ArXiv ID", style="cyan", width=15)
    table.add_column("제목", style="white", max_width=40)
    table.add_column("저자", style="dim", max_width=18)
    table.add_column("인용수", style="yellow", justify="right", width=8)
    table.add_column("영향력", style="green", justify="right", width=8)
    table.add_column("연도", style="cyan", width=6)

    for idx, paper in enumerate(papers, start_idx + 1):
        authors = ", ".join(paper.authors[:2])
        if len(paper.authors) > 2:
            authors += f" 외 {len(paper.authors) - 2}명"

        # ArXiv ID 표시 (없으면 "-")
        arxiv_display = paper.arxiv_id if paper.arxiv_id else "-"

        table.add_row(
            str(idx),
            arxiv_display,
            paper.title[:40] + "..." if len(paper.title) > 40 else paper.title,
            authors[:18] + "..." if len(authors) > 18 else authors,
            f"{paper.citation_count:,}",
            f"{paper.influential_citation_count:,}",
            str(paper.year) if paper.year else "-",
        )

    return table


@app.command("discover", help="논문 검색 및 발견")
def discover(
    source: str = typer.Option(
        "arxiv", "--source", "-s",
        help="검색 소스 (arxiv, semantic-scholar)"
    ),
    query: Optional[str] = typer.Option(
        None, "--query", "-q",
        help="검색어"
    ),
    domain: str = typer.Option(
        "General", "--domain", "-d",
        help="도메인 (NLP, CV, ML, RL, Speech, General)"
    ),
    page: int = typer.Option(
        1, "--page", "-p",
        help="페이지 번호 (1부터 시작)"
    ),
    per_page: int = typer.Option(
        10, "--per-page",
        help="페이지당 결과 수 (기본: 10)"
    ),
    max_results: int = typer.Option(
        100, "--max-results", "-n",
        help="최대 결과 수 (전체)"
    ),
    min_citations: int = typer.Option(
        0, "--min-citations", "-c",
        help="최소 인용수 (Semantic Scholar 전용)"
    ),
    year_from: Optional[int] = typer.Option(
        None, "--year-from", "-y",
        help="시작 연도 필터"
    ),
    trending: bool = typer.Option(
        False, "--trending", "-t",
        help="인기/최신 논문 조회"
    ),
    highly_cited: bool = typer.Option(
        False, "--highly-cited", "-h",
        help="고인용 논문 조회 (Semantic Scholar 전용)"
    ),
    json_output: bool = typer.Option(
        False, "--json",
        help="JSON 형식 출력"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="상세 정보 출력"
    ),
):
    """
    ArXiv 또는 Semantic Scholar에서 논문을 검색합니다.

    사용 예시:

        # ArXiv에서 transformer 검색
        paper-translator discover --source arxiv --query "transformer" --domain NLP

        # ArXiv 최신 트렌딩 논문
        paper-translator discover --source arxiv --domain NLP --trending

        # Semantic Scholar에서 고인용 논문 검색
        paper-translator discover --source semantic-scholar --domain ML --highly-cited

        # Semantic Scholar에서 인용수 100 이상 필터
        paper-translator discover --source semantic-scholar --query "BERT" --min-citations 100

        # 페이지네이션 (2페이지 조회)
        paper-translator discover --source arxiv --domain NLP --trending --page 2
    """
    print_header("논문 검색")

    source = source.lower()
    valid_sources = ["arxiv", "semantic-scholar", "s2"]

    if source not in valid_sources:
        print_error(f"지원하지 않는 소스: {source}")
        print_info(f"사용 가능한 소스: {', '.join(valid_sources)}")
        raise typer.Exit(1)

    # 검색어 또는 옵션 확인
    if not query and not trending and not highly_cited:
        print_error("--query, --trending, 또는 --highly-cited 중 하나를 지정해야 합니다.")
        raise typer.Exit(1)

    # 페이지 유효성 검사
    if page < 1:
        print_error("페이지 번호는 1 이상이어야 합니다.")
        raise typer.Exit(1)
    if per_page < 1 or per_page > 100:
        print_error("페이지당 결과 수는 1~100 사이여야 합니다.")
        raise typer.Exit(1)

    console.print(f"소스: [cyan]{source}[/cyan]")
    console.print(f"도메인: [cyan]{domain}[/cyan]")
    if query:
        console.print(f"검색어: [cyan]{query}[/cyan]")
    console.print(f"페이지: [cyan]{page}[/cyan] (페이지당 {per_page}개)")
    console.print()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]검색 중...", total=None)

            # ArXiv 검색
            if source == "arxiv":
                collector = ArxivCollector(max_results=max_results)

                if trending:
                    papers = collector.get_trending(domain=domain, max_results=max_results)
                elif query:
                    papers = collector.search_by_domain(
                        query=query,
                        domain=domain,
                        max_results=max_results,
                    )
                else:
                    papers = collector.get_recent(domain=domain, max_results=max_results)

                progress.update(task, completed=True)

                if not papers:
                    print_warning("검색 결과가 없습니다.")
                    return

                # 전체 결과 저장 (페이지네이션용)
                all_papers = papers
                total_count = len(all_papers)

                # 페이지네이션 적용
                papers, total_pages, start_idx = paginate_results(all_papers, page, per_page)

                if not papers:
                    print_warning(f"페이지 {page}에 결과가 없습니다. (총 {total_pages}페이지)")
                    return

                if json_output:
                    console.print_json(data=[p.to_dict() for p in papers])
                    return

                console.print()
                console.print(create_arxiv_table(papers, start_idx))

                # 상세 정보 출력
                if verbose:
                    console.print()
                    for idx, paper in enumerate(papers, start_idx + 1):
                        console.print(Panel(
                            f"[bold]제목:[/bold] {paper.title}\n"
                            f"[bold]저자:[/bold] {', '.join(paper.authors)}\n"
                            f"[bold]ArXiv ID:[/bold] {paper.arxiv_id}\n"
                            f"[bold]카테고리:[/bold] {', '.join(paper.categories)}\n"
                            f"[bold]게시일:[/bold] {paper.published.strftime('%Y-%m-%d')}\n"
                            f"[bold]PDF:[/bold] {paper.pdf_url}\n\n"
                            f"[bold]초록:[/bold]\n{paper.abstract[:500]}{'...' if len(paper.abstract) > 500 else ''}",
                            title=f"[{idx}] 상세 정보",
                            expand=False,
                        ))

                # 페이지 정보 출력
                console.print()
                console.print(f"[dim]페이지 {page}/{total_pages} (총 {total_count}개 결과)[/dim]")

                if page < total_pages:
                    next_cmd = f"paper-translator discover --source arxiv"
                    if query:
                        next_cmd += f' --query "{query}"'
                    next_cmd += f" --domain {domain} --page {page + 1}"
                    if trending:
                        next_cmd += " --trending"
                    console.print(f"[dim]다음 페이지: {next_cmd}[/dim]")

            # Semantic Scholar 검색
            else:  # semantic-scholar or s2
                collector = SemanticScholarCollector(max_results=max_results)

                if highly_cited:
                    papers = collector.get_highly_cited(
                        domain=domain,
                        max_results=max_results,
                        year_from=year_from,
                        min_citations=max(min_citations, 100),  # 최소 100
                    )
                elif trending:
                    papers = collector.get_influential(
                        domain=domain,
                        max_results=max_results,
                        year_from=year_from,
                    )
                elif query:
                    papers = collector.search_by_domain(
                        query=query,
                        domain=domain,
                        max_results=max_results,
                        year_from=year_from,
                        min_citations=min_citations,
                    )
                else:
                    papers = collector.search_by_domain(
                        query=domain,
                        domain=domain,
                        max_results=max_results,
                    )

                progress.update(task, completed=True)

                if not papers:
                    print_warning("검색 결과가 없습니다.")
                    return

                # 전체 결과 저장 (페이지네이션용)
                all_papers = papers
                total_count = len(all_papers)

                # 페이지네이션 적용
                papers, total_pages, start_idx = paginate_results(all_papers, page, per_page)

                if not papers:
                    print_warning(f"페이지 {page}에 결과가 없습니다. (총 {total_pages}페이지)")
                    return

                if json_output:
                    console.print_json(data=[p.to_dict() for p in papers])
                    return

                console.print()
                console.print(create_semantic_scholar_table(papers, start_idx))

                # 상세 정보 출력
                if verbose:
                    console.print()
                    for idx, paper in enumerate(papers, start_idx + 1):
                        arxiv_info = f"\n[bold]ArXiv ID:[/bold] {paper.arxiv_id}" if paper.arxiv_id else ""
                        pdf_info = f"\n[bold]PDF:[/bold] {paper.pdf_url}" if paper.pdf_url else ""

                        console.print(Panel(
                            f"[bold]제목:[/bold] {paper.title}\n"
                            f"[bold]저자:[/bold] {', '.join(paper.authors)}\n"
                            f"[bold]연도:[/bold] {paper.year or 'N/A'}\n"
                            f"[bold]인용수:[/bold] {paper.citation_count:,}\n"
                            f"[bold]영향력 인용수:[/bold] {paper.influential_citation_count:,}\n"
                            f"[bold]분야:[/bold] {', '.join(paper.fields_of_study) if paper.fields_of_study else 'N/A'}"
                            f"{arxiv_info}{pdf_info}\n\n"
                            f"[bold]초록:[/bold]\n{paper.abstract[:500] if paper.abstract else 'N/A'}{'...' if paper.abstract and len(paper.abstract) > 500 else ''}",
                            title=f"[{idx}] 상세 정보",
                            expand=False,
                        ))

                # 페이지 정보 출력
                console.print()
                console.print(f"[dim]페이지 {page}/{total_pages} (총 {total_count}개 결과)[/dim]")

                if page < total_pages:
                    next_cmd = f"paper-translator discover --source semantic-scholar"
                    if query:
                        next_cmd += f' --query "{query}"'
                    next_cmd += f" --domain {domain} --page {page + 1}"
                    if highly_cited:
                        next_cmd += " --highly-cited"
                    if trending:
                        next_cmd += " --trending"
                    console.print(f"[dim]다음 페이지: {next_cmd}[/dim]")

        console.print()
        print_success(f"{len(papers)}개 논문 표시 (현재 페이지)")

        # 번역 제안
        console.print()
        console.print("[dim]논문 번역하기:[/dim]")
        if source == "arxiv" and papers:
            sample = papers[0]
            console.print(f"  [dim]paper-translator translate --arxiv-id {sample.arxiv_id} --domain {domain}[/dim]")
        elif papers and papers[0].arxiv_id:
            sample = papers[0]
            console.print(f"  [dim]paper-translator translate --arxiv-id {sample.arxiv_id} --domain {domain}[/dim]")

    except KeyboardInterrupt:
        print_warning("\n검색이 중단되었습니다.")
        raise typer.Exit(130)
    except RateLimitError as e:
        print_error(f"API 요청 제한 초과")
        console.print()
        console.print("[yellow]해결 방법:[/yellow]")
        console.print("  1. 몇 분 후 다시 시도")
        console.print("  2. Semantic Scholar API 키 발급 (무료):")
        console.print("     [dim]https://www.semanticscholar.org/product/api#api-key-form[/dim]")
        console.print("  3. 환경변수로 API 키 설정:")
        console.print("     [dim]export SEMANTIC_SCHOLAR_API_KEY=your_key[/dim]")
        console.print()
        console.print("[dim]또는 ArXiv를 사용해 보세요:[/dim]")
        console.print(f"  [dim]paper-translator discover --source arxiv --domain {domain} --trending[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"검색 실패: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


# === 메인 엔트리 ===

def main():
    """메인 엔트리 포인트"""
    app()


if __name__ == "__main__":
    main()
