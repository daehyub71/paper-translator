"""
LangGraph 워크플로우 모듈
번역 파이프라인의 노드와 그래프 정의
"""
import logging
from datetime import datetime
from typing import Optional, Callable

from langgraph.graph import StateGraph, END

from src.state import TranslationState, create_initial_state, get_state_summary
from src.parsers import PDFParser, ParsedPaper
from src.processors import (
    TextChunker,
    PreProcessor,
    Translator,
    PostProcessor,
)
from src.outputs import MarkdownWriter

# 로깅 설정
logger = logging.getLogger(__name__)


# === 노드 함수 정의 ===

def fetch_pdf(state: TranslationState) -> TranslationState:
    """
    PDF 다운로드 또는 로드 노드

    - URL인 경우: 다운로드
    - ArXiv ID인 경우: PDF URL로 변환 후 다운로드
    - 로컬 파일인 경우: 로드
    """
    state["current_node"] = "fetch_pdf"
    state["status"] = "running"

    try:
        source = state["source"]
        parser = PDFParser()

        # 소스 타입 판별
        if source.startswith("http"):
            logger.info(f"URL에서 PDF 다운로드: {source}")
            pdf_bytes = parser.download_pdf(source)
        elif source.replace(".", "").replace("/", "").isdigit():
            # ArXiv ID 형식 (예: 1706.03762)
            logger.info(f"ArXiv ID에서 PDF 다운로드: {source}")
            url = parser.arxiv_id_to_pdf_url(source)
            pdf_bytes = parser.download_pdf(url)
        else:
            logger.info(f"로컬 파일 로드: {source}")
            pdf_bytes = parser.load_pdf(source)

        state["pdf_bytes"] = pdf_bytes
        logger.info(f"PDF 로드 완료: {len(pdf_bytes):,} bytes")

    except Exception as e:
        logger.error(f"PDF 로드 실패: {e}")
        state["status"] = "failed"
        state["error"] = f"PDF 로드 실패: {str(e)}"

    return state


def parse_pdf(state: TranslationState) -> TranslationState:
    """
    PDF 파싱 노드

    - 텍스트 추출
    - 표 추출
    - 섹션 감지
    """
    state["current_node"] = "parse_pdf"

    # 이전 단계 에러 체크
    if state.get("status") == "failed":
        return state

    try:
        pdf_bytes = state["pdf_bytes"]
        config = state.get("config", {})

        parser = PDFParser()

        # PyPDF2로 텍스트 추출
        raw_text, total_pages = parser.extract_text_pypdf2(pdf_bytes)
        logger.info(f"텍스트 추출 완료: {total_pages} 페이지")

        # 표 추출 (옵션)
        tables = []
        if config.get("extract_tables", True):
            tables = parser.extract_tables_pdfplumber(pdf_bytes)
            logger.info(f"표 추출 완료: {len(tables)}개 표")

        # 섹션 감지
        sections = parser.detect_sections(raw_text)

        # References 제외 (옵션)
        if config.get("exclude_references", True):
            sections = parser.filter_references(sections)
            logger.info(f"References 제외 후 섹션: {len(sections)}개")

        # 섹션 내용 정리
        for section in sections:
            section.content = parser.clean_text(section.content)

        # 제목 추출
        title = "Unknown"
        clean_raw = parser.clean_text(raw_text)
        for line in clean_raw.split("\n"):
            line = line.strip()
            if line and len(line) > 10 and not line.startswith("["):
                title = line
                break

        # ArXiv ID 추출
        arxiv_id = parser.extract_arxiv_id(state["source"])

        # ParsedPaper 생성
        parsed_paper = ParsedPaper(
            title=title,
            raw_text=parser.clean_text(raw_text),
            sections=sections,
            tables=tables,
            total_pages=total_pages,
            source_path=state["source"],
            arxiv_id=arxiv_id
        )

        state["parsed_paper"] = parsed_paper

        # 메타데이터 업데이트
        state["metadata"] = {
            "title": title,
            "arxiv_id": arxiv_id,
            "total_pages": total_pages,
            "source_path": state["source"],
            "domain": config.get("domain", "General"),
        }

        logger.info(f"파싱 완료: '{title[:50]}...' ({len(sections)}개 섹션)")

    except Exception as e:
        logger.error(f"PDF 파싱 실패: {e}")
        state["status"] = "failed"
        state["error"] = f"PDF 파싱 실패: {str(e)}"

    return state


def chunk_text(state: TranslationState) -> TranslationState:
    """
    텍스트 청킹 노드

    - 섹션 기반 + 토큰 기반 하이브리드 청킹
    """
    state["current_node"] = "chunk_text"

    if state.get("status") == "failed":
        return state

    try:
        parsed_paper = state["parsed_paper"]
        config = state.get("config", {})

        chunker = TextChunker(
            max_chunk_tokens=config.get("max_chunk_tokens", 800),
            overlap_tokens=config.get("overlap_tokens", 100),
            strategy=config.get("chunking_strategy", "hybrid")
        )

        chunks = chunker.chunk_paper(parsed_paper)
        state["chunks"] = chunks

        # 청크 요약
        summary = chunker.get_chunk_summary(chunks)
        logger.info(
            f"청킹 완료: {summary['total_chunks']}개 청크, "
            f"{summary['total_tokens']:,} 토큰, "
            f"{summary['unique_sections']}개 섹션"
        )

    except Exception as e:
        logger.error(f"청킹 실패: {e}")
        state["status"] = "failed"
        state["error"] = f"청킹 실패: {str(e)}"

    return state


def pre_process(state: TranslationState) -> TranslationState:
    """
    전처리 노드

    - 용어 매칭
    - 용어 프롬프트 생성
    """
    state["current_node"] = "pre_process"

    if state.get("status") == "failed":
        return state

    try:
        chunks = state["chunks"]
        config = state.get("config", {})
        metadata = state.get("metadata", {})

        preprocessor = PreProcessor(
            pre_process_limit=config.get("pre_process_limit", 20),
            domain=config.get("domain")
        )

        paper_title = metadata.get("title", "")
        processed_chunks = preprocessor.process_chunks(chunks, paper_title=paper_title)
        state["processed_chunks"] = processed_chunks

        # 전처리 요약
        summary = preprocessor.get_processing_summary(processed_chunks)
        logger.info(
            f"전처리 완료: {summary['chunks_with_terms']}/{summary['total_chunks']} 청크에 용어 적용, "
            f"고유 용어 {summary['unique_terms']}개"
        )

    except Exception as e:
        logger.error(f"전처리 실패: {e}")
        state["status"] = "failed"
        state["error"] = f"전처리 실패: {str(e)}"

    return state


def translate_chunks(state: TranslationState) -> TranslationState:
    """
    번역 노드

    - LLM으로 각 청크 번역
    - 토큰 사용량 추적
    """
    state["current_node"] = "translate_chunks"

    if state.get("status") == "failed":
        return state

    try:
        processed_chunks = state["processed_chunks"]
        config = state.get("config", {})

        # 진행 상황 콜백
        def progress_callback(current: int, total: int, message: str):
            logger.info(message)

        translator = Translator(
            temperature=config.get("temperature", 0.1),
            progress_callback=progress_callback
        )

        translated_chunks = translator.translate_chunks(processed_chunks)
        state["translated_chunks"] = translated_chunks

        # 번역 통계
        stats = translator.get_stats_summary()
        state["stats"] = {
            "total_chunks": stats["total_chunks"],
            "completed_chunks": stats["completed"],
            "failed_chunks": stats["failed"],
            "total_input_tokens": stats["input_tokens"],
            "total_output_tokens": stats["output_tokens"],
            "total_tokens": stats["total_tokens"],
            "total_time_sec": stats["total_time_sec"],
            "estimated_cost_usd": stats["estimated_cost_usd"],
        }

        logger.info(
            f"번역 완료: {stats['completed']}/{stats['total_chunks']} 청크, "
            f"{stats['total_tokens']:,} 토큰, "
            f"{stats['estimated_cost_usd']}"
        )

    except Exception as e:
        logger.error(f"번역 실패: {e}")
        state["status"] = "failed"
        state["error"] = f"번역 실패: {str(e)}"

    return state


def post_process(state: TranslationState) -> TranslationState:
    """
    후처리 노드

    - 용어 검증
    - 불일치 용어 교정
    """
    state["current_node"] = "post_process"

    if state.get("status") == "failed":
        return state

    try:
        translated_chunks = state["translated_chunks"]
        config = state.get("config", {})

        postprocessor = PostProcessor(
            threshold=config.get("threshold", 0.8),
            auto_correct=config.get("auto_correct", True),
            log_corrections=True
        )

        post_processed_chunks = postprocessor.process_chunks(translated_chunks)
        state["post_processed_chunks"] = post_processed_chunks

        # 후처리 통계
        pp_stats = postprocessor.get_stats_summary()
        state["stats"]["term_match_rate"] = pp_stats["match_rate"]
        state["stats"]["corrections_made"] = pp_stats["corrections_made"]

        logger.info(
            f"후처리 완료: 매칭률 {pp_stats['match_rate']}, "
            f"교정 {pp_stats['corrections_made']}건"
        )

    except Exception as e:
        logger.error(f"후처리 실패: {e}")
        state["status"] = "failed"
        state["error"] = f"후처리 실패: {str(e)}"

    return state


def generate_markdown(state: TranslationState) -> TranslationState:
    """
    마크다운 생성 노드

    - 청크들을 논문 구조로 변환
    - Jinja2 템플릿으로 렌더링
    """
    state["current_node"] = "generate_markdown"

    if state.get("status") == "failed":
        return state

    try:
        post_processed_chunks = state["post_processed_chunks"]
        metadata = state.get("metadata", {})
        stats = state.get("stats", {})

        writer = MarkdownWriter()

        # 메타데이터에 통계 정보 추가
        paper_metadata = {
            **metadata,
            "total_tokens": stats.get("total_tokens", 0),
            "estimated_cost": stats.get("estimated_cost_usd", "N/A"),
            "term_match_rate": stats.get("term_match_rate", "N/A"),
        }

        # 청크를 논문 구조로 변환
        paper = writer.chunks_to_paper(post_processed_chunks, paper_metadata)

        # 마크다운 렌더링
        markdown_content = writer.render(paper)
        state["markdown_content"] = markdown_content

        logger.info(f"마크다운 생성 완료: {len(markdown_content):,} 문자")

    except Exception as e:
        logger.error(f"마크다운 생성 실패: {e}")
        state["status"] = "failed"
        state["error"] = f"마크다운 생성 실패: {str(e)}"

    return state


def save_output(state: TranslationState) -> TranslationState:
    """
    출력 저장 노드

    - 마크다운 파일 저장
    - MD5 해시 계산
    """
    state["current_node"] = "save_output"

    if state.get("status") == "failed":
        return state

    try:
        markdown_content = state["markdown_content"]
        metadata = state.get("metadata", {})

        writer = MarkdownWriter()

        # 파일명 생성
        filename = writer.generate_filename(
            title=metadata.get("title_ko") or metadata.get("title", "untitled"),
            arxiv_id=metadata.get("arxiv_id")
        )

        # 저장
        file_path = writer.save(markdown_content, filename)

        # 해시 계산
        md5_hash = writer.calculate_md5(markdown_content)

        state["output"] = {
            "file_path": str(file_path),
            "filename": filename,
            "md5_hash": md5_hash,
            "content_length": len(markdown_content),
            "rendered_at": datetime.now().isoformat(),
        }

        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()

        logger.info(f"파일 저장 완료: {file_path}")

    except Exception as e:
        logger.error(f"파일 저장 실패: {e}")
        state["status"] = "failed"
        state["error"] = f"파일 저장 실패: {str(e)}"

    return state


# === 그래프 빌더 ===

def create_translation_graph() -> StateGraph:
    """
    번역 워크플로우 그래프 생성

    노드 흐름:
    fetch_pdf → parse_pdf → chunk_text → pre_process →
    translate_chunks → post_process → generate_markdown → save_output

    Returns:
        컴파일된 StateGraph
    """
    # 그래프 생성
    workflow = StateGraph(TranslationState)

    # 노드 추가
    workflow.add_node("fetch_pdf", fetch_pdf)
    workflow.add_node("parse_pdf", parse_pdf)
    workflow.add_node("chunk_text", chunk_text)
    workflow.add_node("pre_process", pre_process)
    workflow.add_node("translate_chunks", translate_chunks)
    workflow.add_node("post_process", post_process)
    workflow.add_node("generate_markdown", generate_markdown)
    workflow.add_node("save_output", save_output)

    # 엣지 연결 (순차 실행)
    workflow.add_edge("fetch_pdf", "parse_pdf")
    workflow.add_edge("parse_pdf", "chunk_text")
    workflow.add_edge("chunk_text", "pre_process")
    workflow.add_edge("pre_process", "translate_chunks")
    workflow.add_edge("translate_chunks", "post_process")
    workflow.add_edge("post_process", "generate_markdown")
    workflow.add_edge("generate_markdown", "save_output")
    workflow.add_edge("save_output", END)

    # 시작 노드 설정
    workflow.set_entry_point("fetch_pdf")

    return workflow


def compile_graph():
    """
    그래프 컴파일

    Returns:
        실행 가능한 CompiledGraph
    """
    workflow = create_translation_graph()
    return workflow.compile()


# === 실행 함수 ===

def run_translation(
    source: str,
    domain: str = "General",
    exclude_references: bool = True,
    extract_tables: bool = True,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    **kwargs
) -> TranslationState:
    """
    번역 파이프라인 실행

    Args:
        source: PDF URL, ArXiv ID, 또는 로컬 파일 경로
        domain: 용어 도메인 (NLP, CV, RL, General)
        exclude_references: References 섹션 제외 여부
        extract_tables: 표 추출 여부
        progress_callback: 진행 상황 콜백 (node_name, status)
        **kwargs: 추가 설정

    Returns:
        최종 상태 (TranslationState)
    """
    # 초기 상태 생성
    initial_state = create_initial_state(
        source=source,
        domain=domain,
        exclude_references=exclude_references,
        extract_tables=extract_tables,
        **kwargs
    )

    # 그래프 컴파일 및 실행
    app = compile_graph()

    logger.info(f"번역 파이프라인 시작: {source}")

    # 스트리밍 실행
    final_state = None
    for output in app.stream(initial_state):
        for node_name, state in output.items():
            final_state = state
            current_node = state.get("current_node", "")
            status = state.get("status", "")

            if progress_callback:
                progress_callback(current_node, status)

            if status == "failed":
                logger.error(f"파이프라인 실패 at {current_node}: {state.get('error')}")
                return state

    logger.info("번역 파이프라인 완료")
    return final_state


def run_translation_sync(
    source: str,
    **kwargs
) -> dict:
    """
    동기 방식 번역 실행 (간편 인터페이스)

    Args:
        source: PDF URL, ArXiv ID, 또는 로컬 파일 경로
        **kwargs: 추가 설정

    Returns:
        결과 딕셔너리
    """
    state = run_translation(source, **kwargs)

    if state.get("status") == "failed":
        return {
            "success": False,
            "error": state.get("error"),
            "state_summary": get_state_summary(state)
        }

    return {
        "success": True,
        "output": state.get("output"),
        "stats": state.get("stats"),
        "metadata": state.get("metadata"),
        "state_summary": get_state_summary(state)
    }


# 편의 함수
def translate_paper(source: str, **kwargs) -> dict:
    """논문 번역 (단축 함수)"""
    return run_translation_sync(source, **kwargs)


def translate_arxiv(arxiv_id: str, **kwargs) -> dict:
    """ArXiv 논문 번역 (단축 함수)"""
    return run_translation_sync(arxiv_id, **kwargs)
