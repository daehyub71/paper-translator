"""
Markdown Writer 모듈
번역 결과를 마크다운 파일로 출력
"""
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.utils import settings
from src.processors import PostProcessedChunk


# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class TranslatedSection:
    """번역된 섹션"""
    title: str                      # 원문 섹션 제목
    title_ko: str = ""              # 한글 섹션 제목
    content: str = ""               # 번역된 내용
    tables: list[str] = field(default_factory=list)  # 포함된 표


@dataclass
class TranslatedPaper:
    """번역된 논문 데이터"""
    title: str                      # 원문 제목
    title_ko: str                   # 한글 제목
    arxiv_id: Optional[str] = None  # ArXiv ID
    authors: list[str] = field(default_factory=list)  # 저자 목록
    domain: str = "General"         # 분야
    abstract: str = ""              # 번역된 초록
    sections: list[TranslatedSection] = field(default_factory=list)  # 번역된 섹션들
    total_pages: int = 0            # 원본 페이지 수
    total_chunks: int = 0           # 청크 수
    total_tokens: int = 0           # 총 토큰 수
    estimated_cost: str = ""        # 예상 비용
    term_match_rate: str = ""       # 용어 매칭률
    terminology_used: list[dict] = field(default_factory=list)  # 사용된 용어
    translated_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))


class MarkdownWriter:
    """마크다운 출력 생성기"""

    # 섹션 제목 한글 매핑
    SECTION_TITLE_MAP = {
        "abstract": "초록",
        "introduction": "서론",
        "related work": "관련 연구",
        "background": "배경",
        "method": "방법",
        "methods": "방법",
        "methodology": "방법론",
        "approach": "접근법",
        "model": "모델",
        "architecture": "아키텍처",
        "experiment": "실험",
        "experiments": "실험",
        "results": "결과",
        "result": "결과",
        "discussion": "논의",
        "conclusion": "결론",
        "conclusions": "결론",
        "acknowledgment": "감사의 글",
        "acknowledgments": "감사의 글",
        "appendix": "부록",
        "main content": "본문",
    }

    def __init__(
        self,
        output_dir: Optional[str] = None,
        filename_format: Optional[str] = None,
        template_name: str = "paper_template.md.j2"
    ):
        """
        Args:
            output_dir: 출력 디렉토리 (기본: settings에서)
            filename_format: 파일명 형식 (기본: settings에서)
            template_name: 템플릿 파일명
        """
        self.output_dir = Path(output_dir or settings.output_directory)
        self.filename_format = filename_format or settings.filename_format
        self.template_name = template_name

        # Jinja2 환경 설정
        template_dir = PROJECT_ROOT / "templates"
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # 커스텀 필터 등록
        self._env.filters["slugify"] = self._slugify

    def _slugify(self, text: str) -> str:
        """URL-safe 슬러그 생성"""
        # 한글, 영문, 숫자만 유지
        text = re.sub(r"[^\w가-힣\s-]", "", text)
        # 공백을 하이픈으로
        text = re.sub(r"\s+", "-", text)
        # 연속 하이픈 제거
        text = re.sub(r"-+", "-", text)
        return text.strip("-").lower()

    def translate_section_title(self, title: str) -> str:
        """섹션 제목 한글화"""
        # 숫자 제거 후 매핑 검색
        clean_title = re.sub(r"^\d+\.?\s*", "", title).strip()
        title_lower = clean_title.lower()

        for eng, kor in self.SECTION_TITLE_MAP.items():
            if eng in title_lower:
                # 원래 번호 유지
                number_match = re.match(r"^(\d+\.?\s*)", title)
                prefix = number_match.group(1) if number_match else ""
                return f"{prefix}{kor}"

        return title  # 매핑 없으면 원본 반환

    def calculate_md5(self, content: str) -> str:
        """
        콘텐츠의 MD5 해시 계산

        Args:
            content: 해시 대상 문자열

        Returns:
            MD5 해시 문자열
        """
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def generate_filename(
        self,
        title: str,
        date: Optional[str] = None,
        arxiv_id: Optional[str] = None
    ) -> str:
        """
        파일명 생성

        Args:
            title: 논문 제목
            date: 날짜 (기본: 오늘)
            arxiv_id: ArXiv ID

        Returns:
            생성된 파일명 (확장자 제외)
        """
        date = date or datetime.now().strftime("%Y%m%d")

        # 제목 정리
        clean_title = self._slugify(title)[:50]  # 최대 50자

        # 형식 적용
        filename = self.filename_format
        filename = filename.replace("{date}", date)
        filename = filename.replace("{title}", clean_title)

        if arxiv_id:
            filename = filename.replace("{arxiv_id}", arxiv_id)
        else:
            filename = filename.replace("{arxiv_id}", "")
            filename = filename.replace("__", "_")

        # 파일명 정리
        filename = re.sub(r"[^\w가-힣\-_]", "", filename)
        filename = re.sub(r"_+", "_", filename)

        return filename.strip("_")

    def chunks_to_paper(
        self,
        processed_chunks: list[PostProcessedChunk],
        paper_metadata: dict
    ) -> TranslatedPaper:
        """
        후처리된 청크들을 TranslatedPaper로 변환

        Args:
            processed_chunks: 후처리된 청크 리스트
            paper_metadata: 논문 메타데이터

        Returns:
            TranslatedPaper 객체
        """
        # 섹션별로 그룹화
        sections_dict: dict[str, list[str]] = {}
        sections_tables: dict[str, list[str]] = {}

        for pc in processed_chunks:
            chunk = pc.translated_chunk.processed_chunk.chunk
            section_title = chunk.section_title

            if section_title not in sections_dict:
                sections_dict[section_title] = []
                sections_tables[section_title] = []

            sections_dict[section_title].append(pc.corrected_text)
            sections_tables[section_title].extend(chunk.tables)

        # TranslatedSection 생성
        sections = []
        for title, contents in sections_dict.items():
            section = TranslatedSection(
                title=title,
                title_ko=self.translate_section_title(title),
                content="\n\n".join(contents),
                tables=sections_tables.get(title, [])
            )
            sections.append(section)

        # 사용된 용어 수집
        all_terms = []
        seen_sources = set()
        for pc in processed_chunks:
            for term in pc.translated_chunk.processed_chunk.matched_terms:
                source = term.get("source_text", "")
                if source and source not in seen_sources:
                    all_terms.append({
                        "source": source,
                        "target": term.get("target_text", "")
                    })
                    seen_sources.add(source)

        return TranslatedPaper(
            title=paper_metadata.get("title", "Unknown"),
            title_ko=paper_metadata.get("title_ko", paper_metadata.get("title", "알 수 없음")),
            arxiv_id=paper_metadata.get("arxiv_id"),
            authors=paper_metadata.get("authors", []),
            domain=paper_metadata.get("domain", "General"),
            abstract=paper_metadata.get("abstract_ko", ""),
            sections=sections,
            total_pages=paper_metadata.get("total_pages", 0),
            total_chunks=len(processed_chunks),
            total_tokens=paper_metadata.get("total_tokens", 0),
            estimated_cost=paper_metadata.get("estimated_cost", ""),
            term_match_rate=paper_metadata.get("term_match_rate", ""),
            terminology_used=all_terms
        )

    def render(self, paper: TranslatedPaper) -> str:
        """
        TranslatedPaper를 마크다운으로 렌더링

        Args:
            paper: 번역된 논문 데이터

        Returns:
            마크다운 문자열
        """
        template = self._env.get_template(self.template_name)

        return template.render(
            title=paper.title,
            title_ko=paper.title_ko,
            arxiv_id=paper.arxiv_id,
            authors=paper.authors,
            domain=paper.domain,
            abstract=paper.abstract,
            sections=paper.sections,
            total_pages=paper.total_pages,
            total_chunks=paper.total_chunks,
            total_tokens=paper.total_tokens,
            estimated_cost=paper.estimated_cost,
            term_match_rate=paper.term_match_rate,
            terminology_used=paper.terminology_used,
            translated_date=paper.translated_date
        )

    def save(
        self,
        content: str,
        filename: str,
        create_dir: bool = True
    ) -> Path:
        """
        마크다운 파일 저장

        Args:
            content: 마크다운 내용
            filename: 파일명 (확장자 제외)
            create_dir: 디렉토리 자동 생성 여부

        Returns:
            저장된 파일 경로
        """
        if create_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # 확장자 추가
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        file_path = self.output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return file_path

    def write_paper(
        self,
        processed_chunks: list[PostProcessedChunk],
        paper_metadata: dict,
        filename: Optional[str] = None
    ) -> dict:
        """
        번역된 논문을 마크다운 파일로 저장

        Args:
            processed_chunks: 후처리된 청크 리스트
            paper_metadata: 논문 메타데이터
            filename: 파일명 (None이면 자동 생성)

        Returns:
            {
                "file_path": Path,
                "filename": str,
                "md5_hash": str,
                "content_length": int
            }
        """
        # 데이터 변환
        paper = self.chunks_to_paper(processed_chunks, paper_metadata)

        # 렌더링
        content = self.render(paper)

        # 파일명 생성
        if filename is None:
            filename = self.generate_filename(
                title=paper.title_ko or paper.title,
                arxiv_id=paper.arxiv_id
            )

        # 저장
        file_path = self.save(content, filename)

        # 해시 계산
        md5_hash = self.calculate_md5(content)

        return {
            "file_path": file_path,
            "filename": filename,
            "md5_hash": md5_hash,
            "content_length": len(content)
        }

    def render_simple(
        self,
        title: str,
        sections: list[dict],
        metadata: Optional[dict] = None
    ) -> str:
        """
        간단한 마크다운 렌더링 (템플릿 없이)

        Args:
            title: 제목
            sections: [{"title": str, "content": str}, ...]
            metadata: 추가 메타데이터

        Returns:
            마크다운 문자열
        """
        lines = [f"# {title}\n"]

        if metadata:
            if metadata.get("arxiv_id"):
                lines.append(f"> ArXiv: {metadata['arxiv_id']}\n")
            if metadata.get("translated_date"):
                lines.append(f"> 번역일: {metadata['translated_date']}\n")
            lines.append("")

        lines.append("---\n")

        for section in sections:
            title = section.get("title", "")
            content = section.get("content", "")
            lines.append(f"## {title}\n")
            lines.append(f"{content}\n")

        lines.append("---\n")
        lines.append("*Paper Translator로 자동 번역됨*")

        return "\n".join(lines)


# 편의 함수
def render_markdown(
    processed_chunks: list[PostProcessedChunk],
    paper_metadata: dict
) -> str:
    """마크다운 렌더링 (단축 함수)"""
    writer = MarkdownWriter()
    paper = writer.chunks_to_paper(processed_chunks, paper_metadata)
    return writer.render(paper)


def save_markdown(
    content: str,
    filename: str,
    output_dir: Optional[str] = None
) -> Path:
    """마크다운 저장 (단축 함수)"""
    writer = MarkdownWriter(output_dir=output_dir)
    return writer.save(content, filename)


def write_translated_paper(
    processed_chunks: list[PostProcessedChunk],
    paper_metadata: dict,
    filename: Optional[str] = None
) -> dict:
    """번역 논문 저장 (단축 함수)"""
    writer = MarkdownWriter()
    return writer.write_paper(processed_chunks, paper_metadata, filename)


def calculate_content_hash(content: str) -> str:
    """콘텐츠 MD5 해시 계산 (단축 함수)"""
    writer = MarkdownWriter()
    return writer.calculate_md5(content)
