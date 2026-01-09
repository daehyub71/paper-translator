"""
PDF 파서 모듈
ArXiv 논문 PDF 다운로드 및 텍스트 추출
"""
import io
import re
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import requests
import PyPDF2
import pdfplumber


@dataclass
class ParsedSection:
    """파싱된 섹션"""
    title: str
    content: str
    page_start: int
    page_end: int
    tables: list[str] = field(default_factory=list)


@dataclass
class ParsedPaper:
    """파싱된 논문"""
    title: str
    raw_text: str
    sections: list[ParsedSection]
    tables: list[str]
    total_pages: int
    source_path: str
    arxiv_id: Optional[str] = None


class PDFParser:
    """PDF 파서 클래스"""

    # ArXiv PDF URL 패턴
    ARXIV_PDF_PATTERNS = [
        r"arxiv\.org/pdf/(\d+\.\d+)",  # https://arxiv.org/pdf/1706.03762
        r"arxiv\.org/abs/(\d+\.\d+)",  # https://arxiv.org/abs/1706.03762
        r"ar5iv\.labs\.arxiv\.org/html/(\d+\.\d+)",  # ar5iv HTML
    ]

    # 섹션 헤더 패턴
    SECTION_PATTERNS = [
        r"^(?:(\d+\.?\s*)?(Abstract|Introduction|Related Work|Background|Method|Methodology|"
        r"Approach|Model|Architecture|Experiment|Results|Discussion|Conclusion|"
        r"Acknowledgment|References|Appendix|Supplementary)s?)\s*$",
        r"^(\d+\.)\s+([A-Z][a-zA-Z\s]+)$",  # "1. Introduction"
        r"^(ABSTRACT|INTRODUCTION|BACKGROUND|RELATED WORK|METHOD(?:S|OLOGY)?|"
        r"EXPERIMENT(?:S)?|RESULTS?|DISCUSSION|CONCLUSION(?:S)?|ACKNOWLEDGMENT(?:S)?|"
        r"REFERENCES|APPENDIX)\s*$",  # All caps known headers only
    ]

    # LaTeX 수식 패턴 (보존용)
    LATEX_PATTERNS = [
        r"\$[^$]+\$",  # 인라인 수식
        r"\$\$[^$]+\$\$",  # 디스플레이 수식
        r"\\begin\{equation\}.*?\\end\{equation\}",
        r"\\begin\{align\}.*?\\end\{align\}",
    ]

    def __init__(self):
        self._temp_dir = tempfile.mkdtemp()

    @staticmethod
    def extract_arxiv_id(url_or_id: str) -> Optional[str]:
        """URL 또는 ID에서 ArXiv ID 추출"""
        # 이미 ID 형식인 경우
        if re.match(r"^\d+\.\d+$", url_or_id):
            return url_or_id

        # URL에서 추출
        for pattern in PDFParser.ARXIV_PDF_PATTERNS:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)

        return None

    @staticmethod
    def arxiv_id_to_pdf_url(arxiv_id: str) -> str:
        """ArXiv ID를 PDF URL로 변환"""
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    def download_pdf(self, url: str, timeout: int = 60) -> bytes:
        """
        URL에서 PDF 다운로드

        Args:
            url: PDF URL
            timeout: 타임아웃 (초)

        Returns:
            PDF 바이너리 데이터
        """
        # ArXiv URL 정규화
        arxiv_id = self.extract_arxiv_id(url)
        if arxiv_id:
            url = self.arxiv_id_to_pdf_url(arxiv_id)

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PaperTranslator/1.0)"
        }

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # PDF 형식 검증
        if not response.content[:4] == b"%PDF":
            raise ValueError(f"유효한 PDF 파일이 아닙니다: {url}")

        return response.content

    def load_pdf(self, path: str) -> bytes:
        """로컬 PDF 파일 로드"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        with open(path, "rb") as f:
            content = f.read()

        if not content[:4] == b"%PDF":
            raise ValueError(f"유효한 PDF 파일이 아닙니다: {path}")

        return content

    def extract_text_pypdf2(self, pdf_bytes: bytes) -> tuple[str, int]:
        """
        PyPDF2를 사용하여 텍스트 추출

        Returns:
            (추출된 텍스트, 총 페이지 수)
        """
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)

        text_parts = []
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            text_parts.append(f"[PAGE {page_num + 1}]\n{page_text}")

        return "\n\n".join(text_parts), total_pages

    def extract_tables_pdfplumber(self, pdf_bytes: bytes) -> list[str]:
        """
        pdfplumber를 사용하여 표 추출

        Returns:
            표 목록 (마크다운 형식)
        """
        tables = []

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()

                for table_idx, table in enumerate(page_tables):
                    if table and len(table) > 1:
                        # 마크다운 테이블로 변환
                        md_table = self._table_to_markdown(table, page_num + 1, table_idx + 1)
                        if md_table:
                            tables.append(md_table)

        return tables

    def _table_to_markdown(self, table: list[list], page: int, table_num: int) -> str:
        """테이블을 마크다운 형식으로 변환"""
        if not table or len(table) < 2:
            return ""

        # 셀 정리
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)

        # 마크다운 생성
        lines = [f"[TABLE Page {page}, #{table_num}]"]

        # 헤더
        header = cleaned_table[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        # 데이터 행
        for row in cleaned_table[1:]:
            # 컬럼 수 맞추기
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row[:len(header)]) + " |")

        return "\n".join(lines)

    def detect_sections(self, text: str) -> list[ParsedSection]:
        """텍스트에서 섹션 감지"""
        sections = []
        lines = text.split("\n")

        current_section = None
        current_content = []
        current_page = 1

        for line in lines:
            # 페이지 마커 감지
            page_match = re.match(r"\[PAGE (\d+)\]", line)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            # 섹션 헤더 감지
            is_header = False
            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    # 이전 섹션 저장
                    if current_section:
                        current_section.content = "\n".join(current_content).strip()
                        current_section.page_end = current_page
                        sections.append(current_section)

                    # 새 섹션 시작
                    current_section = ParsedSection(
                        title=line.strip(),
                        content="",
                        page_start=current_page,
                        page_end=current_page
                    )
                    current_content = []
                    is_header = True
                    break

            if not is_header and current_section:
                current_content.append(line)

        # 마지막 섹션 저장
        if current_section:
            current_section.content = "\n".join(current_content).strip()
            current_section.page_end = current_page
            sections.append(current_section)

        # 섹션이 감지되지 않은 경우 전체를 하나의 섹션으로
        if not sections:
            sections = [ParsedSection(
                title="Main Content",
                content=text,
                page_start=1,
                page_end=current_page
            )]

        return sections

    def filter_references(self, sections: list[ParsedSection]) -> list[ParsedSection]:
        """References 섹션 제외"""
        filtered = []
        for section in sections:
            title_lower = section.title.lower()
            if "reference" not in title_lower and "bibliography" not in title_lower:
                filtered.append(section)
        return filtered

    def preserve_latex(self, text: str) -> str:
        """LaTeX 수식 보존 (마커로 대체 후 복원용)"""
        # 이 메서드는 실제로는 수식을 그대로 유지합니다.
        # 필요시 마커로 대체하는 로직 추가 가능
        return text

    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 연속된 공백 정리
        text = re.sub(r" +", " ", text)

        # 연속된 줄바꿈 정리
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 페이지 마커 제거
        text = re.sub(r"\[PAGE \d+\]\s*", "", text)

        # 하이픈 줄바꿈 수정 (단어가 줄 끝에서 나뉜 경우)
        text = re.sub(r"-\s*\n\s*", "", text)

        return text.strip()

    def parse(
        self,
        source: str,
        exclude_references: bool = True,
        extract_tables: bool = True
    ) -> ParsedPaper:
        """
        PDF 파싱 메인 함수

        Args:
            source: PDF URL, ArXiv ID, 또는 로컬 파일 경로
            exclude_references: References 섹션 제외 여부
            extract_tables: 표 추출 여부

        Returns:
            ParsedPaper 객체
        """
        # 소스 타입 판별 및 PDF 로드
        arxiv_id = None
        if source.startswith("http"):
            arxiv_id = self.extract_arxiv_id(source)
            pdf_bytes = self.download_pdf(source)
            source_path = source
        elif re.match(r"^\d+\.\d+$", source):
            arxiv_id = source
            pdf_bytes = self.download_pdf(self.arxiv_id_to_pdf_url(source))
            source_path = self.arxiv_id_to_pdf_url(source)
        else:
            pdf_bytes = self.load_pdf(source)
            source_path = source

        # 텍스트 추출
        raw_text, total_pages = self.extract_text_pypdf2(pdf_bytes)

        # 표 추출
        tables = []
        if extract_tables:
            tables = self.extract_tables_pdfplumber(pdf_bytes)

        # 섹션 감지
        sections = self.detect_sections(raw_text)

        # References 제외
        if exclude_references:
            sections = self.filter_references(sections)

        # 섹션 내용 정리
        for section in sections:
            section.content = self.clean_text(section.content)

        # 제목 추출 (첫 번째 비어있지 않은 줄)
        title = "Unknown"
        clean_raw = self.clean_text(raw_text)
        for line in clean_raw.split("\n"):
            line = line.strip()
            if line and len(line) > 10 and not line.startswith("["):
                title = line
                break

        return ParsedPaper(
            title=title,
            raw_text=self.clean_text(raw_text),
            sections=sections,
            tables=tables,
            total_pages=total_pages,
            source_path=source_path,
            arxiv_id=arxiv_id
        )


# 편의 함수
def parse_pdf(source: str, **kwargs) -> ParsedPaper:
    """PDF 파싱 단축 함수"""
    parser = PDFParser()
    return parser.parse(source, **kwargs)


def download_arxiv_pdf(arxiv_id: str) -> bytes:
    """ArXiv PDF 다운로드 단축 함수"""
    parser = PDFParser()
    url = parser.arxiv_id_to_pdf_url(arxiv_id)
    return parser.download_pdf(url)
