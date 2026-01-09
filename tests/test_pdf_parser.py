"""
PDF 파서 단위 테스트
"""
import io
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from src.parsers.pdf_parser import (
    PDFParser,
    ParsedPaper,
    ParsedSection,
    parse_pdf,
    download_arxiv_pdf,
)


class TestPDFParser:
    """PDFParser 클래스 테스트"""

    def setup_method(self):
        """각 테스트 전 실행"""
        self.parser = PDFParser()

    # === ArXiv ID 추출 테스트 ===

    def test_extract_arxiv_id_from_pdf_url(self):
        """PDF URL에서 ArXiv ID 추출"""
        url = "https://arxiv.org/pdf/1706.03762"
        assert PDFParser.extract_arxiv_id(url) == "1706.03762"

    def test_extract_arxiv_id_from_abs_url(self):
        """Abstract URL에서 ArXiv ID 추출"""
        url = "https://arxiv.org/abs/1706.03762"
        assert PDFParser.extract_arxiv_id(url) == "1706.03762"

    def test_extract_arxiv_id_from_ar5iv_url(self):
        """ar5iv URL에서 ArXiv ID 추출"""
        url = "https://ar5iv.labs.arxiv.org/html/1706.03762"
        assert PDFParser.extract_arxiv_id(url) == "1706.03762"

    def test_extract_arxiv_id_direct(self):
        """직접 ID 입력 시 그대로 반환"""
        arxiv_id = "2301.00001"
        assert PDFParser.extract_arxiv_id(arxiv_id) == "2301.00001"

    def test_extract_arxiv_id_invalid(self):
        """유효하지 않은 입력 시 None 반환"""
        assert PDFParser.extract_arxiv_id("invalid_url") is None
        assert PDFParser.extract_arxiv_id("https://google.com") is None

    # === PDF URL 변환 테스트 ===

    def test_arxiv_id_to_pdf_url(self):
        """ArXiv ID를 PDF URL로 변환"""
        arxiv_id = "1706.03762"
        expected = "https://arxiv.org/pdf/1706.03762.pdf"
        assert PDFParser.arxiv_id_to_pdf_url(arxiv_id) == expected

    # === PDF 다운로드 테스트 ===

    @patch("src.parsers.pdf_parser.requests.get")
    def test_download_pdf_success(self, mock_get):
        """PDF 다운로드 성공"""
        # Mock PDF 응답
        mock_response = Mock()
        mock_response.content = b"%PDF-1.4 fake pdf content"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.parser.download_pdf("https://arxiv.org/pdf/1706.03762")

        assert result == b"%PDF-1.4 fake pdf content"
        mock_get.assert_called_once()

    @patch("src.parsers.pdf_parser.requests.get")
    def test_download_pdf_invalid_format(self, mock_get):
        """잘못된 PDF 형식"""
        mock_response = Mock()
        mock_response.content = b"Not a PDF file"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="유효한 PDF 파일이 아닙니다"):
            self.parser.download_pdf("https://example.com/file.pdf")

    @patch("src.parsers.pdf_parser.requests.get")
    def test_download_pdf_http_error(self, mock_get):
        """HTTP 에러 처리"""
        mock_get.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            self.parser.download_pdf("https://arxiv.org/pdf/invalid")

    # === 로컬 PDF 로드 테스트 ===

    def test_load_pdf_file_not_found(self):
        """파일 없음 에러"""
        with pytest.raises(FileNotFoundError):
            self.parser.load_pdf("/nonexistent/path.pdf")

    @patch("builtins.open", create=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_pdf_invalid_format(self, mock_exists, mock_open):
        """잘못된 PDF 형식"""
        mock_open.return_value.__enter__.return_value.read.return_value = b"Not PDF"

        with pytest.raises(ValueError, match="유효한 PDF 파일이 아닙니다"):
            self.parser.load_pdf("/fake/path.pdf")

    # === 텍스트 추출 테스트 ===

    @patch("src.parsers.pdf_parser.PyPDF2.PdfReader")
    def test_extract_text_pypdf2(self, mock_reader):
        """PyPDF2로 텍스트 추출"""
        # Mock PDF reader
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"

        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_reader.return_value = mock_reader_instance

        text, total_pages = self.parser.extract_text_pypdf2(b"%PDF-1.4 fake")

        assert total_pages == 2
        assert "[PAGE 1]" in text
        assert "Page 1 content" in text
        assert "[PAGE 2]" in text
        assert "Page 2 content" in text

    @patch("src.parsers.pdf_parser.PyPDF2.PdfReader")
    def test_extract_text_empty_page(self, mock_reader):
        """빈 페이지 처리"""
        mock_page = Mock()
        mock_page.extract_text.return_value = None

        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        text, total_pages = self.parser.extract_text_pypdf2(b"%PDF-1.4")

        assert total_pages == 1
        assert "[PAGE 1]" in text

    # === 표 추출 테스트 ===

    @patch("src.parsers.pdf_parser.pdfplumber.open")
    def test_extract_tables_pdfplumber(self, mock_open):
        """pdfplumber로 표 추출"""
        # Mock table data
        mock_table = [
            ["Header1", "Header2"],
            ["Data1", "Data2"],
            ["Data3", "Data4"]
        ]

        mock_page = Mock()
        mock_page.extract_tables.return_value = [mock_table]

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_open.return_value = mock_pdf

        tables = self.parser.extract_tables_pdfplumber(b"%PDF-1.4")

        assert len(tables) == 1
        assert "[TABLE Page 1, #1]" in tables[0]
        assert "Header1" in tables[0]
        assert "Data1" in tables[0]

    @patch("src.parsers.pdf_parser.pdfplumber.open")
    def test_extract_tables_empty(self, mock_open):
        """표 없는 경우"""
        mock_page = Mock()
        mock_page.extract_tables.return_value = []

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_open.return_value = mock_pdf

        tables = self.parser.extract_tables_pdfplumber(b"%PDF-1.4")

        assert tables == []

    # === 테이블 마크다운 변환 테스트 ===

    def test_table_to_markdown(self):
        """테이블을 마크다운으로 변환"""
        table = [
            ["A", "B", "C"],
            ["1", "2", "3"],
            ["4", "5", "6"]
        ]

        result = self.parser._table_to_markdown(table, 1, 1)

        assert "[TABLE Page 1, #1]" in result
        assert "| A | B | C |" in result
        assert "|---|---|---|" in result
        assert "| 1 | 2 | 3 |" in result

    def test_table_to_markdown_empty(self):
        """빈 테이블"""
        assert self.parser._table_to_markdown([], 1, 1) == ""
        assert self.parser._table_to_markdown([["A"]], 1, 1) == ""

    # === 섹션 감지 테스트 ===

    def test_detect_sections_abstract(self):
        """Abstract 섹션 감지"""
        text = """[PAGE 1]
Abstract
This is the abstract content.

[PAGE 2]
Introduction
This is the introduction.
"""
        sections = self.parser.detect_sections(text)

        assert len(sections) >= 2
        assert any("Abstract" in s.title for s in sections)
        assert any("Introduction" in s.title for s in sections)

    def test_detect_sections_numbered(self):
        """번호 붙은 섹션 감지"""
        text = """[PAGE 1]
1. Introduction
This is the introduction.

2. Method
This is the method section.
"""
        sections = self.parser.detect_sections(text)

        assert len(sections) >= 2

    def test_detect_sections_none(self):
        """섹션 없는 경우 전체를 하나로"""
        text = "Just some regular text without any section headers."

        sections = self.parser.detect_sections(text)

        assert len(sections) == 1
        assert sections[0].title == "Main Content"

    # === References 필터링 테스트 ===

    def test_filter_references(self):
        """References 섹션 제외"""
        sections = [
            ParsedSection("Abstract", "content", 1, 1),
            ParsedSection("Introduction", "content", 2, 2),
            ParsedSection("References", "ref content", 10, 12),
        ]

        filtered = self.parser.filter_references(sections)

        assert len(filtered) == 2
        assert not any("reference" in s.title.lower() for s in filtered)

    def test_filter_references_bibliography(self):
        """Bibliography 섹션도 제외"""
        sections = [
            ParsedSection("Method", "content", 1, 1),
            ParsedSection("Bibliography", "bib content", 10, 12),
        ]

        filtered = self.parser.filter_references(sections)

        assert len(filtered) == 1

    # === 텍스트 정리 테스트 ===

    def test_clean_text_remove_page_markers(self):
        """페이지 마커 제거"""
        text = "[PAGE 1]\nSome text\n[PAGE 2]\nMore text"
        cleaned = self.parser.clean_text(text)

        assert "[PAGE" not in cleaned
        assert "Some text" in cleaned

    def test_clean_text_multiple_newlines(self):
        """연속 줄바꿈 정리"""
        text = "Line 1\n\n\n\nLine 2"
        cleaned = self.parser.clean_text(text)

        assert "\n\n\n\n" not in cleaned

    def test_clean_text_hyphen_linebreak(self):
        """하이픈 줄바꿈 수정"""
        text = "transfor-\nmer"
        cleaned = self.parser.clean_text(text)

        assert "transformer" in cleaned

    # === LaTeX 보존 테스트 ===

    def test_preserve_latex(self):
        """LaTeX 수식 보존"""
        text = "The equation $E = mc^2$ is famous."
        result = self.parser.preserve_latex(text)

        assert "$E = mc^2$" in result

    # === 전체 파싱 테스트 ===

    @patch.object(PDFParser, "download_pdf")
    @patch.object(PDFParser, "extract_text_pypdf2")
    @patch.object(PDFParser, "extract_tables_pdfplumber")
    def test_parse_from_url(self, mock_tables, mock_text, mock_download):
        """URL에서 파싱"""
        mock_download.return_value = b"%PDF-1.4"
        mock_text.return_value = ("Title Line\n\nAbstract\nContent here", 2)
        mock_tables.return_value = []

        result = self.parser.parse("https://arxiv.org/pdf/1706.03762")

        assert isinstance(result, ParsedPaper)
        assert result.arxiv_id == "1706.03762"
        assert result.total_pages == 2

    @patch.object(PDFParser, "load_pdf")
    @patch.object(PDFParser, "extract_text_pypdf2")
    @patch.object(PDFParser, "extract_tables_pdfplumber")
    def test_parse_from_file(self, mock_tables, mock_text, mock_load):
        """로컬 파일 파싱"""
        mock_load.return_value = b"%PDF-1.4"
        mock_text.return_value = ("Paper Title\n\nAbstract\nPaper content", 5)
        mock_tables.return_value = []

        result = self.parser.parse("/fake/paper.pdf")

        assert isinstance(result, ParsedPaper)
        assert result.arxiv_id is None
        assert result.total_pages == 5

    @patch.object(PDFParser, "download_pdf")
    @patch.object(PDFParser, "extract_text_pypdf2")
    @patch.object(PDFParser, "extract_tables_pdfplumber")
    def test_parse_exclude_references(self, mock_tables, mock_text, mock_download):
        """References 제외 옵션"""
        mock_download.return_value = b"%PDF-1.4"
        mock_text.return_value = (
            "Title\n\n1. Introduction\nIntro content\n\nReferences\n[1] Ref",
            3
        )
        mock_tables.return_value = []

        result = self.parser.parse("https://arxiv.org/pdf/1706.03762", exclude_references=True)

        assert not any("reference" in s.title.lower() for s in result.sections)


class TestConvenienceFunctions:
    """편의 함수 테스트"""

    @patch("src.parsers.pdf_parser.PDFParser.parse")
    def test_parse_pdf_function(self, mock_parse):
        """parse_pdf 단축 함수"""
        mock_parse.return_value = ParsedPaper(
            title="Test",
            raw_text="content",
            sections=[],
            tables=[],
            total_pages=1,
            source_path="/test.pdf"
        )

        result = parse_pdf("/test.pdf")

        assert isinstance(result, ParsedPaper)
        mock_parse.assert_called_once()

    @patch("src.parsers.pdf_parser.PDFParser.download_pdf")
    def test_download_arxiv_pdf_function(self, mock_download):
        """download_arxiv_pdf 단축 함수"""
        mock_download.return_value = b"%PDF-1.4 content"

        result = download_arxiv_pdf("1706.03762")

        assert result == b"%PDF-1.4 content"
