"""
하이브리드 청커 단위 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from src.parsers.pdf_parser import ParsedPaper, ParsedSection
from src.processors.chunker import (
    TextChunker,
    Chunk,
    chunk_text,
    chunk_sections,
)


class TestTextChunker:
    """TextChunker 클래스 테스트"""

    def setup_method(self):
        """각 테스트 전 실행"""
        self.chunker = TextChunker(
            max_chunk_tokens=500,
            overlap_tokens=50,
            strategy="hybrid"
        )

    # === 토큰 카운트 테스트 ===

    def test_count_tokens_basic(self):
        """기본 토큰 카운트"""
        text = "Hello world"
        count = self.chunker.count_tokens(text)

        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_empty(self):
        """빈 텍스트"""
        count = self.chunker.count_tokens("")
        assert count == 0

    def test_count_tokens_long_text(self):
        """긴 텍스트"""
        text = "word " * 1000
        count = self.chunker.count_tokens(text)

        # 대략 1000 단어 = ~1000 토큰 예상
        assert count > 500

    # === 섹션 헤더 감지 테스트 ===

    def test_detect_section_headers_numbered(self):
        """번호 붙은 섹션 헤더 감지"""
        text = """1 Introduction
Some content here.

2 Method
More content here.

2.1 Subsection
Subsection content.
"""
        headers = self.chunker.detect_section_headers(text)

        assert len(headers) >= 2
        titles = [h[1] for h in headers]
        assert "Introduction" in titles
        assert "Method" in titles

    def test_detect_section_headers_named(self):
        """이름 기반 섹션 헤더 감지"""
        text = """Abstract
This is abstract.

Introduction
This is intro.

Conclusion
This is conclusion.
"""
        headers = self.chunker.detect_section_headers(text)

        assert len(headers) >= 3

    def test_detect_section_headers_markdown(self):
        """마크다운 헤더 감지"""
        text = """# Main Title
Content

## Section One
More content

### Subsection
Even more content
"""
        headers = self.chunker.detect_section_headers(text)

        assert len(headers) >= 2

    def test_detect_section_headers_none(self):
        """헤더 없는 텍스트"""
        text = "Just regular text without any section headers."
        headers = self.chunker.detect_section_headers(text)

        assert len(headers) == 0

    # === 토큰 기반 분할 테스트 ===

    def test_split_by_tokens_short_text(self):
        """짧은 텍스트 - 분할 없음"""
        text = "Short text that fits in one chunk."
        chunks = self.chunker.split_by_tokens(text, max_tokens=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_by_tokens_long_text(self):
        """긴 텍스트 - 여러 청크로 분할"""
        text = "word " * 500  # 약 500+ 토큰
        chunks = self.chunker.split_by_tokens(text, max_tokens=100)

        assert len(chunks) > 1
        # 각 청크가 최대 토큰 수 근처인지 확인
        for chunk in chunks:
            assert self.chunker.count_tokens(chunk) <= 150  # 약간의 여유

    def test_split_by_tokens_overlap(self):
        """오버랩 적용 확인"""
        text = "word " * 300
        chunks = self.chunker.split_by_tokens(text, max_tokens=100, overlap=20)

        # 오버랩이 있으면 청크 간 겹치는 부분이 있어야 함
        assert len(chunks) > 1
        # 총 문자 수가 원본보다 많아야 함 (오버랩 때문에)
        total_chars = sum(len(c) for c in chunks)
        assert total_chars > len(text) * 0.9  # 오버랩으로 인해 더 많음

    def test_split_by_tokens_sentence_boundary(self):
        """문장 경계에서 분할 시도"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = self.chunker.split_by_tokens(text, max_tokens=10)

        # 문장 끝에서 잘릴 가능성 확인
        assert len(chunks) > 1

    # === 문장 경계 찾기 테스트 ===

    def test_find_sentence_boundary_period(self):
        """마침표로 문장 경계 찾기"""
        text = "First sentence. Second sentence."
        pos = self.chunker._find_sentence_boundary(text)

        assert pos > 0
        assert pos <= len(text)

    def test_find_sentence_boundary_question(self):
        """물음표로 문장 경계 찾기"""
        text = "Is this a question? Yes it is."
        pos = self.chunker._find_sentence_boundary(text)

        assert pos > 0

    def test_find_sentence_boundary_none(self):
        """문장 경계 없음"""
        text = "No sentence boundary here"
        pos = self.chunker._find_sentence_boundary(text)

        assert pos == len(text)

    # === 섹션 청킹 테스트 ===

    def test_chunk_section_small(self):
        """작은 섹션 - 분할 없음"""
        section = ParsedSection(
            title="Introduction",
            content="Short introduction text.",
            page_start=1,
            page_end=1
        )

        chunks = self.chunker.chunk_section(section)

        assert len(chunks) == 1
        assert chunks[0].section_title == "Introduction"
        assert chunks[0].content == "Short introduction text."

    def test_chunk_section_large(self):
        """큰 섹션 - 여러 청크로 분할"""
        section = ParsedSection(
            title="Method",
            content="word " * 1000,
            page_start=2,
            page_end=5
        )

        chunks = self.chunker.chunk_section(section)

        assert len(chunks) > 1
        assert all(c.section_title == "Method" for c in chunks)

    def test_chunk_section_preserves_tables(self):
        """섹션 내 표 보존"""
        section = ParsedSection(
            title="Results",
            content="Some results text.",
            page_start=3,
            page_end=4,
            tables=["| A | B |"]
        )

        chunks = self.chunker.chunk_section(section)

        assert chunks[0].tables == ["| A | B |"]

    def test_chunk_section_metadata(self):
        """청크 메타데이터 확인"""
        section = ParsedSection(
            title="Discussion",
            content="Discussion content here.",
            page_start=5,
            page_end=6
        )

        chunks = self.chunker.chunk_section(section, section_chunk_start_index=10)

        assert chunks[0].index == 10
        assert chunks[0].section_index == 0

    # === 논문 전체 청킹 테스트 ===

    def test_chunk_paper_hybrid(self):
        """하이브리드 전략으로 논문 청킹"""
        paper = ParsedPaper(
            title="Test Paper",
            raw_text="Full paper text",
            sections=[
                ParsedSection("Abstract", "Abstract content", 1, 1),
                ParsedSection("Introduction", "Intro content", 2, 3),
            ],
            tables=[],
            total_pages=3,
            source_path="/test.pdf"
        )

        self.chunker.strategy = "hybrid"
        chunks = self.chunker.chunk_paper(paper)

        assert len(chunks) >= 2
        assert any(c.section_title == "Abstract" for c in chunks)
        assert any(c.section_title == "Introduction" for c in chunks)

    def test_chunk_paper_token_only(self):
        """토큰 전략으로 논문 청킹"""
        paper = ParsedPaper(
            title="Test Paper",
            raw_text="word " * 200,
            sections=[],
            tables=[],
            total_pages=2,
            source_path="/test.pdf"
        )

        self.chunker.strategy = "token"
        chunks = self.chunker.chunk_paper(paper)

        # 모든 청크가 "Full Document" 섹션
        assert all(c.section_title == "Full Document" for c in chunks)

    def test_chunk_paper_section_only(self):
        """섹션 전략으로 논문 청킹"""
        paper = ParsedPaper(
            title="Test Paper",
            raw_text="Full text",
            sections=[
                ParsedSection("Abstract", "Abstract content", 1, 1),
                ParsedSection("Conclusion", "Conclusion content", 10, 10),
            ],
            tables=[],
            total_pages=10,
            source_path="/test.pdf"
        )

        self.chunker.strategy = "section"
        chunks = self.chunker.chunk_paper(paper)

        # 섹션 수만큼 청크
        assert len(chunks) == 2
        assert not any(c.has_overlap for c in chunks)

    # === 청크 요약 테스트 ===

    def test_get_chunk_summary(self):
        """청크 요약 정보"""
        chunks = [
            Chunk(0, "Text1", "Intro", 0, 10, 0, 10, False, []),
            Chunk(1, "Text2", "Intro", 1, 15, 10, 20, True, []),
            Chunk(2, "Text3", "Method", 0, 20, 20, 40, False, ["Table1"]),
        ]

        summary = self.chunker.get_chunk_summary(chunks)

        assert summary["total_chunks"] == 3
        assert summary["total_tokens"] == 45
        assert summary["unique_sections"] == 2
        assert summary["chunks_with_overlap"] == 1
        assert summary["chunks_with_tables"] == 1

    def test_get_chunk_summary_empty(self):
        """빈 청크 목록"""
        summary = self.chunker.get_chunk_summary([])

        assert summary["total_chunks"] == 0
        assert summary["avg_tokens_per_chunk"] == 0


class TestConvenienceFunctions:
    """편의 함수 테스트"""

    def test_chunk_text_function(self):
        """chunk_text 단축 함수"""
        text = "word " * 100
        chunks = chunk_text(text, max_tokens=50)

        assert len(chunks) > 1

    def test_chunk_sections_function(self):
        """chunk_sections 단축 함수"""
        paper = ParsedPaper(
            title="Test",
            raw_text="content",
            sections=[
                ParsedSection("Abstract", "Abstract text", 1, 1),
            ],
            tables=[],
            total_pages=1,
            source_path="/test.pdf"
        )

        chunks = chunk_sections(paper, strategy="hybrid")

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)


class TestChunkDataclass:
    """Chunk 데이터클래스 테스트"""

    def test_chunk_creation(self):
        """청크 생성"""
        chunk = Chunk(
            index=0,
            content="Test content",
            section_title="Introduction",
            section_index=0,
            token_count=10,
            start_char=0,
            end_char=12,
            has_overlap=False,
            tables=[]
        )

        assert chunk.index == 0
        assert chunk.content == "Test content"
        assert chunk.section_title == "Introduction"
        assert not chunk.has_overlap

    def test_chunk_default_values(self):
        """청크 기본값"""
        chunk = Chunk(
            index=0,
            content="Text",
            section_title="Intro",
            section_index=0,
            token_count=5,
            start_char=0,
            end_char=4
        )

        assert chunk.has_overlap is False
        assert chunk.tables == []
