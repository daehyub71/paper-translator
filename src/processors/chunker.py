"""
하이브리드 청커 모듈
섹션 기반 + 토큰 기반 청킹 지원
"""
import re
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

from src.parsers import ParsedPaper, ParsedSection
from src.utils import settings


@dataclass
class Chunk:
    """청크 데이터 클래스"""
    index: int                          # 전체 청크 내 순서
    content: str                        # 청크 텍스트
    section_title: str                  # 소속 섹션 제목
    section_index: int                  # 섹션 내 청크 순서
    token_count: int                    # 토큰 수
    start_char: int                     # 원문 내 시작 위치
    end_char: int                       # 원문 내 끝 위치
    has_overlap: bool = False           # 오버랩 포함 여부
    tables: list[str] = field(default_factory=list)  # 포함된 표


class TextChunker:
    """하이브리드 텍스트 청커"""

    # 섹션 헤더 패턴 (우선순위 순)
    SECTION_PATTERNS = [
        # 번호가 붙은 섹션 (1. Introduction, 2.1 Method)
        r"^(\d+(?:\.\d+)*)\s+([A-Z][^.!?\n]{2,}?)$",
        # 대문자로 시작하는 단독 라인
        r"^((?:Abstract|Introduction|Background|Related Work|Method(?:s|ology)?|"
        r"Experiment(?:s)?|Results?|Discussion|Conclusion(?:s)?|"
        r"Acknowledgment(?:s)?|References|Appendix))\s*$",
        # 볼드/헤더 형식 (## Header)
        r"^#+\s+(.+)$",
    ]

    def __init__(
        self,
        max_chunk_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
        strategy: Optional[str] = None
    ):
        """
        Args:
            max_chunk_tokens: 최대 청크 토큰 수 (기본: settings에서)
            overlap_tokens: 오버랩 토큰 수 (기본: settings에서)
            strategy: 청킹 전략 ('section' | 'token' | 'hybrid')
        """
        self.max_chunk_tokens = max_chunk_tokens or settings.max_chunk_tokens
        self.overlap_tokens = overlap_tokens or settings.overlap_tokens
        self.strategy = strategy or settings.chunking_strategy

        # tiktoken 인코딩 초기화
        try:
            self._encoding = tiktoken.encoding_for_model(settings.openai_model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return len(self._encoding.encode(text))

    def detect_section_headers(self, text: str) -> list[tuple[int, str]]:
        """
        텍스트에서 섹션 헤더 감지

        Returns:
            list of (position, header_title)
        """
        headers = []
        lines = text.split("\n")
        position = 0

        for line in lines:
            stripped = line.strip()
            if stripped:
                for pattern in self.SECTION_PATTERNS:
                    match = re.match(pattern, stripped, re.IGNORECASE)
                    if match:
                        # 그룹이 있으면 마지막 그룹 (제목), 없으면 전체 매치
                        title = match.group(match.lastindex) if match.lastindex else match.group(0)
                        headers.append((position, title.strip()))
                        break

            position += len(line) + 1  # +1 for newline

        return headers

    def split_by_tokens(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> list[str]:
        """
        토큰 기반 텍스트 분할

        Args:
            text: 분할할 텍스트
            max_tokens: 최대 토큰 수
            overlap: 오버랩 토큰 수

        Returns:
            분할된 텍스트 리스트
        """
        max_tokens = max_tokens or self.max_chunk_tokens
        overlap = overlap or self.overlap_tokens

        tokens = self._encoding.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._encoding.decode(chunk_tokens)

            # 문장 경계에서 자르기 시도
            if end < len(tokens):
                # 마지막 문장 끝 찾기
                last_sentence_end = self._find_sentence_boundary(chunk_text)
                if last_sentence_end > len(chunk_text) * 0.5:  # 50% 이상이면 자르기
                    chunk_text = chunk_text[:last_sentence_end]
                    # 실제 토큰 수 재계산
                    chunk_tokens = self._encoding.encode(chunk_text)

            chunks.append(chunk_text)

            # 다음 시작점 계산 (오버랩 적용)
            # 최소한 1 토큰은 전진해야 무한 루프 방지
            step = max(1, len(chunk_tokens) - overlap)
            start = start + step

            # 끝에 도달하면 종료
            if start >= len(tokens):
                break

        return chunks

    def _find_sentence_boundary(self, text: str) -> int:
        """문장 경계 위치 찾기"""
        # 뒤에서부터 문장 종결 부호 찾기
        sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]

        best_pos = len(text)
        for ending in sentence_endings:
            pos = text.rfind(ending)
            if pos > 0 and pos < best_pos:
                best_pos = pos + len(ending)

        return best_pos if best_pos < len(text) else len(text)

    def chunk_section(
        self,
        section: ParsedSection,
        section_chunk_start_index: int = 0
    ) -> list[Chunk]:
        """
        단일 섹션을 청크로 분할

        Args:
            section: 파싱된 섹션
            section_chunk_start_index: 전체 청크 내 시작 인덱스

        Returns:
            Chunk 리스트
        """
        content = section.content
        token_count = self.count_tokens(content)

        # 토큰 수가 제한 이하면 그대로 반환
        if token_count <= self.max_chunk_tokens:
            return [Chunk(
                index=section_chunk_start_index,
                content=content,
                section_title=section.title,
                section_index=0,
                token_count=token_count,
                start_char=0,
                end_char=len(content),
                has_overlap=False,
                tables=section.tables
            )]

        # 토큰 기반 분할
        text_chunks = self.split_by_tokens(content)
        chunks = []
        char_pos = 0

        for i, text in enumerate(text_chunks):
            chunk = Chunk(
                index=section_chunk_start_index + i,
                content=text,
                section_title=section.title,
                section_index=i,
                token_count=self.count_tokens(text),
                start_char=char_pos,
                end_char=char_pos + len(text),
                has_overlap=(i > 0),  # 첫 번째 청크 외에는 오버랩 있음
                tables=section.tables if i == 0 else []  # 표는 첫 청크에만
            )
            chunks.append(chunk)

            # 다음 청크 시작 위치 (오버랩 고려)
            if i < len(text_chunks) - 1:
                # 오버랩 토큰에 해당하는 문자 수 추정
                overlap_chars = int(len(text) * self.overlap_tokens / self.count_tokens(text))
                char_pos += len(text) - overlap_chars

        return chunks

    def chunk_paper(self, paper: ParsedPaper) -> list[Chunk]:
        """
        전체 논문을 청크로 분할 (하이브리드 전략)

        Args:
            paper: 파싱된 논문

        Returns:
            Chunk 리스트
        """
        if self.strategy == "token":
            return self._chunk_by_tokens_only(paper)
        elif self.strategy == "section":
            return self._chunk_by_sections_only(paper)
        else:  # hybrid
            return self._chunk_hybrid(paper)

    def _chunk_by_tokens_only(self, paper: ParsedPaper) -> list[Chunk]:
        """순수 토큰 기반 청킹"""
        text_chunks = self.split_by_tokens(paper.raw_text)
        chunks = []
        char_pos = 0

        for i, text in enumerate(text_chunks):
            chunk = Chunk(
                index=i,
                content=text,
                section_title="Full Document",
                section_index=i,
                token_count=self.count_tokens(text),
                start_char=char_pos,
                end_char=char_pos + len(text),
                has_overlap=(i > 0),
                tables=paper.tables if i == 0 else []
            )
            chunks.append(chunk)
            char_pos += len(text)

        return chunks

    def _chunk_by_sections_only(self, paper: ParsedPaper) -> list[Chunk]:
        """순수 섹션 기반 청킹 (토큰 제한 무시)"""
        chunks = []
        chunk_index = 0
        char_pos = 0

        for section in paper.sections:
            chunk = Chunk(
                index=chunk_index,
                content=section.content,
                section_title=section.title,
                section_index=0,
                token_count=self.count_tokens(section.content),
                start_char=char_pos,
                end_char=char_pos + len(section.content),
                has_overlap=False,
                tables=section.tables
            )
            chunks.append(chunk)
            chunk_index += 1
            char_pos += len(section.content)

        return chunks

    def _chunk_hybrid(self, paper: ParsedPaper) -> list[Chunk]:
        """하이브리드 청킹 (섹션 + 토큰)"""
        chunks = []
        chunk_index = 0

        for section in paper.sections:
            section_chunks = self.chunk_section(section, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def get_chunk_summary(self, chunks: list[Chunk]) -> dict:
        """청크 요약 정보"""
        total_tokens = sum(c.token_count for c in chunks)
        sections = set(c.section_title for c in chunks)

        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": total_tokens // len(chunks) if chunks else 0,
            "unique_sections": len(sections),
            "sections": list(sections),
            "chunks_with_overlap": sum(1 for c in chunks if c.has_overlap),
            "chunks_with_tables": sum(1 for c in chunks if c.tables),
        }


# 편의 함수
def chunk_text(
    text: str,
    max_tokens: Optional[int] = None,
    overlap: Optional[int] = None
) -> list[str]:
    """텍스트를 토큰 기반으로 분할 (단축 함수)"""
    chunker = TextChunker(max_tokens, overlap)
    return chunker.split_by_tokens(text, max_tokens, overlap)


def chunk_sections(
    paper: ParsedPaper,
    strategy: str = "hybrid"
) -> list[Chunk]:
    """논문을 섹션 기반으로 청킹 (단축 함수)"""
    chunker = TextChunker(strategy=strategy)
    return chunker.chunk_paper(paper)
