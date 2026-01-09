"""
LangGraph 상태 정의 모듈
번역 파이프라인의 상태를 TypedDict로 정의
"""
from typing import TypedDict, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime

from src.parsers import ParsedPaper
from src.processors import (
    Chunk,
    ProcessedChunk,
    TranslatedChunk,
    PostProcessedChunk,
)


class PaperMetadata(TypedDict, total=False):
    """논문 메타데이터"""
    title: str                      # 원문 제목
    title_ko: str                   # 한글 제목 (번역 후)
    arxiv_id: Optional[str]         # ArXiv ID
    authors: list[str]              # 저자 목록
    domain: str                     # 분야 (NLP, CV, RL, General)
    total_pages: int                # 총 페이지 수
    source_path: str                # 원본 PDF 경로/URL


class TranslationConfig(TypedDict, total=False):
    """번역 설정"""
    domain: str                     # 도메인 필터 (용어 조회용)
    exclude_references: bool        # References 섹션 제외
    extract_tables: bool            # 표 추출 여부
    chunking_strategy: str          # 청킹 전략 (hybrid, section, token)
    max_chunk_tokens: int           # 최대 청크 토큰 수
    overlap_tokens: int             # 오버랩 토큰 수
    pre_process_limit: int          # 용어 프롬프트 제한
    temperature: float              # 번역 temperature
    auto_correct: bool              # 용어 자동 교정
    threshold: float                # 교정 유사도 임계값


class TranslationStats(TypedDict, total=False):
    """번역 통계"""
    total_chunks: int               # 총 청크 수
    completed_chunks: int           # 완료된 청크 수
    failed_chunks: int              # 실패한 청크 수
    total_input_tokens: int         # 입력 토큰 수
    total_output_tokens: int        # 출력 토큰 수
    total_tokens: int               # 총 토큰 수
    total_time_sec: float           # 총 소요 시간 (초)
    estimated_cost_usd: str         # 예상 비용 (USD)
    term_match_rate: str            # 용어 매칭률
    corrections_made: int           # 교정 횟수


class OutputInfo(TypedDict, total=False):
    """출력 파일 정보"""
    file_path: str                  # 저장된 파일 경로
    filename: str                   # 파일명
    md5_hash: str                   # 콘텐츠 MD5 해시
    content_length: int             # 콘텐츠 길이 (bytes)
    rendered_at: str                # 렌더링 시간


class TranslationState(TypedDict, total=False):
    """
    번역 파이프라인 전체 상태

    LangGraph의 각 노드는 이 상태를 읽고 업데이트합니다.
    """
    # === 입력 ===
    source: str                     # PDF URL, ArXiv ID, 또는 로컬 파일 경로
    config: TranslationConfig       # 번역 설정

    # === 중간 상태 ===
    pdf_bytes: bytes                # 다운로드된 PDF 바이너리
    parsed_paper: ParsedPaper       # 파싱된 논문
    chunks: list[Chunk]             # 청크 목록
    processed_chunks: list[ProcessedChunk]      # 전처리된 청크 목록
    translated_chunks: list[TranslatedChunk]    # 번역된 청크 목록
    post_processed_chunks: list[PostProcessedChunk]  # 후처리된 청크 목록

    # === 메타데이터 ===
    metadata: PaperMetadata         # 논문 메타데이터
    stats: TranslationStats         # 번역 통계

    # === 출력 ===
    markdown_content: str           # 렌더링된 마크다운
    output: OutputInfo              # 출력 파일 정보

    # === 제어 ===
    current_node: str               # 현재 실행 중인 노드
    status: Literal["pending", "running", "completed", "failed"]  # 파이프라인 상태
    error: Optional[str]            # 에러 메시지
    started_at: str                 # 시작 시간
    completed_at: Optional[str]     # 완료 시간


def create_initial_state(
    source: str,
    domain: str = "General",
    exclude_references: bool = True,
    extract_tables: bool = True,
    **kwargs
) -> TranslationState:
    """
    초기 상태 생성

    Args:
        source: PDF URL, ArXiv ID, 또는 로컬 파일 경로
        domain: 용어 도메인 (NLP, CV, RL, General)
        exclude_references: References 섹션 제외 여부
        extract_tables: 표 추출 여부
        **kwargs: 추가 설정

    Returns:
        초기화된 TranslationState
    """
    config: TranslationConfig = {
        "domain": domain,
        "exclude_references": exclude_references,
        "extract_tables": extract_tables,
        "chunking_strategy": kwargs.get("chunking_strategy", "hybrid"),
        "max_chunk_tokens": kwargs.get("max_chunk_tokens", 800),
        "overlap_tokens": kwargs.get("overlap_tokens", 100),
        "pre_process_limit": kwargs.get("pre_process_limit", 20),
        "temperature": kwargs.get("temperature", 0.1),
        "auto_correct": kwargs.get("auto_correct", True),
        "threshold": kwargs.get("threshold", 0.8),
    }

    return TranslationState(
        source=source,
        config=config,
        pdf_bytes=b"",
        parsed_paper=None,  # type: ignore
        chunks=[],
        processed_chunks=[],
        translated_chunks=[],
        post_processed_chunks=[],
        metadata={},
        stats={},
        markdown_content="",
        output={},
        current_node="",
        status="pending",
        error=None,
        started_at=datetime.now().isoformat(),
        completed_at=None,
    )


def get_state_summary(state: TranslationState) -> dict:
    """
    상태 요약 정보 반환

    Args:
        state: 현재 상태

    Returns:
        요약 딕셔너리
    """
    return {
        "source": state.get("source", ""),
        "status": state.get("status", "pending"),
        "current_node": state.get("current_node", ""),
        "chunks_count": len(state.get("chunks", [])),
        "processed_count": len(state.get("processed_chunks", [])),
        "translated_count": len(state.get("translated_chunks", [])),
        "post_processed_count": len(state.get("post_processed_chunks", [])),
        "has_output": bool(state.get("output", {}).get("file_path")),
        "error": state.get("error"),
        "started_at": state.get("started_at"),
        "completed_at": state.get("completed_at"),
    }
