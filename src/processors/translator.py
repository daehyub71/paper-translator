"""
Translator 모듈
청크 번역 및 토큰 사용량 추적
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable
from enum import Enum

from openai import (
    APIError,
    APIConnectionError,
    RateLimitError,
    APIStatusError,
)

from src.utils import settings
from src.utils.llm_client import get_llm_client, LLMClient
from .pre_processor import ProcessedChunk


class TranslationStatus(Enum):
    """번역 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class TranslatedChunk:
    """번역된 청크"""
    processed_chunk: ProcessedChunk         # 전처리된 청크
    translated_text: str                    # 번역된 텍스트
    status: TranslationStatus               # 번역 상태
    input_tokens: int = 0                   # 입력 토큰 수
    output_tokens: int = 0                  # 출력 토큰 수
    total_tokens: int = 0                   # 총 토큰 수
    retry_count: int = 0                    # 재시도 횟수
    error_message: str = ""                 # 에러 메시지 (실패 시)
    translation_time: float = 0.0           # 번역 소요 시간 (초)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TranslationStats:
    """번역 통계"""
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    avg_time_per_chunk: float = 0.0
    estimated_cost_usd: float = 0.0


class Translator:
    """청크 번역기"""

    # 번역 프롬프트 템플릿
    SYSTEM_PROMPT_TEMPLATE = """당신은 AI/ML 분야 전문 번역가입니다.
주어진 영어 논문 텍스트를 한국어로 번역하세요.

번역 규칙:
1. 학술적이고 정확한 한국어를 사용하세요.
2. 전문 용어는 아래 용어집을 따르세요. 용어집에 없는 용어는 일반적인 번역을 사용하세요.
3. 수식, 코드, 참조(예: [1], Figure 1)는 원본 그대로 유지하세요.
4. 문장 구조를 자연스러운 한국어로 재구성하세요.
5. 원문의 의미를 정확히 전달하면서도 읽기 쉽게 번역하세요.
6. 표(table)의 내용도 번역하되, 마크다운 형식은 유지하세요.
{terminology_section}
{context_section}"""

    USER_PROMPT_TEMPLATE = """다음 텍스트를 한국어로 번역하세요:

{text}"""

    # GPT-4o-mini 가격 (2024년 기준, USD per 1M tokens)
    PRICE_INPUT_PER_1M = 0.15   # $0.15 per 1M input tokens
    PRICE_OUTPUT_PER_1M = 0.60  # $0.60 per 1M output tokens

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.5,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """
        Args:
            temperature: 번역 temperature (기본: settings에서)
            max_tokens: 최대 출력 토큰 (기본: settings에서)
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간 대기 시간 (초)
            rate_limit_delay: 청크 간 대기 시간 (초, rate limiting용)
            progress_callback: 진행 상황 콜백 (current, total, message)
        """
        self.temperature = temperature or settings.translation_temperature
        self.max_tokens = max_tokens or settings.max_tokens_per_chunk
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.progress_callback = progress_callback

        self._llm: Optional[LLMClient] = None
        self._stats = TranslationStats()

    @property
    def llm(self) -> LLMClient:
        """LLM 클라이언트 (lazy loading)"""
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    def _build_system_prompt(
        self,
        terminology_prompt: str = "",
        context_hint: str = ""
    ) -> str:
        """시스템 프롬프트 구성"""
        terminology_section = ""
        if terminology_prompt:
            terminology_section = f"\n## 용어집\n{terminology_prompt}"

        context_section = ""
        if context_hint:
            context_section = f"\n## 컨텍스트\n{context_hint}"

        return self.SYSTEM_PROMPT_TEMPLATE.format(
            terminology_section=terminology_section,
            context_section=context_section
        )

    def _build_user_prompt(self, text: str) -> str:
        """사용자 프롬프트 구성"""
        return self.USER_PROMPT_TEMPLATE.format(text=text)

    def _call_api_with_retry(
        self,
        messages: list[dict],
        chunk_index: int = 0
    ) -> dict:
        """
        API 호출 (재시도 로직 포함)

        Returns:
            dict: {
                "content": str,
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int,
                "retry_count": int,
                "error": Optional[str]
            }
        """
        last_error = None
        retry_count = 0

        for attempt in range(self.max_retries + 1):
            try:
                result = self.llm.completion(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    json_mode=False
                )

                return {
                    "content": result["content"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"],
                    "retry_count": retry_count,
                    "error": None
                }

            except RateLimitError as e:
                # Rate limit: 더 긴 대기 시간
                last_error = str(e)
                retry_count += 1
                wait_time = self.retry_delay * (2 ** attempt)  # 지수 백오프
                if self.progress_callback:
                    self.progress_callback(
                        chunk_index, 0,
                        f"Rate limit 도달, {wait_time:.1f}초 대기 중..."
                    )
                time.sleep(wait_time)

            except APIConnectionError as e:
                # 연결 오류: 재시도
                last_error = str(e)
                retry_count += 1
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

            except APIStatusError as e:
                # 서버 오류 (5xx): 재시도
                if e.status_code >= 500:
                    last_error = str(e)
                    retry_count += 1
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                else:
                    # 클라이언트 오류 (4xx): 즉시 실패
                    return {
                        "content": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "retry_count": retry_count,
                        "error": str(e)
                    }

            except APIError as e:
                # 일반 API 오류
                last_error = str(e)
                retry_count += 1
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        # 모든 재시도 실패
        return {
            "content": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "retry_count": retry_count,
            "error": f"최대 재시도 횟수 초과: {last_error}"
        }

    def translate_chunk(
        self,
        processed_chunk: ProcessedChunk
    ) -> TranslatedChunk:
        """
        단일 청크 번역

        Args:
            processed_chunk: 전처리된 청크

        Returns:
            TranslatedChunk 객체
        """
        start_time = time.time()

        # 프롬프트 구성
        system_prompt = self._build_system_prompt(
            terminology_prompt=processed_chunk.terminology_prompt,
            context_hint=processed_chunk.context_hint
        )
        user_prompt = self._build_user_prompt(processed_chunk.chunk.content)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # API 호출
        result = self._call_api_with_retry(
            messages=messages,
            chunk_index=processed_chunk.chunk.index
        )

        elapsed_time = time.time() - start_time

        # 결과 생성
        if result["error"]:
            return TranslatedChunk(
                processed_chunk=processed_chunk,
                translated_text="",
                status=TranslationStatus.FAILED,
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                total_tokens=result["total_tokens"],
                retry_count=result["retry_count"],
                error_message=result["error"],
                translation_time=elapsed_time
            )

        return TranslatedChunk(
            processed_chunk=processed_chunk,
            translated_text=result["content"],
            status=TranslationStatus.COMPLETED,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            total_tokens=result["total_tokens"],
            retry_count=result["retry_count"],
            error_message="",
            translation_time=elapsed_time
        )

    def translate_chunks(
        self,
        processed_chunks: list[ProcessedChunk],
        stop_on_error: bool = False
    ) -> list[TranslatedChunk]:
        """
        여러 청크 배치 번역

        Args:
            processed_chunks: 전처리된 청크 목록
            stop_on_error: 에러 시 중단 여부

        Returns:
            TranslatedChunk 목록
        """
        total = len(processed_chunks)
        translated_chunks = []

        # 통계 초기화
        self._stats = TranslationStats(total_chunks=total)
        batch_start_time = time.time()

        for i, processed_chunk in enumerate(processed_chunks):
            # 진행 상황 콜백
            if self.progress_callback:
                self.progress_callback(
                    i + 1, total,
                    f"청크 {i + 1}/{total} 번역 중... "
                    f"(섹션: {processed_chunk.chunk.section_title})"
                )

            # 번역 실행
            translated = self.translate_chunk(processed_chunk)
            translated_chunks.append(translated)

            # 통계 업데이트
            self._update_stats(translated)

            # 에러 처리
            if translated.status == TranslationStatus.FAILED:
                self._stats.failed_chunks += 1
                if stop_on_error:
                    break
            else:
                self._stats.completed_chunks += 1

            # Rate limiting 대기 (마지막 청크 제외)
            if i < total - 1 and self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

        # 최종 통계
        self._stats.total_time = time.time() - batch_start_time
        if self._stats.completed_chunks > 0:
            self._stats.avg_time_per_chunk = (
                self._stats.total_time / self._stats.completed_chunks
            )

        return translated_chunks

    def _update_stats(self, translated: TranslatedChunk) -> None:
        """통계 업데이트"""
        self._stats.total_input_tokens += translated.input_tokens
        self._stats.total_output_tokens += translated.output_tokens
        self._stats.total_tokens += translated.total_tokens

        # 비용 계산 (GPT-4o-mini 기준)
        input_cost = (
            self._stats.total_input_tokens / 1_000_000 * self.PRICE_INPUT_PER_1M
        )
        output_cost = (
            self._stats.total_output_tokens / 1_000_000 * self.PRICE_OUTPUT_PER_1M
        )
        self._stats.estimated_cost_usd = input_cost + output_cost

    def get_stats(self) -> TranslationStats:
        """현재 번역 통계 반환"""
        return self._stats

    def get_stats_summary(self) -> dict:
        """통계 요약 딕셔너리 반환"""
        stats = self._stats
        return {
            "total_chunks": stats.total_chunks,
            "completed": stats.completed_chunks,
            "failed": stats.failed_chunks,
            "success_rate": (
                f"{stats.completed_chunks / stats.total_chunks * 100:.1f}%"
                if stats.total_chunks > 0 else "N/A"
            ),
            "total_tokens": stats.total_tokens,
            "input_tokens": stats.total_input_tokens,
            "output_tokens": stats.total_output_tokens,
            "total_time_sec": round(stats.total_time, 2),
            "avg_time_per_chunk_sec": round(stats.avg_time_per_chunk, 2),
            "estimated_cost_usd": f"${stats.estimated_cost_usd:.4f}",
        }

    def estimate_cost(
        self,
        processed_chunks: list[ProcessedChunk]
    ) -> dict:
        """
        번역 전 예상 비용 계산

        Args:
            processed_chunks: 전처리된 청크 목록

        Returns:
            예상 비용 정보
        """
        total_input_tokens = 0

        for pc in processed_chunks:
            # 시스템 프롬프트 토큰
            system_prompt = self._build_system_prompt(
                terminology_prompt=pc.terminology_prompt,
                context_hint=pc.context_hint
            )
            # 사용자 프롬프트 토큰
            user_prompt = self._build_user_prompt(pc.chunk.content)

            total_input_tokens += self.llm.count_tokens(system_prompt)
            total_input_tokens += self.llm.count_tokens(user_prompt)
            total_input_tokens += 10  # 메시지 오버헤드

        # 출력 토큰 추정 (입력의 약 1.5배로 가정, 한국어는 더 길어짐)
        estimated_output_tokens = int(total_input_tokens * 1.5)

        input_cost = total_input_tokens / 1_000_000 * self.PRICE_INPUT_PER_1M
        output_cost = estimated_output_tokens / 1_000_000 * self.PRICE_OUTPUT_PER_1M
        total_cost = input_cost + output_cost

        return {
            "total_chunks": len(processed_chunks),
            "estimated_input_tokens": total_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": total_input_tokens + estimated_output_tokens,
            "estimated_input_cost_usd": f"${input_cost:.4f}",
            "estimated_output_cost_usd": f"${output_cost:.4f}",
            "estimated_total_cost_usd": f"${total_cost:.4f}",
            "estimated_time_sec": len(processed_chunks) * 2,  # 청크당 약 2초 가정
        }


# 편의 함수
def translate_chunk(
    processed_chunk: ProcessedChunk,
    temperature: float = 0.1
) -> TranslatedChunk:
    """단일 청크 번역 (단축 함수)"""
    translator = Translator(temperature=temperature)
    return translator.translate_chunk(processed_chunk)


def translate_chunks(
    processed_chunks: list[ProcessedChunk],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> list[TranslatedChunk]:
    """여러 청크 번역 (단축 함수)"""
    translator = Translator(progress_callback=progress_callback)
    return translator.translate_chunks(processed_chunks)


def estimate_translation_cost(
    processed_chunks: list[ProcessedChunk]
) -> dict:
    """번역 예상 비용 계산 (단축 함수)"""
    translator = Translator()
    return translator.estimate_cost(processed_chunks)
