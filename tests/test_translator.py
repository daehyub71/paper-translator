"""
Translator 단위 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field

from openai import RateLimitError, APIConnectionError, APIStatusError, APIError

from src.processors.translator import (
    Translator,
    TranslatedChunk,
    TranslationStats,
    TranslationStatus,
    translate_chunk,
    translate_chunks,
    estimate_translation_cost,
)
from src.processors.chunker import Chunk
from src.processors.pre_processor import ProcessedChunk


def create_mock_chunk(index=0, content="Test content"):
    """테스트용 청크 생성"""
    return Chunk(
        index=index,
        content=content,
        section_title="Test Section",
        section_index=0,
        token_count=10,
        start_char=0,
        end_char=len(content),
        has_overlap=False,
        tables=[]
    )


def create_mock_processed_chunk(index=0, content="Test content"):
    """테스트용 전처리된 청크 생성"""
    return ProcessedChunk(
        chunk=create_mock_chunk(index, content),
        matched_terms=[],
        terminology_prompt="",
        context_hint=""
    )


class TestTranslator:
    """Translator 클래스 테스트"""

    def setup_method(self):
        """각 테스트 전 실행"""
        self.translator = Translator(
            temperature=0.1,
            max_tokens=2000,
            max_retries=3,
            retry_delay=0.1,
            rate_limit_delay=0.0
        )

    # === 프롬프트 구성 테스트 ===

    def test_build_system_prompt_basic(self):
        """기본 시스템 프롬프트"""
        prompt = self.translator._build_system_prompt()

        assert "AI/ML 분야 전문 번역가" in prompt
        assert "번역 규칙" in prompt

    def test_build_system_prompt_with_terminology(self):
        """용어집 포함 프롬프트"""
        terminology = "transformer: 트랜스포머\nattention: 어텐션"
        prompt = self.translator._build_system_prompt(terminology_prompt=terminology)

        assert "용어집" in prompt
        assert "transformer" in prompt

    def test_build_system_prompt_with_context(self):
        """컨텍스트 포함 프롬프트"""
        context = "이 섹션은 모델 아키텍처를 설명합니다."
        prompt = self.translator._build_system_prompt(context_hint=context)

        assert "컨텍스트" in prompt
        assert "모델 아키텍처" in prompt

    def test_build_user_prompt(self):
        """사용자 프롬프트"""
        text = "The transformer model uses self-attention."
        prompt = self.translator._build_user_prompt(text)

        assert "한국어로 번역" in prompt
        assert text in prompt

    # === API 호출 테스트 ===

    @patch.object(Translator, "llm", new_callable=lambda: Mock())
    def test_call_api_success(self, mock_llm):
        """API 호출 성공"""
        mock_llm.completion.return_value = {
            "content": "번역된 텍스트",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        }
        self.translator._llm = mock_llm

        messages = [{"role": "user", "content": "Test"}]
        result = self.translator._call_api_with_retry(messages)

        assert result["content"] == "번역된 텍스트"
        assert result["error"] is None
        assert result["retry_count"] == 0

    @patch.object(Translator, "llm", new_callable=lambda: Mock())
    def test_call_api_retry_on_rate_limit(self, mock_llm):
        """Rate limit 시 재시도"""
        mock_llm.completion.side_effect = [
            RateLimitError("Rate limit", response=Mock(status_code=429), body={}),
            {
                "content": "번역됨",
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150
            }
        ]
        self.translator._llm = mock_llm

        messages = [{"role": "user", "content": "Test"}]
        result = self.translator._call_api_with_retry(messages)

        assert result["content"] == "번역됨"
        assert result["retry_count"] == 1

    @patch.object(Translator, "llm", new_callable=lambda: Mock())
    def test_call_api_max_retries_exceeded(self, mock_llm):
        """최대 재시도 초과"""
        mock_llm.completion.side_effect = APIConnectionError(
            message="Connection failed",
            request=Mock()
        )
        self.translator._llm = mock_llm
        self.translator.max_retries = 2

        messages = [{"role": "user", "content": "Test"}]
        result = self.translator._call_api_with_retry(messages)

        assert result["content"] == ""
        assert result["error"] is not None
        assert "최대 재시도" in result["error"]

    # === 단일 청크 번역 테스트 ===

    @patch.object(Translator, "llm", new_callable=lambda: Mock())
    def test_translate_chunk_success(self, mock_llm):
        """청크 번역 성공"""
        mock_llm.completion.return_value = {
            "content": "번역된 내용입니다.",
            "input_tokens": 100,
            "output_tokens": 80,
            "total_tokens": 180
        }
        self.translator._llm = mock_llm

        processed = create_mock_processed_chunk(content="Original text")
        result = self.translator.translate_chunk(processed)

        assert isinstance(result, TranslatedChunk)
        assert result.status == TranslationStatus.COMPLETED
        assert result.translated_text == "번역된 내용입니다."
        assert result.input_tokens == 100
        assert result.output_tokens == 80

    @patch.object(Translator, "llm", new_callable=lambda: Mock())
    def test_translate_chunk_failure(self, mock_llm):
        """청크 번역 실패"""
        mock_llm.completion.side_effect = APIError(
            message="API Error",
            request=Mock(),
            body={}
        )
        self.translator._llm = mock_llm
        self.translator.max_retries = 0

        processed = create_mock_processed_chunk(content="Original text")
        result = self.translator.translate_chunk(processed)

        assert result.status == TranslationStatus.FAILED
        assert result.translated_text == ""
        assert result.error_message != ""

    # === 배치 번역 테스트 ===

    @patch.object(Translator, "llm", new_callable=lambda: Mock())
    def test_translate_chunks_batch(self, mock_llm):
        """여러 청크 배치 번역"""
        mock_llm.completion.return_value = {
            "content": "번역됨",
            "input_tokens": 50,
            "output_tokens": 40,
            "total_tokens": 90
        }
        self.translator._llm = mock_llm

        chunks = [
            create_mock_processed_chunk(i, f"Content {i}")
            for i in range(3)
        ]

        results = self.translator.translate_chunks(chunks)

        assert len(results) == 3
        assert all(r.status == TranslationStatus.COMPLETED for r in results)

    @patch.object(Translator, "llm", new_callable=lambda: Mock())
    def test_translate_chunks_with_callback(self, mock_llm):
        """진행 콜백 호출 확인"""
        mock_llm.completion.return_value = {
            "content": "번역됨",
            "input_tokens": 50,
            "output_tokens": 40,
            "total_tokens": 90
        }
        self.translator._llm = mock_llm

        callback_calls = []

        def callback(current, total, message):
            callback_calls.append((current, total, message))

        self.translator.progress_callback = callback

        chunks = [create_mock_processed_chunk(i) for i in range(2)]
        self.translator.translate_chunks(chunks)

        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 1
        assert callback_calls[1][0] == 2

    @patch.object(Translator, "llm", new_callable=lambda: Mock())
    def test_translate_chunks_stop_on_error(self, mock_llm):
        """에러 시 중단 옵션"""
        mock_llm.completion.side_effect = [
            {"content": "OK", "input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            APIError(message="Error", request=Mock(), body={}),
        ]
        self.translator._llm = mock_llm
        self.translator.max_retries = 0

        chunks = [create_mock_processed_chunk(i) for i in range(3)]
        results = self.translator.translate_chunks(chunks, stop_on_error=True)

        # 첫 번째 성공, 두 번째 실패, 세 번째는 실행 안 됨
        assert len(results) == 2
        assert results[0].status == TranslationStatus.COMPLETED
        assert results[1].status == TranslationStatus.FAILED

    # === 통계 테스트 ===

    @patch.object(Translator, "llm", new_callable=lambda: Mock())
    def test_translation_stats(self, mock_llm):
        """번역 통계 수집"""
        mock_llm.completion.return_value = {
            "content": "번역됨",
            "input_tokens": 100,
            "output_tokens": 80,
            "total_tokens": 180
        }
        self.translator._llm = mock_llm

        chunks = [create_mock_processed_chunk(i) for i in range(2)]
        self.translator.translate_chunks(chunks)

        stats = self.translator.get_stats()

        assert stats.total_chunks == 2
        assert stats.completed_chunks == 2
        assert stats.failed_chunks == 0
        assert stats.total_input_tokens == 200
        assert stats.total_output_tokens == 160

    def test_get_stats_summary(self):
        """통계 요약"""
        self.translator._stats = TranslationStats(
            total_chunks=10,
            completed_chunks=9,
            failed_chunks=1,
            total_input_tokens=1000,
            total_output_tokens=800,
            total_tokens=1800,
            total_time=30.0,
            avg_time_per_chunk=3.0,
            estimated_cost_usd=0.01
        )

        summary = self.translator.get_stats_summary()

        assert summary["total_chunks"] == 10
        assert summary["completed"] == 9
        assert summary["failed"] == 1
        assert "90.0%" in summary["success_rate"]

    # === 비용 추정 테스트 ===

    @patch.object(Translator, "llm", new_callable=lambda: Mock())
    def test_estimate_cost(self, mock_llm):
        """번역 비용 추정"""
        mock_llm.count_tokens.return_value = 100
        self.translator._llm = mock_llm

        chunks = [create_mock_processed_chunk(i) for i in range(5)]
        estimate = self.translator.estimate_cost(chunks)

        assert estimate["total_chunks"] == 5
        assert estimate["estimated_input_tokens"] > 0
        assert estimate["estimated_output_tokens"] > 0
        assert "$" in estimate["estimated_total_cost_usd"]


class TestTranslationStatus:
    """TranslationStatus 열거형 테스트"""

    def test_status_values(self):
        """상태 값 확인"""
        assert TranslationStatus.PENDING.value == "pending"
        assert TranslationStatus.IN_PROGRESS.value == "in_progress"
        assert TranslationStatus.COMPLETED.value == "completed"
        assert TranslationStatus.FAILED.value == "failed"


class TestTranslatedChunk:
    """TranslatedChunk 데이터클래스 테스트"""

    def test_translated_chunk_creation(self):
        """번역된 청크 생성"""
        processed = create_mock_processed_chunk()
        chunk = TranslatedChunk(
            processed_chunk=processed,
            translated_text="번역된 텍스트",
            status=TranslationStatus.COMPLETED,
            input_tokens=100,
            output_tokens=80,
            total_tokens=180
        )

        assert chunk.translated_text == "번역된 텍스트"
        assert chunk.status == TranslationStatus.COMPLETED


class TestConvenienceFunctions:
    """편의 함수 테스트"""

    @patch("src.processors.translator.Translator.translate_chunk")
    def test_translate_chunk_function(self, mock_translate):
        """translate_chunk 단축 함수"""
        mock_translate.return_value = TranslatedChunk(
            processed_chunk=create_mock_processed_chunk(),
            translated_text="번역됨",
            status=TranslationStatus.COMPLETED
        )

        processed = create_mock_processed_chunk()
        result = translate_chunk(processed)

        assert result.translated_text == "번역됨"

    @patch("src.processors.translator.Translator.translate_chunks")
    def test_translate_chunks_function(self, mock_translate):
        """translate_chunks 단축 함수"""
        mock_translate.return_value = [
            TranslatedChunk(
                processed_chunk=create_mock_processed_chunk(),
                translated_text="번역됨",
                status=TranslationStatus.COMPLETED
            )
        ]

        chunks = [create_mock_processed_chunk()]
        results = translate_chunks(chunks)

        assert len(results) == 1

    @patch("src.processors.translator.Translator.estimate_cost")
    def test_estimate_translation_cost_function(self, mock_estimate):
        """estimate_translation_cost 단축 함수"""
        mock_estimate.return_value = {
            "total_chunks": 5,
            "estimated_total_cost_usd": "$0.01"
        }

        chunks = [create_mock_processed_chunk()]
        estimate = estimate_translation_cost(chunks)

        assert "estimated_total_cost_usd" in estimate
