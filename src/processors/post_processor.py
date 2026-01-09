"""
Post-processor 모듈
번역 결과의 용어 매칭 검증 및 교정
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from difflib import SequenceMatcher

from src.utils import settings
from .translator import TranslatedChunk, TranslationStatus


# 로거 설정
logger = logging.getLogger(__name__)


class TermMatchStatus(Enum):
    """용어 매칭 상태"""
    MATCHED = "matched"           # 정확히 매칭됨
    SIMILAR = "similar"           # 유사하게 매칭됨 (threshold 이상)
    MISSING = "missing"           # 번역문에 없음
    MISMATCHED = "mismatched"     # 다르게 번역됨


@dataclass
class TermValidation:
    """용어 검증 결과"""
    source_text: str              # 원문 용어
    expected_target: str          # 예상 번역
    actual_target: Optional[str]  # 실제 번역 (없으면 None)
    status: TermMatchStatus       # 매칭 상태
    similarity: float = 0.0       # 유사도 (0-1)
    corrected: bool = False       # 교정 여부
    position: int = -1            # 원문에서의 위치


@dataclass
class PostProcessedChunk:
    """후처리된 청크"""
    translated_chunk: TranslatedChunk   # 원본 번역 청크
    corrected_text: str                 # 교정된 번역문
    validations: list[TermValidation] = field(default_factory=list)  # 용어 검증 결과
    corrections_made: int = 0           # 교정 횟수
    missing_terms: int = 0              # 누락 용어 수
    mismatched_terms: int = 0           # 불일치 용어 수


@dataclass
class PostProcessStats:
    """후처리 통계"""
    total_chunks: int = 0
    total_terms_checked: int = 0
    matched_terms: int = 0
    similar_terms: int = 0
    missing_terms: int = 0
    mismatched_terms: int = 0
    corrections_made: int = 0
    correction_rate: float = 0.0


class PostProcessor:
    """번역 후처리기"""

    def __init__(
        self,
        threshold: Optional[float] = None,
        auto_correct: bool = True,
        log_corrections: bool = True
    ):
        """
        Args:
            threshold: 유사도 임계값 (기본: settings에서)
            auto_correct: 자동 교정 여부
            log_corrections: 교정 내용 로깅 여부
        """
        self.threshold = threshold or settings.post_process_threshold
        self.auto_correct = auto_correct
        self.log_corrections = log_corrections

        self._stats = PostProcessStats()

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        두 문자열의 유사도 계산 (0-1)

        Args:
            str1: 첫 번째 문자열
            str2: 두 번째 문자열

        Returns:
            유사도 (0-1)
        """
        if not str1 or not str2:
            return 0.0

        # 정규화 (소문자, 공백 제거)
        s1 = str1.lower().strip()
        s2 = str2.lower().strip()

        if s1 == s2:
            return 1.0

        return SequenceMatcher(None, s1, s2).ratio()

    def find_term_in_text(
        self,
        text: str,
        term: str,
        case_sensitive: bool = False
    ) -> list[tuple[int, int]]:
        """
        텍스트에서 용어 위치 찾기

        Args:
            text: 검색 대상 텍스트
            term: 찾을 용어
            case_sensitive: 대소문자 구분 여부

        Returns:
            (시작위치, 끝위치) 튜플 리스트
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.escape(term)
        matches = []

        for match in re.finditer(pattern, text, flags):
            matches.append((match.start(), match.end()))

        return matches

    def find_similar_term_in_text(
        self,
        text: str,
        term: str,
        min_similarity: float = 0.7
    ) -> Optional[tuple[str, float, int]]:
        """
        텍스트에서 유사한 용어 찾기

        Args:
            text: 검색 대상 텍스트
            term: 찾을 용어
            min_similarity: 최소 유사도

        Returns:
            (찾은 용어, 유사도, 위치) 또는 None
        """
        # 한국어 단어 추출 패턴
        # 한글, 영문, 숫자를 포함한 단어 단위 추출
        word_pattern = r"[\w가-힣]+"
        words = re.findall(word_pattern, text)

        best_match = None
        best_similarity = 0.0
        best_position = -1

        for word in words:
            similarity = self.calculate_similarity(term, word)
            if similarity >= min_similarity and similarity > best_similarity:
                best_similarity = similarity
                best_match = word
                # 위치 찾기
                match = re.search(re.escape(word), text)
                if match:
                    best_position = match.start()

        if best_match:
            return (best_match, best_similarity, best_position)
        return None

    def validate_term(
        self,
        source_text: str,
        expected_target: str,
        original_text: str,
        translated_text: str
    ) -> TermValidation:
        """
        단일 용어 검증

        Args:
            source_text: 원문 용어 (영어)
            expected_target: 예상 번역 (한국어)
            original_text: 원본 청크 텍스트
            translated_text: 번역된 텍스트

        Returns:
            TermValidation 객체
        """
        # 원문에 용어가 있는지 확인
        source_positions = self.find_term_in_text(original_text, source_text)
        if not source_positions:
            # 원문에 용어가 없으면 검증 대상 아님
            return TermValidation(
                source_text=source_text,
                expected_target=expected_target,
                actual_target=None,
                status=TermMatchStatus.MATCHED,  # 해당 없음으로 처리
                similarity=1.0,
                position=-1
            )

        position = source_positions[0][0]

        # 번역문에서 예상 용어 찾기
        target_positions = self.find_term_in_text(translated_text, expected_target)

        if target_positions:
            # 정확히 매칭됨
            return TermValidation(
                source_text=source_text,
                expected_target=expected_target,
                actual_target=expected_target,
                status=TermMatchStatus.MATCHED,
                similarity=1.0,
                position=position
            )

        # 유사한 용어 찾기
        similar = self.find_similar_term_in_text(
            translated_text,
            expected_target,
            min_similarity=self.threshold
        )

        if similar:
            actual, similarity, _ = similar
            if similarity >= self.threshold:
                return TermValidation(
                    source_text=source_text,
                    expected_target=expected_target,
                    actual_target=actual,
                    status=TermMatchStatus.SIMILAR,
                    similarity=similarity,
                    position=position
                )
            else:
                return TermValidation(
                    source_text=source_text,
                    expected_target=expected_target,
                    actual_target=actual,
                    status=TermMatchStatus.MISMATCHED,
                    similarity=similarity,
                    position=position
                )

        # 용어가 번역문에 없음
        return TermValidation(
            source_text=source_text,
            expected_target=expected_target,
            actual_target=None,
            status=TermMatchStatus.MISSING,
            similarity=0.0,
            position=position
        )

    def validate_chunk_terms(
        self,
        translated_chunk: TranslatedChunk
    ) -> list[TermValidation]:
        """
        청크의 모든 용어 검증

        Args:
            translated_chunk: 번역된 청크

        Returns:
            TermValidation 리스트
        """
        validations = []
        processed_chunk = translated_chunk.processed_chunk
        matched_terms = processed_chunk.matched_terms

        if not matched_terms:
            return validations

        original_text = processed_chunk.chunk.content
        translated_text = translated_chunk.translated_text

        for term in matched_terms:
            source = term.get("source_text", "")
            target = term.get("target_text", "")

            if source and target:
                validation = self.validate_term(
                    source_text=source,
                    expected_target=target,
                    original_text=original_text,
                    translated_text=translated_text
                )
                validations.append(validation)

        return validations

    def detect_missing_terms(
        self,
        validations: list[TermValidation]
    ) -> list[TermValidation]:
        """
        누락된 용어 검출

        Args:
            validations: 용어 검증 결과 리스트

        Returns:
            누락된 용어 검증 리스트
        """
        return [v for v in validations if v.status == TermMatchStatus.MISSING]

    def detect_mismatched_terms(
        self,
        validations: list[TermValidation]
    ) -> list[TermValidation]:
        """
        불일치 용어 검출

        Args:
            validations: 용어 검증 결과 리스트

        Returns:
            불일치 용어 검증 리스트
        """
        return [v for v in validations if v.status == TermMatchStatus.MISMATCHED]

    def correct_term(
        self,
        text: str,
        actual: str,
        expected: str
    ) -> str:
        """
        텍스트에서 용어 교정

        Args:
            text: 원본 텍스트
            actual: 실제 사용된 용어
            expected: 올바른 용어

        Returns:
            교정된 텍스트
        """
        if not actual or not expected:
            return text

        # 정확한 매칭으로 교체
        pattern = re.compile(re.escape(actual))
        return pattern.sub(expected, text, count=1)

    def correct_terms(
        self,
        translated_text: str,
        validations: list[TermValidation]
    ) -> tuple[str, int]:
        """
        불일치 용어들 교정

        Args:
            translated_text: 번역된 텍스트
            validations: 용어 검증 결과 리스트

        Returns:
            (교정된 텍스트, 교정 횟수)
        """
        corrected_text = translated_text
        correction_count = 0

        for validation in validations:
            if validation.status == TermMatchStatus.MISMATCHED:
                if validation.actual_target and validation.expected_target:
                    new_text = self.correct_term(
                        corrected_text,
                        validation.actual_target,
                        validation.expected_target
                    )
                    if new_text != corrected_text:
                        corrected_text = new_text
                        validation.corrected = True
                        correction_count += 1

                        if self.log_corrections:
                            logger.info(
                                f"용어 교정: '{validation.actual_target}' → "
                                f"'{validation.expected_target}' "
                                f"(원문: {validation.source_text})"
                            )

        return corrected_text, correction_count

    def process_chunk(
        self,
        translated_chunk: TranslatedChunk
    ) -> PostProcessedChunk:
        """
        단일 청크 후처리

        Args:
            translated_chunk: 번역된 청크

        Returns:
            PostProcessedChunk 객체
        """
        # 번역 실패한 경우 건너뛰기
        if translated_chunk.status != TranslationStatus.COMPLETED:
            return PostProcessedChunk(
                translated_chunk=translated_chunk,
                corrected_text=translated_chunk.translated_text,
                validations=[],
                corrections_made=0,
                missing_terms=0,
                mismatched_terms=0
            )

        # 용어 검증
        validations = self.validate_chunk_terms(translated_chunk)

        # 누락/불일치 용어 감지
        missing = self.detect_missing_terms(validations)
        mismatched = self.detect_mismatched_terms(validations)

        # 교정
        corrected_text = translated_chunk.translated_text
        corrections_made = 0

        if self.auto_correct and mismatched:
            corrected_text, corrections_made = self.correct_terms(
                translated_chunk.translated_text,
                validations
            )

        # 로깅
        if self.log_corrections:
            if missing:
                logger.warning(
                    f"청크 {translated_chunk.processed_chunk.chunk.index}: "
                    f"{len(missing)}개 용어 누락"
                )
                for v in missing:
                    logger.warning(f"  - '{v.source_text}' → '{v.expected_target}'")

        return PostProcessedChunk(
            translated_chunk=translated_chunk,
            corrected_text=corrected_text,
            validations=validations,
            corrections_made=corrections_made,
            missing_terms=len(missing),
            mismatched_terms=len(mismatched)
        )

    def process_chunks(
        self,
        translated_chunks: list[TranslatedChunk]
    ) -> list[PostProcessedChunk]:
        """
        여러 청크 후처리

        Args:
            translated_chunks: 번역된 청크 리스트

        Returns:
            PostProcessedChunk 리스트
        """
        processed = []

        # 통계 초기화
        self._stats = PostProcessStats(total_chunks=len(translated_chunks))

        for chunk in translated_chunks:
            post_processed = self.process_chunk(chunk)
            processed.append(post_processed)

            # 통계 업데이트
            self._update_stats(post_processed)

        # 교정률 계산
        if self._stats.total_terms_checked > 0:
            self._stats.correction_rate = (
                self._stats.corrections_made / self._stats.total_terms_checked
            )

        return processed

    def _update_stats(self, post_processed: PostProcessedChunk) -> None:
        """통계 업데이트"""
        for v in post_processed.validations:
            self._stats.total_terms_checked += 1

            if v.status == TermMatchStatus.MATCHED:
                self._stats.matched_terms += 1
            elif v.status == TermMatchStatus.SIMILAR:
                self._stats.similar_terms += 1
            elif v.status == TermMatchStatus.MISSING:
                self._stats.missing_terms += 1
            elif v.status == TermMatchStatus.MISMATCHED:
                self._stats.mismatched_terms += 1

            if v.corrected:
                self._stats.corrections_made += 1

    def get_stats(self) -> PostProcessStats:
        """통계 반환"""
        return self._stats

    def get_stats_summary(self) -> dict:
        """통계 요약 딕셔너리"""
        stats = self._stats
        return {
            "total_chunks": stats.total_chunks,
            "total_terms_checked": stats.total_terms_checked,
            "matched_terms": stats.matched_terms,
            "similar_terms": stats.similar_terms,
            "missing_terms": stats.missing_terms,
            "mismatched_terms": stats.mismatched_terms,
            "corrections_made": stats.corrections_made,
            "match_rate": (
                f"{(stats.matched_terms + stats.similar_terms) / stats.total_terms_checked * 100:.1f}%"
                if stats.total_terms_checked > 0 else "N/A"
            ),
            "correction_rate": (
                f"{stats.correction_rate * 100:.1f}%"
                if stats.total_terms_checked > 0 else "N/A"
            ),
        }

    def generate_correction_report(
        self,
        processed_chunks: list[PostProcessedChunk]
    ) -> str:
        """
        교정 보고서 생성

        Args:
            processed_chunks: 후처리된 청크 리스트

        Returns:
            마크다운 형식 보고서
        """
        report_lines = ["# 용어 교정 보고서\n"]

        # 요약
        stats = self.get_stats_summary()
        report_lines.append("## 요약\n")
        report_lines.append(f"- 총 청크: {stats['total_chunks']}")
        report_lines.append(f"- 검사 용어: {stats['total_terms_checked']}")
        report_lines.append(f"- 매칭 성공: {stats['matched_terms']}")
        report_lines.append(f"- 유사 매칭: {stats['similar_terms']}")
        report_lines.append(f"- 누락 용어: {stats['missing_terms']}")
        report_lines.append(f"- 불일치 용어: {stats['mismatched_terms']}")
        report_lines.append(f"- 교정 횟수: {stats['corrections_made']}")
        report_lines.append(f"- 매칭률: {stats['match_rate']}")
        report_lines.append("")

        # 상세 내역
        report_lines.append("## 상세 내역\n")

        for pc in processed_chunks:
            chunk = pc.translated_chunk.processed_chunk.chunk
            if pc.missing_terms > 0 or pc.mismatched_terms > 0 or pc.corrections_made > 0:
                report_lines.append(f"### 청크 {chunk.index} ({chunk.section_title})\n")

                for v in pc.validations:
                    if v.status in [TermMatchStatus.MISSING, TermMatchStatus.MISMATCHED]:
                        status_icon = "❌" if v.status == TermMatchStatus.MISSING else "⚠️"
                        corrected_icon = " ✅ 교정됨" if v.corrected else ""
                        report_lines.append(
                            f"- {status_icon} **{v.source_text}**: "
                            f"예상 '{v.expected_target}', "
                            f"실제 '{v.actual_target or '(없음)'}'"
                            f"{corrected_icon}"
                        )

                report_lines.append("")

        return "\n".join(report_lines)


# 편의 함수
def postprocess_chunk(
    translated_chunk: TranslatedChunk,
    auto_correct: bool = True
) -> PostProcessedChunk:
    """단일 청크 후처리 (단축 함수)"""
    processor = PostProcessor(auto_correct=auto_correct)
    return processor.process_chunk(translated_chunk)


def postprocess_chunks(
    translated_chunks: list[TranslatedChunk],
    auto_correct: bool = True
) -> list[PostProcessedChunk]:
    """여러 청크 후처리 (단축 함수)"""
    processor = PostProcessor(auto_correct=auto_correct)
    return processor.process_chunks(translated_chunks)


def validate_terminology(
    translated_chunk: TranslatedChunk
) -> list[TermValidation]:
    """용어 검증 (단축 함수)"""
    processor = PostProcessor(auto_correct=False)
    return processor.validate_chunk_terms(translated_chunk)
