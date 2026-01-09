"""
Diff Analyzer 모듈
번역된 마크다운 파일의 변경 사항을 분석하고 용어 변경을 추출
"""
import re
import hashlib
import difflib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from enum import Enum

from src.utils import settings
from src.utils.llm_client import get_llm_client
from src.db.repositories import TranslationRepository, TerminologyRepository, TermChangeRepository


class ChangeType(Enum):
    """변경 유형"""
    ADDED = "added"           # 새로 추가됨
    REMOVED = "removed"       # 삭제됨
    MODIFIED = "modified"     # 수정됨
    TERM_CHANGED = "term_changed"  # 용어 변경


@dataclass
class TextChange:
    """텍스트 변경 사항"""
    change_type: ChangeType
    original_text: str          # 원본 텍스트
    new_text: str               # 변경된 텍스트
    line_number: int            # 라인 번호
    context: str = ""           # 주변 컨텍스트


@dataclass
class TermChange:
    """용어 변경 사항"""
    source_text: str            # 원문 용어 (영어)
    old_target: str             # 기존 번역어
    new_target: str             # 새 번역어
    occurrences: int = 1        # 발생 횟수
    confidence: float = 0.0     # 변경 확신도 (0.0 ~ 1.0)
    context_samples: list[str] = field(default_factory=list)  # 컨텍스트 샘플


@dataclass
class DiffResult:
    """Diff 분석 결과"""
    file_path: str              # 분석 파일 경로
    original_hash: str          # 원본 해시
    current_hash: str           # 현재 해시
    has_changes: bool           # 변경 여부
    text_changes: list[TextChange] = field(default_factory=list)
    term_changes: list[TermChange] = field(default_factory=list)
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DiffAnalyzer:
    """마크다운 파일 변경 분석기"""

    # 용어 변경 분석 프롬프트
    TERM_ANALYSIS_PROMPT = """다음은 번역된 논문 마크다운 파일의 변경 사항입니다.
변경된 부분에서 전문용어 번역이 변경된 경우를 찾아주세요.

변경 사항:
{diff_text}

다음 형식으로 용어 변경만 추출해주세요 (JSON 배열):
[
  {{
    "source_text": "영어 원문 용어",
    "old_target": "기존 한국어 번역",
    "new_target": "새로운 한국어 번역",
    "confidence": 0.9
  }}
]

주의사항:
1. 전문용어(기술 용어, 학술 용어)의 번역 변경만 추출하세요.
2. 일반적인 문장 수정은 제외하세요.
3. 확신도(confidence)는 0.0~1.0 사이 값으로, 명확한 용어 변경일수록 높게 설정하세요.
4. 용어 변경이 없으면 빈 배열 []을 반환하세요.

JSON 배열만 출력하세요:"""

    def __init__(
        self,
        use_llm_analysis: bool = True,
        similarity_threshold: float = 0.6,
        min_confidence: float = 0.7
    ):
        """
        Args:
            use_llm_analysis: LLM을 사용한 용어 변경 분석 여부
            similarity_threshold: 유사도 임계값 (difflib용)
            min_confidence: 최소 확신도 (이 이상만 반영)
        """
        self.use_llm_analysis = use_llm_analysis
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self._llm = None

    @property
    def llm(self):
        """LLM 클라이언트 (lazy loading)"""
        if self._llm is None and self.use_llm_analysis:
            self._llm = get_llm_client()
        return self._llm

    @staticmethod
    def calculate_hash(content: str) -> str:
        """콘텐츠 MD5 해시 계산"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    @staticmethod
    def read_file(file_path: str) -> str:
        """파일 읽기"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def compare_hashes(self, file_path: str, stored_hash: str) -> bool:
        """
        파일 해시 비교

        Args:
            file_path: 파일 경로
            stored_hash: 저장된 해시

        Returns:
            변경 여부 (True면 변경됨)
        """
        content = self.read_file(file_path)
        current_hash = self.calculate_hash(content)
        return current_hash != stored_hash

    def get_stored_hash(self, file_path: str) -> Optional[str]:
        """
        DB에서 저장된 해시 조회

        Args:
            file_path: 파일 경로

        Returns:
            저장된 해시 (없으면 None)
        """
        # 파일명으로 번역 기록 조회
        filename = Path(file_path).stem
        translations = TranslationRepository.get_by_filename(filename)

        if translations and len(translations) > 0:
            return translations[0].get("current_md_hash")
        return None

    def extract_text_changes(
        self,
        original_content: str,
        new_content: str
    ) -> list[TextChange]:
        """
        텍스트 변경 사항 추출 (difflib 활용)

        Args:
            original_content: 원본 콘텐츠
            new_content: 새 콘텐츠

        Returns:
            TextChange 리스트
        """
        changes = []

        original_lines = original_content.splitlines()
        new_lines = new_content.splitlines()

        # unified diff 생성
        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            lineterm="",
            n=2  # 컨텍스트 라인 수
        )

        current_line = 0
        context_buffer = []

        for line in diff:
            if line.startswith("@@"):
                # 라인 번호 추출
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_line = int(match.group(1))
            elif line.startswith("-") and not line.startswith("---"):
                # 삭제된 라인
                changes.append(TextChange(
                    change_type=ChangeType.REMOVED,
                    original_text=line[1:],
                    new_text="",
                    line_number=current_line,
                    context="\n".join(context_buffer[-2:])
                ))
            elif line.startswith("+") and not line.startswith("+++"):
                # 추가된 라인
                changes.append(TextChange(
                    change_type=ChangeType.ADDED,
                    original_text="",
                    new_text=line[1:],
                    line_number=current_line,
                    context="\n".join(context_buffer[-2:])
                ))
                current_line += 1
            elif line.startswith(" "):
                # 컨텍스트 라인
                context_buffer.append(line[1:])
                if len(context_buffer) > 4:
                    context_buffer.pop(0)
                current_line += 1

        # 수정된 라인 병합 (연속된 삭제+추가를 수정으로 변환)
        merged_changes = self._merge_adjacent_changes(changes)

        return merged_changes

    def _merge_adjacent_changes(
        self,
        changes: list[TextChange]
    ) -> list[TextChange]:
        """연속된 삭제+추가를 수정으로 병합"""
        if not changes:
            return []

        merged = []
        i = 0

        while i < len(changes):
            current = changes[i]

            # 삭제 후 추가가 연속되면 수정으로 병합
            if (current.change_type == ChangeType.REMOVED and
                i + 1 < len(changes) and
                changes[i + 1].change_type == ChangeType.ADDED):

                next_change = changes[i + 1]

                # 유사도 체크 (너무 다르면 병합하지 않음)
                similarity = difflib.SequenceMatcher(
                    None,
                    current.original_text,
                    next_change.new_text
                ).ratio()

                if similarity >= self.similarity_threshold:
                    merged.append(TextChange(
                        change_type=ChangeType.MODIFIED,
                        original_text=current.original_text,
                        new_text=next_change.new_text,
                        line_number=current.line_number,
                        context=current.context
                    ))
                    i += 2
                    continue

            merged.append(current)
            i += 1

        return merged

    def analyze_term_changes_with_llm(
        self,
        text_changes: list[TextChange]
    ) -> list[TermChange]:
        """
        LLM을 사용하여 용어 변경 분석

        Args:
            text_changes: 텍스트 변경 목록

        Returns:
            TermChange 리스트
        """
        if not self.llm or not text_changes:
            return []

        # 변경 사항을 텍스트로 변환
        diff_lines = []
        for change in text_changes:
            if change.change_type == ChangeType.MODIFIED:
                diff_lines.append(f"- 기존: {change.original_text}")
                diff_lines.append(f"+ 변경: {change.new_text}")
            elif change.change_type == ChangeType.REMOVED:
                diff_lines.append(f"- 삭제: {change.original_text}")
            elif change.change_type == ChangeType.ADDED:
                diff_lines.append(f"+ 추가: {change.new_text}")

        diff_text = "\n".join(diff_lines)

        if not diff_text.strip():
            return []

        # LLM 호출
        try:
            prompt = self.TERM_ANALYSIS_PROMPT.format(diff_text=diff_text)

            result = self.llm.completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                json_mode=True
            )

            # JSON 파싱
            import json
            response_text = result.get("content", "[]")

            # JSON 배열 추출
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                term_data = json.loads(json_match.group())
            else:
                term_data = []

            # TermChange 객체로 변환
            term_changes = []
            for item in term_data:
                if isinstance(item, dict):
                    confidence = float(item.get("confidence", 0.5))
                    if confidence >= self.min_confidence:
                        term_changes.append(TermChange(
                            source_text=item.get("source_text", ""),
                            old_target=item.get("old_target", ""),
                            new_target=item.get("new_target", ""),
                            confidence=confidence
                        ))

            return term_changes

        except Exception as e:
            import logging
            logging.warning(f"LLM 용어 분석 실패: {e}")
            return []

    def analyze_term_changes_heuristic(
        self,
        text_changes: list[TextChange]
    ) -> list[TermChange]:
        """
        휴리스틱 기반 용어 변경 분석 (LLM 없이)

        DB에 있는 기존 용어를 기반으로 변경 감지

        Args:
            text_changes: 텍스트 변경 목록

        Returns:
            TermChange 리스트
        """
        # 현재 DB의 모든 용어 조회
        all_terms = TerminologyRepository.get_all(limit=1000)
        term_map = {t["target_text"]: t["source_text"] for t in all_terms}

        term_changes = []

        for change in text_changes:
            if change.change_type != ChangeType.MODIFIED:
                continue

            original = change.original_text
            new = change.new_text

            # 기존 용어가 있는지 확인
            for target_text, source_text in term_map.items():
                if target_text in original and target_text not in new:
                    # 기존 용어가 사라짐 - 새로운 번역어 찾기
                    # 비슷한 위치에 새로운 한글 단어가 있는지 확인
                    original_pos = original.find(target_text)

                    # 새 텍스트에서 해당 위치 주변의 한글 단어 추출
                    korean_pattern = r"[가-힣]+"
                    new_words = re.findall(korean_pattern, new)

                    for new_word in new_words:
                        if new_word != target_text and len(new_word) >= 2:
                            # 용어 변경으로 추정
                            term_changes.append(TermChange(
                                source_text=source_text,
                                old_target=target_text,
                                new_target=new_word,
                                confidence=0.6,  # 휴리스틱이므로 낮은 확신도
                                context_samples=[change.context]
                            ))
                            break

        return term_changes

    def analyze_file(
        self,
        file_path: str,
        original_content: Optional[str] = None,
        original_hash: Optional[str] = None
    ) -> DiffResult:
        """
        파일 분석

        Args:
            file_path: 분석할 파일 경로
            original_content: 원본 콘텐츠 (없으면 DB에서 조회 시도)
            original_hash: 원본 해시 (없으면 DB에서 조회)

        Returns:
            DiffResult 객체
        """
        # 현재 파일 읽기
        current_content = self.read_file(file_path)
        current_hash = self.calculate_hash(current_content)

        # 원본 해시 조회
        if original_hash is None:
            original_hash = self.get_stored_hash(file_path) or ""

        # 변경 여부 확인
        has_changes = current_hash != original_hash if original_hash else True

        if not has_changes:
            return DiffResult(
                file_path=file_path,
                original_hash=original_hash,
                current_hash=current_hash,
                has_changes=False
            )

        # 원본 콘텐츠가 없으면 빈 문자열 사용
        if original_content is None:
            original_content = ""

        # 텍스트 변경 추출
        text_changes = self.extract_text_changes(original_content, current_content)

        # 용어 변경 분석
        if self.use_llm_analysis and self.llm:
            term_changes = self.analyze_term_changes_with_llm(text_changes)
        else:
            term_changes = self.analyze_term_changes_heuristic(text_changes)

        return DiffResult(
            file_path=file_path,
            original_hash=original_hash,
            current_hash=current_hash,
            has_changes=True,
            text_changes=text_changes,
            term_changes=term_changes
        )

    def compare_files(
        self,
        original_file: str,
        new_file: str
    ) -> DiffResult:
        """
        두 파일 비교

        Args:
            original_file: 원본 파일 경로
            new_file: 새 파일 경로

        Returns:
            DiffResult 객체
        """
        original_content = self.read_file(original_file)
        original_hash = self.calculate_hash(original_content)

        return self.analyze_file(
            file_path=new_file,
            original_content=original_content,
            original_hash=original_hash
        )

    def compare_with_content(
        self,
        original_content: str,
        new_content: str,
        file_path: str = "memory"
    ) -> DiffResult:
        """
        콘텐츠 직접 비교

        Args:
            original_content: 원본 콘텐츠
            new_content: 새 콘텐츠
            file_path: 파일 경로 (표시용)

        Returns:
            DiffResult 객체
        """
        original_hash = self.calculate_hash(original_content)
        current_hash = self.calculate_hash(new_content)

        has_changes = original_hash != current_hash

        if not has_changes:
            return DiffResult(
                file_path=file_path,
                original_hash=original_hash,
                current_hash=current_hash,
                has_changes=False
            )

        # 텍스트 변경 추출
        text_changes = self.extract_text_changes(original_content, new_content)

        # 용어 변경 분석
        if self.use_llm_analysis and self.llm:
            term_changes = self.analyze_term_changes_with_llm(text_changes)
        else:
            term_changes = self.analyze_term_changes_heuristic(text_changes)

        return DiffResult(
            file_path=file_path,
            original_hash=original_hash,
            current_hash=current_hash,
            has_changes=True,
            text_changes=text_changes,
            term_changes=term_changes
        )

    def get_change_summary(self, result: DiffResult) -> dict:
        """
        변경 사항 요약

        Args:
            result: DiffResult 객체

        Returns:
            요약 딕셔너리
        """
        if not result.has_changes:
            return {
                "has_changes": False,
                "file_path": result.file_path,
                "message": "변경 사항 없음"
            }

        # 변경 유형별 카운트
        type_counts = {}
        for change in result.text_changes:
            change_type = change.change_type.value
            type_counts[change_type] = type_counts.get(change_type, 0) + 1

        return {
            "has_changes": True,
            "file_path": result.file_path,
            "original_hash": result.original_hash,
            "current_hash": result.current_hash,
            "total_text_changes": len(result.text_changes),
            "change_types": type_counts,
            "term_changes_count": len(result.term_changes),
            "term_changes": [
                {
                    "source": tc.source_text,
                    "old": tc.old_target,
                    "new": tc.new_target,
                    "confidence": tc.confidence
                }
                for tc in result.term_changes
            ],
            "analyzed_at": result.analyzed_at
        }

    def format_diff_report(self, result: DiffResult) -> str:
        """
        변경 보고서 생성

        Args:
            result: DiffResult 객체

        Returns:
            포맷된 보고서 문자열
        """
        lines = [
            "=" * 60,
            "변경 분석 보고서",
            "=" * 60,
            f"파일: {result.file_path}",
            f"분석 시간: {result.analyzed_at}",
            f"원본 해시: {result.original_hash[:16]}...",
            f"현재 해시: {result.current_hash[:16]}...",
            "",
        ]

        if not result.has_changes:
            lines.append("변경 사항 없음")
            return "\n".join(lines)

        # 텍스트 변경
        lines.append(f"## 텍스트 변경: {len(result.text_changes)}건")
        lines.append("-" * 40)

        for i, change in enumerate(result.text_changes[:10], 1):
            lines.append(f"{i}. [{change.change_type.value}] 라인 {change.line_number}")
            if change.original_text:
                lines.append(f"   - {change.original_text[:50]}...")
            if change.new_text:
                lines.append(f"   + {change.new_text[:50]}...")

        if len(result.text_changes) > 10:
            lines.append(f"   ... 외 {len(result.text_changes) - 10}건")

        # 용어 변경
        if result.term_changes:
            lines.append("")
            lines.append(f"## 용어 변경: {len(result.term_changes)}건")
            lines.append("-" * 40)

            for tc in result.term_changes:
                lines.append(
                    f"  • {tc.source_text}: "
                    f"'{tc.old_target}' → '{tc.new_target}' "
                    f"(확신도: {tc.confidence:.0%})"
                )

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# 편의 함수
def analyze_file(file_path: str, use_llm: bool = True) -> DiffResult:
    """파일 분석 (단축 함수)"""
    analyzer = DiffAnalyzer(use_llm_analysis=use_llm)
    return analyzer.analyze_file(file_path)


def compare_files(original_file: str, new_file: str, use_llm: bool = True) -> DiffResult:
    """두 파일 비교 (단축 함수)"""
    analyzer = DiffAnalyzer(use_llm_analysis=use_llm)
    return analyzer.compare_files(original_file, new_file)


def compare_content(original: str, new: str, use_llm: bool = False) -> DiffResult:
    """콘텐츠 비교 (단축 함수)"""
    analyzer = DiffAnalyzer(use_llm_analysis=use_llm)
    return analyzer.compare_with_content(original, new)


def get_file_hash(file_path: str) -> str:
    """파일 해시 계산 (단축 함수)"""
    return DiffAnalyzer.calculate_hash(DiffAnalyzer.read_file(file_path))
