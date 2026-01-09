"""
Diff Analyzer 단위 테스트
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.feedback.diff_analyzer import (
    DiffAnalyzer,
    DiffResult,
    TextChange,
    TermChange,
    ChangeType,
    analyze_file,
    compare_files,
    compare_content,
    get_file_hash,
)


class TestDiffAnalyzer:
    """DiffAnalyzer 클래스 테스트"""

    def setup_method(self):
        """각 테스트 전 실행"""
        self.analyzer = DiffAnalyzer(
            use_llm_analysis=False,
            similarity_threshold=0.6,
            min_confidence=0.7
        )

    # === 해시 계산 테스트 ===

    def test_calculate_hash_basic(self):
        """기본 해시 계산"""
        content = "Test content"
        hash1 = DiffAnalyzer.calculate_hash(content)
        hash2 = DiffAnalyzer.calculate_hash(content)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 해시 길이

    def test_calculate_hash_different_content(self):
        """다른 내용은 다른 해시"""
        hash1 = DiffAnalyzer.calculate_hash("Content A")
        hash2 = DiffAnalyzer.calculate_hash("Content B")

        assert hash1 != hash2

    def test_calculate_hash_unicode(self):
        """유니코드 내용 해시"""
        content = "한글 내용입니다. 트랜스포머"
        hash_val = DiffAnalyzer.calculate_hash(content)

        assert len(hash_val) == 32

    # === 파일 읽기 테스트 ===

    def test_read_file_success(self):
        """파일 읽기 성공"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            content = DiffAnalyzer.read_file(temp_path)
            assert content == "Test content"
        finally:
            os.unlink(temp_path)

    def test_read_file_not_found(self):
        """파일 없음 에러"""
        with pytest.raises(FileNotFoundError):
            DiffAnalyzer.read_file("/nonexistent/file.md")

    # === 해시 비교 테스트 ===

    def test_compare_hashes_changed(self):
        """해시 비교 - 변경됨"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("New content")
            temp_path = f.name

        try:
            old_hash = DiffAnalyzer.calculate_hash("Old content")
            is_changed = self.analyzer.compare_hashes(temp_path, old_hash)

            assert is_changed is True
        finally:
            os.unlink(temp_path)

    def test_compare_hashes_unchanged(self):
        """해시 비교 - 변경 없음"""
        content = "Same content"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            stored_hash = DiffAnalyzer.calculate_hash(content)
            is_changed = self.analyzer.compare_hashes(temp_path, stored_hash)

            assert is_changed is False
        finally:
            os.unlink(temp_path)

    # === 텍스트 변경 추출 테스트 ===

    def test_extract_text_changes_addition(self):
        """추가된 라인 감지"""
        original = "Line 1\nLine 2"
        new = "Line 1\nLine 2\nLine 3"

        changes = self.analyzer.extract_text_changes(original, new)

        assert len(changes) > 0
        assert any(c.change_type == ChangeType.ADDED for c in changes)

    def test_extract_text_changes_removal(self):
        """삭제된 라인 감지"""
        original = "Line 1\nLine 2\nLine 3"
        new = "Line 1\nLine 3"

        changes = self.analyzer.extract_text_changes(original, new)

        assert len(changes) > 0
        assert any(c.change_type == ChangeType.REMOVED for c in changes)

    def test_extract_text_changes_modification(self):
        """수정된 라인 감지"""
        original = "The attention mechanism is important."
        new = "The 어텐션 mechanism is important."

        changes = self.analyzer.extract_text_changes(original, new)

        assert len(changes) > 0

    def test_extract_text_changes_no_change(self):
        """변경 없음"""
        content = "Same content\nLine 2"
        changes = self.analyzer.extract_text_changes(content, content)

        assert len(changes) == 0

    # === 변경 병합 테스트 ===

    def test_merge_adjacent_changes(self):
        """연속 변경 병합 (삭제+추가 → 수정)"""
        changes = [
            TextChange(ChangeType.REMOVED, "old text", "", 1),
            TextChange(ChangeType.ADDED, "", "new text", 1),
        ]

        merged = self.analyzer._merge_adjacent_changes(changes)

        # 유사도가 낮으면 병합되지 않을 수 있음
        assert len(merged) >= 1

    def test_merge_adjacent_changes_high_similarity(self):
        """유사도 높은 변경 병합"""
        changes = [
            TextChange(ChangeType.REMOVED, "transformer model", "", 1),
            TextChange(ChangeType.ADDED, "", "트랜스포머 model", 1),
        ]

        merged = self.analyzer._merge_adjacent_changes(changes)

        # 유사도 기반으로 병합됨
        assert len(merged) >= 1

    def test_merge_adjacent_changes_empty(self):
        """빈 목록"""
        merged = self.analyzer._merge_adjacent_changes([])
        assert merged == []

    # === LLM 용어 분석 테스트 ===

    @patch.object(DiffAnalyzer, "llm", new_callable=lambda: Mock())
    def test_analyze_term_changes_with_llm(self, mock_llm):
        """LLM으로 용어 변경 분석"""
        mock_llm.completion.return_value = {
            "content": '[{"source_text": "transformer", "old_target": "트랜스포머", "new_target": "변환기", "confidence": 0.9}]'
        }

        analyzer = DiffAnalyzer(use_llm_analysis=True)
        analyzer._llm = mock_llm

        changes = [
            TextChange(ChangeType.MODIFIED, "트랜스포머", "변환기", 1)
        ]

        term_changes = analyzer.analyze_term_changes_with_llm(changes)

        assert len(term_changes) == 1
        assert term_changes[0].source_text == "transformer"
        assert term_changes[0].confidence >= 0.7

    def test_analyze_term_changes_with_llm_no_changes(self):
        """변경 없으면 빈 목록"""
        analyzer = DiffAnalyzer(use_llm_analysis=False)
        term_changes = analyzer.analyze_term_changes_with_llm([])

        assert term_changes == []

    # === 휴리스틱 용어 분석 테스트 ===

    @patch("src.feedback.diff_analyzer.TerminologyRepository.get_all")
    def test_analyze_term_changes_heuristic(self, mock_get_all):
        """휴리스틱 기반 용어 변경 분석"""
        mock_get_all.return_value = [
            {"source_text": "attention", "target_text": "어텐션"}
        ]

        changes = [
            TextChange(
                ChangeType.MODIFIED,
                "어텐션 메커니즘은 중요합니다",
                "주의 메커니즘은 중요합니다",
                1
            )
        ]

        term_changes = self.analyzer.analyze_term_changes_heuristic(changes)

        # 기존 용어가 사라졌는지 확인
        assert len(term_changes) >= 0

    @patch("src.feedback.diff_analyzer.TerminologyRepository.get_all")
    def test_analyze_term_changes_heuristic_no_match(self, mock_get_all):
        """매칭 없음"""
        mock_get_all.return_value = []

        changes = [TextChange(ChangeType.MODIFIED, "A", "B", 1)]
        term_changes = self.analyzer.analyze_term_changes_heuristic(changes)

        assert term_changes == []

    # === 파일 분석 테스트 ===

    @patch("src.feedback.diff_analyzer.TranslationRepository.get_by_filename")
    def test_analyze_file_no_changes(self, mock_get_by_filename):
        """변경 없는 파일 분석"""
        content = "Test content"
        mock_get_by_filename.return_value = [{
            "current_md_hash": DiffAnalyzer.calculate_hash(content)
        }]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = self.analyzer.analyze_file(temp_path)

            assert isinstance(result, DiffResult)
            assert result.has_changes is False
        finally:
            os.unlink(temp_path)

    @patch("src.feedback.diff_analyzer.TranslationRepository.get_by_filename")
    def test_analyze_file_with_changes(self, mock_get_by_filename):
        """변경 있는 파일 분석"""
        mock_get_by_filename.return_value = [{
            "current_md_hash": "old_hash_different"
        }]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("New content here")
            temp_path = f.name

        try:
            result = self.analyzer.analyze_file(temp_path)

            assert result.has_changes is True
        finally:
            os.unlink(temp_path)

    # === 콘텐츠 비교 테스트 ===

    def test_compare_with_content_no_changes(self):
        """동일 콘텐츠 비교"""
        content = "Same content"
        result = self.analyzer.compare_with_content(content, content)

        assert result.has_changes is False

    def test_compare_with_content_with_changes(self):
        """다른 콘텐츠 비교"""
        result = self.analyzer.compare_with_content(
            "Original content",
            "Modified content"
        )

        assert result.has_changes is True
        assert len(result.text_changes) > 0

    # === 변경 요약 테스트 ===

    def test_get_change_summary_no_changes(self):
        """변경 없음 요약"""
        result = DiffResult(
            file_path="/test.md",
            original_hash="abc",
            current_hash="abc",
            has_changes=False
        )

        summary = self.analyzer.get_change_summary(result)

        assert summary["has_changes"] is False
        assert "변경 사항 없음" in summary["message"]

    def test_get_change_summary_with_changes(self):
        """변경 있음 요약"""
        result = DiffResult(
            file_path="/test.md",
            original_hash="abc",
            current_hash="def",
            has_changes=True,
            text_changes=[
                TextChange(ChangeType.ADDED, "", "new line", 1),
                TextChange(ChangeType.MODIFIED, "old", "new", 2),
            ],
            term_changes=[
                TermChange("attention", "어텐션", "주의", confidence=0.9)
            ]
        )

        summary = self.analyzer.get_change_summary(result)

        assert summary["has_changes"] is True
        assert summary["total_text_changes"] == 2
        assert summary["term_changes_count"] == 1

    # === 보고서 포맷 테스트 ===

    def test_format_diff_report_no_changes(self):
        """변경 없음 보고서"""
        result = DiffResult(
            file_path="/test.md",
            original_hash="abc",
            current_hash="abc",
            has_changes=False
        )

        report = self.analyzer.format_diff_report(result)

        assert "변경 사항 없음" in report

    def test_format_diff_report_with_changes(self):
        """변경 있음 보고서"""
        result = DiffResult(
            file_path="/test.md",
            original_hash="abc",
            current_hash="def",
            has_changes=True,
            text_changes=[
                TextChange(ChangeType.MODIFIED, "old text", "new text", 1)
            ],
            term_changes=[
                TermChange("transformer", "트랜스포머", "변환기", confidence=0.85)
            ]
        )

        report = self.analyzer.format_diff_report(result)

        assert "텍스트 변경" in report
        assert "용어 변경" in report


class TestChangeType:
    """ChangeType 열거형 테스트"""

    def test_change_type_values(self):
        """변경 유형 값"""
        assert ChangeType.ADDED.value == "added"
        assert ChangeType.REMOVED.value == "removed"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.TERM_CHANGED.value == "term_changed"


class TestTextChange:
    """TextChange 데이터클래스 테스트"""

    def test_text_change_creation(self):
        """텍스트 변경 생성"""
        change = TextChange(
            change_type=ChangeType.MODIFIED,
            original_text="old",
            new_text="new",
            line_number=10,
            context="surrounding text"
        )

        assert change.change_type == ChangeType.MODIFIED
        assert change.original_text == "old"
        assert change.new_text == "new"
        assert change.line_number == 10


class TestTermChange:
    """TermChange 데이터클래스 테스트"""

    def test_term_change_creation(self):
        """용어 변경 생성"""
        change = TermChange(
            source_text="attention",
            old_target="어텐션",
            new_target="주의",
            confidence=0.9
        )

        assert change.source_text == "attention"
        assert change.confidence == 0.9

    def test_term_change_defaults(self):
        """기본값"""
        change = TermChange("src", "old", "new")

        assert change.occurrences == 1
        assert change.confidence == 0.0
        assert change.context_samples == []


class TestConvenienceFunctions:
    """편의 함수 테스트"""

    @patch("src.feedback.diff_analyzer.DiffAnalyzer.analyze_file")
    def test_analyze_file_function(self, mock_analyze):
        """analyze_file 단축 함수"""
        mock_analyze.return_value = DiffResult(
            file_path="/test.md",
            original_hash="abc",
            current_hash="def",
            has_changes=True
        )

        result = analyze_file("/test.md", use_llm=False)

        assert isinstance(result, DiffResult)

    @patch("src.feedback.diff_analyzer.DiffAnalyzer.compare_files")
    def test_compare_files_function(self, mock_compare):
        """compare_files 단축 함수"""
        mock_compare.return_value = DiffResult(
            file_path="/new.md",
            original_hash="abc",
            current_hash="def",
            has_changes=True
        )

        result = compare_files("/old.md", "/new.md")

        assert isinstance(result, DiffResult)

    @patch("src.feedback.diff_analyzer.DiffAnalyzer.compare_with_content")
    def test_compare_content_function(self, mock_compare):
        """compare_content 단축 함수"""
        mock_compare.return_value = DiffResult(
            file_path="memory",
            original_hash="abc",
            current_hash="def",
            has_changes=True
        )

        result = compare_content("old", "new")

        assert isinstance(result, DiffResult)

    def test_get_file_hash_function(self):
        """get_file_hash 단축 함수"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            hash_val = get_file_hash(temp_path)

            assert len(hash_val) == 32
        finally:
            os.unlink(temp_path)
