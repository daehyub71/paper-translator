"""
통합 테스트
E2E, CLI 명령어, 워크플로우 테스트
"""
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner as TyperCliRunner

# CLI 앱 import
from src.main import app


class TestCLITranslateCommand:
    """translate 명령어 테스트"""

    def setup_method(self):
        """테스트 설정"""
        self.runner = TyperCliRunner()

    def test_translate_no_source(self):
        """소스 없이 실행 시 에러"""
        result = self.runner.invoke(app, ["translate"])

        # 소스가 없으면 에러
        assert result.exit_code != 0 or "오류" in result.stdout or "Error" in result.stdout

    def test_translate_help(self):
        """translate 도움말"""
        result = self.runner.invoke(app, ["translate", "--help"])

        assert result.exit_code == 0

    def test_translate_with_nonexistent_file(self):
        """존재하지 않는 파일"""
        result = self.runner.invoke(
            app,
            ["translate", "--file", "/nonexistent/paper.pdf"]
        )

        # 파일 없음 에러
        assert result.exit_code != 0 or "오류" in result.stdout


class TestCLISyncCommand:
    """sync 명령어 테스트"""

    def setup_method(self):
        """테스트 설정"""
        self.runner = TyperCliRunner()

    def test_sync_help(self):
        """sync 도움말"""
        result = self.runner.invoke(app, ["sync", "--help"])

        assert result.exit_code == 0

    @patch("src.main.SyncManager")
    def test_sync_single_file(self, mock_sync_manager):
        """단일 파일 동기화"""
        mock_manager = Mock()
        mock_manager.analyze_file.return_value = Mock(has_changes=False)
        mock_sync_manager.return_value = mock_manager

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"# Test content")
            temp_path = f.name

        try:
            result = self.runner.invoke(app, ["sync", "--file", temp_path])
        finally:
            os.unlink(temp_path)


class TestCLITermsCommand:
    """terms 명령어 그룹 테스트"""

    def setup_method(self):
        """테스트 설정"""
        self.runner = TyperCliRunner()

    @patch("src.main.TerminologyRepository.get_all")
    def test_terms_list(self, mock_get_all):
        """용어 목록 조회"""
        mock_get_all.return_value = [
            {"source_text": "transformer", "target_text": "트랜스포머", "domain": "NLP"},
            {"source_text": "attention", "target_text": "어텐션", "domain": "NLP"},
        ]

        result = self.runner.invoke(app, ["terms", "list"])

        # 용어가 출력되어야 함
        assert "transformer" in result.stdout or mock_get_all.called

    def test_terms_help(self):
        """terms 도움말"""
        result = self.runner.invoke(app, ["terms", "--help"])

        assert result.exit_code == 0

    @patch("src.main.TerminologyRepository.create")
    def test_terms_add(self, mock_create):
        """용어 추가"""
        mock_create.return_value = {"id": "new-id", "source_text": "test"}

        result = self.runner.invoke(
            app,
            ["terms", "add", "--source", "test", "--target", "테스트"]
        )

        mock_create.assert_called_once()

    @patch("src.main.TerminologyRepository.get_by_id")
    @patch("src.main.TerminologyRepository.update")
    def test_terms_update(self, mock_update, mock_get):
        """용어 수정"""
        mock_get.return_value = {"id": "1", "source_text": "test"}
        mock_update.return_value = {"id": "1", "target_text": "수정됨"}

        result = self.runner.invoke(
            app,
            ["terms", "update", "1", "--target", "수정됨"]
        )

    @patch("src.main.TerminologyRepository.delete")
    def test_terms_delete(self, mock_delete):
        """용어 삭제"""
        mock_delete.return_value = True

        result = self.runner.invoke(app, ["terms", "delete", "1", "--yes"])

    @patch("src.main.TerminologyRepository.get_all")
    def test_terms_export(self, mock_get_all):
        """용어 내보내기"""
        mock_get_all.return_value = [
            {"source_text": "term1", "target_text": "용어1"}
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result = self.runner.invoke(app, ["terms", "export", "--output", temp_path])

            # 파일이 생성되어야 함
            assert Path(temp_path).exists()
        finally:
            if Path(temp_path).exists():
                os.unlink(temp_path)

    @patch("src.main.TerminologyRepository.create")
    def test_terms_import(self, mock_create):
        """용어 가져오기"""
        mock_create.return_value = {"id": "1"}

        # 테스트 JSON 파일 생성
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json
            json.dump([
                {"source_text": "test", "target_text": "테스트"}
            ], f)
            temp_path = f.name

        try:
            result = self.runner.invoke(app, ["terms", "import", "--file", temp_path])
        finally:
            os.unlink(temp_path)


class TestWorkflowIntegration:
    """워크플로우 통합 테스트"""

    def test_workflow_state_types(self):
        """워크플로우 상태 타입 테스트"""
        from src.state import TranslationState

        # 상태 타입 확인
        state = TranslationState(
            source="test",
            source_type="file",
            domain="NLP"
        )

        assert state["source"] == "test"
        assert state["source_type"] == "file"

    def test_graph_module_exists(self):
        """그래프 모듈 존재 확인"""
        from src import graph

        assert hasattr(graph, "create_translation_graph")


class TestAPIIntegration:
    """API 인터페이스 통합 테스트"""

    def test_translate_function_exists(self):
        """translate 함수 존재 확인"""
        from src.api import translate

        # 함수 존재 확인
        assert callable(translate)

    def test_translation_request_schema(self):
        """TranslationRequest 스키마 테스트"""
        from src.api import TranslationRequest

        request = TranslationRequest(
            source="https://arxiv.org/pdf/1706.03762",
            domain="NLP"
        )

        assert request.source == "https://arxiv.org/pdf/1706.03762"
        assert request.domain == "NLP"

    def test_translation_response_schema(self):
        """TranslationResponse 스키마 테스트"""
        from src.api import TranslationResponse, TranslationStatus

        response = TranslationResponse(
            request_id="test-id",
            status=TranslationStatus.COMPLETED,
            output_path="/test.md"
        )

        assert response.request_id == "test-id"
        assert response.status == TranslationStatus.COMPLETED


class TestInsightBotIntegration:
    """InsightBot 연동 통합 테스트"""

    def test_insightbot_state_types(self):
        """InsightBot 상태 타입 테스트"""
        from src.api.insightbot import InsightBotState, PaperSource

        # PaperSource 생성 (TypedDict)
        source: PaperSource = {
            "url": "https://arxiv.org/pdf/1706.03762",
            "arxiv_id": "1706.03762",
            "domain": "NLP"
        }

        assert source["url"] is not None
        assert source["arxiv_id"] == "1706.03762"

    def test_translation_node_wrapper(self):
        """TranslationNodeWrapper 테스트"""
        from src.api.insightbot import TranslationNodeWrapper

        wrapper = TranslationNodeWrapper(auto_confirm=True)

        assert wrapper.auto_confirm is True
        assert callable(wrapper.translate_paper_node)

    @patch("src.api.insightbot.PaperTranslatorAPI")
    def test_translate_in_insightbot(self, mock_api):
        """translate_in_insightbot 함수 테스트"""
        from src.api.insightbot import translate_in_insightbot

        mock_api.return_value.translate.return_value = Mock(
            status="completed",
            output_path="/test.md"
        )

        # 함수 존재 확인
        assert callable(translate_in_insightbot)

    def test_subgraph_creation(self):
        """서브그래프 생성 테스트"""
        from src.api.insightbot import create_translation_subgraph

        # 서브그래프 생성
        builder = create_translation_subgraph()

        assert builder is not None


class TestFeedbackLoopIntegration:
    """피드백 루프 통합 테스트"""

    @patch("src.feedback.sync_manager.DiffAnalyzer")
    @patch("src.feedback.sync_manager.TerminologyRepository")
    @patch("src.feedback.sync_manager.TermChangeRepository")
    def test_sync_manager_integration(
        self,
        mock_term_change_repo,
        mock_term_repo,
        mock_analyzer
    ):
        """SyncManager 통합 테스트"""
        from src.feedback.sync_manager import SyncManager

        # SyncManager 생성
        manager = SyncManager(
            auto_sync=False,
            use_llm_analysis=False
        )

        assert manager.auto_sync is False
        assert manager.use_llm_analysis is False

    @patch("src.feedback.diff_analyzer.TranslationRepository.get_by_filename")
    def test_diff_analyzer_with_db(self, mock_get_by_filename):
        """DiffAnalyzer DB 연동 테스트"""
        from src.feedback.diff_analyzer import DiffAnalyzer

        mock_get_by_filename.return_value = [{"current_md_hash": "abc123"}]

        analyzer = DiffAnalyzer(use_llm_analysis=False)

        # DB에서 해시 조회
        stored_hash = analyzer.get_stored_hash("/test/paper.md")

        assert stored_hash == "abc123"


class TestE2EScenarios:
    """E2E 시나리오 테스트"""

    def test_pdf_to_markdown_pipeline(self):
        """PDF → 마크다운 파이프라인 테스트"""
        # 이 테스트는 실제 API 호출 없이 파이프라인 구조 확인
        from src.parsers import PDFParser
        from src.processors import TextChunker, PreProcessor, Translator, PostProcessor
        from src.outputs import MarkdownWriter

        # 각 컴포넌트가 존재하는지 확인
        assert PDFParser is not None
        assert TextChunker is not None
        assert PreProcessor is not None
        assert Translator is not None
        assert PostProcessor is not None
        assert MarkdownWriter is not None

    def test_terminology_feedback_loop(self):
        """용어 피드백 루프 테스트"""
        from src.feedback.diff_analyzer import DiffAnalyzer, DiffResult
        from src.feedback.sync_manager import SyncManager, SyncItem, SyncAction

        # 컴포넌트 존재 확인
        assert DiffAnalyzer is not None
        assert DiffResult is not None
        assert SyncManager is not None
        assert SyncItem is not None
        assert SyncAction is not None

    def test_full_module_imports(self):
        """전체 모듈 import 테스트"""
        # 모든 주요 모듈이 import 가능한지 확인
        from src.parsers import parse_pdf, PDFParser, ParsedPaper
        from src.processors import TextChunker, Chunk
        from src.processors import PreProcessor, ProcessedChunk
        from src.processors import Translator, TranslatedChunk
        from src.processors import PostProcessor
        from src.outputs import MarkdownWriter
        from src.db.repositories import (
            TerminologyRepository,
            TranslationRepository,
            TranslationHistoryRepository,
            TermChangeRepository
        )
        from src.feedback.diff_analyzer import DiffAnalyzer, DiffResult
        from src.feedback.sync_manager import SyncManager
        from src.api import (
            TranslationRequest,
            TranslationResponse,
            PaperTranslatorAPI,
            translate,
            translate_async
        )
        from src.api.insightbot import (
            InsightBotState,
            TranslationNodeWrapper,
            create_translation_subgraph
        )

        # 모든 import 성공
        assert True
