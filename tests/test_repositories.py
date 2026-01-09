"""
Repository 단위 테스트
Supabase 클라이언트를 모킹하여 DB 연동 없이 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.db.repositories import (
    TerminologyRepository,
    TranslationRepository,
    TranslationHistoryRepository,
    TermChangeRepository,
)


# Mock Supabase client
@pytest.fixture
def mock_supabase():
    """Supabase 클라이언트 모킹"""
    with patch("src.db.repositories.get_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        yield mock_client


class TestTerminologyRepository:
    """TerminologyRepository 테스트"""

    # === 조회 테스트 ===

    def test_get_all(self, mock_supabase):
        """모든 용어 조회"""
        mock_response = Mock()
        mock_response.data = [
            {"id": "1", "source_text": "transformer", "target_text": "트랜스포머"},
            {"id": "2", "source_text": "attention", "target_text": "어텐션"},
        ]
        mock_supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

        result = TerminologyRepository.get_all()

        assert len(result) == 2
        assert result[0]["source_text"] == "transformer"

    def test_get_all_with_domain(self, mock_supabase):
        """도메인 필터링"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "domain": "NLP"}]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

        result = TerminologyRepository.get_all(domain="NLP")

        mock_supabase.table.return_value.select.return_value.eq.assert_called_with("domain", "NLP")
        assert len(result) == 1

    def test_get_all_empty(self, mock_supabase):
        """빈 결과"""
        mock_response = Mock()
        mock_response.data = None
        mock_supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

        result = TerminologyRepository.get_all()

        assert result == []

    def test_get_by_id(self, mock_supabase):
        """ID로 조회"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "source_text": "transformer"}]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response

        result = TerminologyRepository.get_by_id("1")

        assert result["id"] == "1"

    def test_get_by_id_not_found(self, mock_supabase):
        """ID로 조회 - 없음"""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response

        result = TerminologyRepository.get_by_id("nonexistent")

        assert result is None

    def test_get_by_source(self, mock_supabase):
        """원문으로 조회"""
        mock_response = Mock()
        mock_response.data = [{"source_text": "attention", "target_text": "어텐션"}]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response

        result = TerminologyRepository.get_by_source("attention")

        assert result["target_text"] == "어텐션"

    def test_search(self, mock_supabase):
        """키워드 검색"""
        mock_response = Mock()
        mock_response.data = [
            {"source_text": "transformer", "target_text": "트랜스포머"}
        ]
        mock_supabase.table.return_value.select.return_value.or_.return_value.limit.return_value.execute.return_value = mock_response

        result = TerminologyRepository.search("trans")

        assert len(result) == 1

    def test_get_matching_terms(self, mock_supabase):
        """텍스트에서 매칭 용어 조회"""
        mock_response = Mock()
        mock_response.data = [
            {"source_text": "transformer", "target_text": "트랜스포머", "usage_count": 10},
            {"source_text": "attention", "target_text": "어텐션", "usage_count": 5},
            {"source_text": "model", "target_text": "모델", "usage_count": 3},
        ]
        mock_supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

        text = "The transformer uses attention mechanism."
        result = TerminologyRepository.get_matching_terms(text)

        # transformer, attention이 텍스트에 있어야 함
        matched_sources = [t["source_text"] for t in result]
        assert "transformer" in matched_sources
        assert "attention" in matched_sources

    # === 생성 테스트 ===

    def test_create(self, mock_supabase):
        """용어 생성"""
        mock_response = Mock()
        mock_response.data = [{"id": "new-id", "source_text": "new term"}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response

        result = TerminologyRepository.create(
            source_text="new term",
            target_text="새 용어",
            domain="General"
        )

        assert result["id"] == "new-id"
        mock_supabase.table.return_value.insert.assert_called_once()

    def test_create_with_all_params(self, mock_supabase):
        """모든 파라미터로 용어 생성"""
        mock_response = Mock()
        mock_response.data = [{"id": "1"}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response

        result = TerminologyRepository.create(
            source_text="term",
            target_text="용어",
            mapping_type="phrase",
            domain="NLP",
            confidence=0.95,
            is_user_defined=True
        )

        # insert에 전달된 데이터 확인
        call_args = mock_supabase.table.return_value.insert.call_args[0][0]
        assert call_args["mapping_type"] == "phrase"
        assert call_args["domain"] == "NLP"
        assert call_args["confidence"] == 0.95

    # === 수정 테스트 ===

    def test_update(self, mock_supabase):
        """용어 수정"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "target_text": "수정됨"}]
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response

        result = TerminologyRepository.update("1", target_text="수정됨")

        assert result["target_text"] == "수정됨"

    def test_update_no_data(self, mock_supabase):
        """수정 데이터 없음"""
        result = TerminologyRepository.update("1")

        assert result is None

    def test_update_invalid_field(self, mock_supabase):
        """허용되지 않은 필드"""
        mock_response = Mock()
        mock_response.data = [{"id": "1"}]
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response

        result = TerminologyRepository.update("1", invalid_field="value", target_text="valid")

        # invalid_field는 무시되고 target_text만 업데이트
        call_args = mock_supabase.table.return_value.update.call_args[0][0]
        assert "invalid_field" not in call_args

    # === 삭제 테스트 ===

    def test_delete(self, mock_supabase):
        """용어 삭제"""
        mock_response = Mock()
        mock_response.data = [{"id": "1"}]
        mock_supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_response

        result = TerminologyRepository.delete("1")

        assert result is True

    def test_delete_not_found(self, mock_supabase):
        """삭제할 용어 없음"""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_response

        result = TerminologyRepository.delete("nonexistent")

        assert result is False

    # === 사용량 증가 테스트 ===

    def test_increment_usage(self, mock_supabase):
        """사용 횟수 증가"""
        # get_by_id 응답
        mock_get_response = Mock()
        mock_get_response.data = [{"id": "1", "usage_count": 5}]

        # update 응답
        mock_update_response = Mock()
        mock_update_response.data = [{"id": "1", "usage_count": 6}]

        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_get_response
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_update_response

        TerminologyRepository.increment_usage("1")

        # update가 호출되었는지 확인
        mock_supabase.table.return_value.update.assert_called()


class TestTranslationRepository:
    """TranslationRepository 테스트"""

    def test_get_all(self, mock_supabase):
        """모든 번역 기록 조회"""
        mock_response = Mock()
        mock_response.data = [
            {"id": "1", "paper_title": "Paper A"},
            {"id": "2", "paper_title": "Paper B"},
        ]
        mock_supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

        result = TranslationRepository.get_all()

        assert len(result) == 2

    def test_get_all_with_status(self, mock_supabase):
        """상태 필터링"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "status": "completed"}]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

        result = TranslationRepository.get_all(status="completed")

        mock_supabase.table.return_value.select.return_value.eq.assert_called_with("status", "completed")

    def test_get_by_id(self, mock_supabase):
        """ID로 조회"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "paper_title": "Test Paper"}]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response

        result = TranslationRepository.get_by_id("1")

        assert result["paper_title"] == "Test Paper"

    def test_get_by_arxiv_id(self, mock_supabase):
        """ArXiv ID로 조회"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "arxiv_id": "1706.03762"}]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response

        result = TranslationRepository.get_by_arxiv_id("1706.03762")

        assert result["arxiv_id"] == "1706.03762"

    def test_get_by_filename(self, mock_supabase):
        """파일명으로 조회"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "output_path": "/output/paper_test.md"}]
        mock_supabase.table.return_value.select.return_value.ilike.return_value.execute.return_value = mock_response

        result = TranslationRepository.get_by_filename("paper_test")

        assert len(result) == 1
        mock_supabase.table.return_value.select.return_value.ilike.assert_called_with("output_path", "%paper_test%")

    def test_create(self, mock_supabase):
        """번역 기록 생성"""
        mock_response = Mock()
        mock_response.data = [{"id": "new-id", "paper_title": "New Paper"}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response

        result = TranslationRepository.create(
            paper_title="New Paper",
            output_path="/output/paper.md"
        )

        assert result["id"] == "new-id"

    def test_update_status(self, mock_supabase):
        """상태 업데이트"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "status": "completed"}]
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response

        result = TranslationRepository.update_status("1", "completed")

        assert result["status"] == "completed"

    def test_update_hashes(self, mock_supabase):
        """해시 업데이트"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "original_md_hash": "abc", "current_md_hash": "def"}]
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response

        result = TranslationRepository.update_hashes("1", "abc", "def")

        assert result is not None

    def test_update_current_hash(self, mock_supabase):
        """현재 해시만 업데이트"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "current_md_hash": "new_hash"}]
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response

        result = TranslationRepository.update_current_hash("1", "new_hash")

        assert result["current_md_hash"] == "new_hash"

    def test_get_changed_translations(self, mock_supabase):
        """변경된 번역 기록 조회"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "original_md_hash": "abc", "current_md_hash": "def"}]
        mock_supabase.table.return_value.select.return_value.neq.return_value.execute.return_value = mock_response

        result = TranslationRepository.get_changed_translations()

        assert len(result) == 1
        mock_supabase.table.return_value.select.return_value.neq.assert_called_with("original_md_hash", "current_md_hash")

    def test_update_general(self, mock_supabase):
        """일반 업데이트"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "paper_title": "Updated"}]
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response

        result = TranslationRepository.update("1", {"paper_title": "Updated"})

        assert result["paper_title"] == "Updated"

    def test_update_general_empty(self, mock_supabase):
        """빈 업데이트"""
        result = TranslationRepository.update("1", {})

        assert result is None

    def test_delete(self, mock_supabase):
        """번역 기록 삭제"""
        mock_response = Mock()
        mock_response.data = [{"id": "1"}]
        mock_supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_response

        result = TranslationRepository.delete("1")

        assert result is True


class TestTranslationHistoryRepository:
    """TranslationHistoryRepository 테스트"""

    def test_get_by_translation(self, mock_supabase):
        """번역 ID로 모든 청크 조회"""
        mock_response = Mock()
        mock_response.data = [
            {"id": "1", "chunk_index": 0},
            {"id": "2", "chunk_index": 1},
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response

        result = TranslationHistoryRepository.get_by_translation("trans-1")

        assert len(result) == 2
        assert result[0]["chunk_index"] == 0

    def test_get_chunk(self, mock_supabase):
        """특정 청크 조회"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "chunk_index": 5}]
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_response

        result = TranslationHistoryRepository.get_chunk("trans-1", 5)

        assert result["chunk_index"] == 5

    def test_create(self, mock_supabase):
        """청크 히스토리 생성"""
        mock_response = Mock()
        mock_response.data = [{"id": "new-id", "chunk_index": 0}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response

        result = TranslationHistoryRepository.create(
            translation_id="trans-1",
            chunk_index=0,
            original_text="Original",
            translated_text="번역됨"
        )

        assert result["id"] == "new-id"

    def test_bulk_create(self, mock_supabase):
        """여러 청크 일괄 생성"""
        mock_response = Mock()
        mock_response.data = [{"id": "1"}, {"id": "2"}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response

        chunks = [
            {"translation_id": "t1", "chunk_index": 0, "original_text": "A", "translated_text": "가"},
            {"translation_id": "t1", "chunk_index": 1, "original_text": "B", "translated_text": "나"},
        ]

        result = TranslationHistoryRepository.bulk_create(chunks)

        assert len(result) == 2

    def test_update_translated_text(self, mock_supabase):
        """번역문 업데이트"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "translated_text": "수정됨"}]
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response

        result = TranslationHistoryRepository.update_translated_text("1", "수정됨")

        assert result["translated_text"] == "수정됨"

    def test_delete_by_translation(self, mock_supabase):
        """번역 ID로 모든 청크 삭제"""
        mock_response = Mock()
        mock_response.data = [{"id": "1"}, {"id": "2"}]
        mock_supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_response

        result = TranslationHistoryRepository.delete_by_translation("trans-1")

        assert result is True


class TestTermChangeRepository:
    """TermChangeRepository 테스트"""

    def test_create(self, mock_supabase):
        """변경 로그 생성 (일반)"""
        mock_response = Mock()
        mock_response.data = [{"id": "new-id"}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response

        result = TermChangeRepository.create({
            "source_text": "attention",
            "old_target": "어텐션",
            "new_target": "주의",
            "change_reason": "사용자 수정"
        })

        assert result["id"] == "new-id"
        # 필드 매핑 확인
        call_args = mock_supabase.table.return_value.insert.call_args[0][0]
        assert call_args["source_text"] == "attention"
        assert call_args["old_target_text"] == "어텐션"
        assert call_args["new_target_text"] == "주의"

    def test_get_all(self, mock_supabase):
        """모든 변경 로그 조회"""
        mock_response = Mock()
        mock_response.data = [{"id": "1"}, {"id": "2"}]
        mock_supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

        result = TermChangeRepository.get_all()

        assert len(result) == 2

    def test_get_by_translation(self, mock_supabase):
        """번역 ID로 변경 로그 조회"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "translation_id": "trans-1"}]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response

        result = TermChangeRepository.get_by_translation("trans-1")

        assert len(result) == 1

    def test_get_by_terminology(self, mock_supabase):
        """용어 ID로 변경 로그 조회"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "terminology_id": "term-1"}]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response

        result = TermChangeRepository.get_by_terminology("term-1")

        assert len(result) == 1

    def test_log_add(self, mock_supabase):
        """용어 추가 로그"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "change_type": "add"}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response

        result = TermChangeRepository.log_add(
            source_text="new term",
            new_target_text="새 용어"
        )

        call_args = mock_supabase.table.return_value.insert.call_args[0][0]
        assert call_args["change_type"] == "add"

    def test_log_update(self, mock_supabase):
        """용어 수정 로그"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "change_type": "update"}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response

        result = TermChangeRepository.log_update(
            source_text="attention",
            old_target_text="어텐션",
            new_target_text="주의"
        )

        call_args = mock_supabase.table.return_value.insert.call_args[0][0]
        assert call_args["change_type"] == "update"
        assert call_args["old_target_text"] == "어텐션"
        assert call_args["new_target_text"] == "주의"

    def test_log_delete(self, mock_supabase):
        """용어 삭제 로그"""
        mock_response = Mock()
        mock_response.data = [{"id": "1", "change_type": "delete"}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response

        result = TermChangeRepository.log_delete(
            source_text="removed term",
            old_target_text="삭제된 용어"
        )

        call_args = mock_supabase.table.return_value.insert.call_args[0][0]
        assert call_args["change_type"] == "delete"
