"""
데이터베이스 Repository 모듈
각 테이블에 대한 CRUD 작업 제공
"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from .supabase_client import get_client


class TerminologyRepository:
    """전문용어 매핑 Repository"""

    TABLE = "terminology_mappings"

    @classmethod
    def get_all(
        cls,
        domain: Optional[str] = None,
        mapping_type: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """모든 용어 조회"""
        query = get_client().table(cls.TABLE).select("*")

        if domain:
            query = query.eq("domain", domain)
        if mapping_type:
            query = query.eq("mapping_type", mapping_type)

        response = query.order("usage_count", desc=True).limit(limit).execute()
        return response.data or []

    @classmethod
    def get_by_id(cls, term_id: str) -> Optional[dict]:
        """ID로 용어 조회"""
        response = get_client().table(cls.TABLE).select("*").eq("id", term_id).execute()
        return response.data[0] if response.data else None

    @classmethod
    def get_by_source(cls, source_text: str, domain: Optional[str] = None) -> Optional[dict]:
        """원문으로 용어 조회"""
        query = get_client().table(cls.TABLE).select("*").eq("source_text", source_text)

        if domain:
            query = query.eq("domain", domain)

        response = query.execute()
        return response.data[0] if response.data else None

    @classmethod
    def search(cls, keyword: str, domain: Optional[str] = None, limit: int = 50) -> list[dict]:
        """키워드로 용어 검색 (source_text, target_text 모두)"""
        query = get_client().table(cls.TABLE).select("*")

        if domain:
            query = query.eq("domain", domain)

        # ilike로 부분 매칭
        query = query.or_(f"source_text.ilike.%{keyword}%,target_text.ilike.%{keyword}%")

        response = query.limit(limit).execute()
        return response.data or []

    @classmethod
    def get_matching_terms(cls, text: str, domain: Optional[str] = None, limit: int = 30) -> list[dict]:
        """텍스트에서 매칭되는 용어 조회 (번역 시 사용)"""
        # 모든 용어 가져와서 텍스트에 포함된 것만 필터링
        all_terms = cls.get_all(domain=domain, limit=500)

        matching = []
        text_lower = text.lower()

        for term in all_terms:
            if term["source_text"].lower() in text_lower:
                matching.append(term)

        # usage_count 기준 정렬 후 limit 적용
        matching.sort(key=lambda x: x.get("usage_count", 0), reverse=True)
        return matching[:limit]

    @classmethod
    def create(
        cls,
        source_text: str,
        target_text: str,
        mapping_type: str = "word",
        domain: str = "General",
        confidence: float = 1.0,
        is_user_defined: bool = True
    ) -> Optional[dict]:
        """용어 생성"""
        data = {
            "source_text": source_text,
            "target_text": target_text,
            "mapping_type": mapping_type,
            "domain": domain,
            "confidence": confidence,
            "is_user_defined": is_user_defined,
            "usage_count": 0
        }

        response = get_client().table(cls.TABLE).insert(data).execute()
        return response.data[0] if response.data else None

    @classmethod
    def update(cls, term_id: str, **kwargs) -> Optional[dict]:
        """용어 수정"""
        # 허용된 필드만 업데이트
        allowed_fields = {"target_text", "mapping_type", "domain", "confidence", "is_user_defined"}
        update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not update_data:
            return None

        response = get_client().table(cls.TABLE).update(update_data).eq("id", term_id).execute()
        return response.data[0] if response.data else None

    @classmethod
    def delete(cls, term_id: str) -> bool:
        """용어 삭제"""
        response = get_client().table(cls.TABLE).delete().eq("id", term_id).execute()
        return bool(response.data)

    @classmethod
    def increment_usage(cls, term_id: str) -> None:
        """사용 횟수 증가"""
        term = cls.get_by_id(term_id)
        if term:
            new_count = term.get("usage_count", 0) + 1
            get_client().table(cls.TABLE).update({"usage_count": new_count}).eq("id", term_id).execute()

    @classmethod
    def bulk_increment_usage(cls, term_ids: list[str]) -> None:
        """여러 용어 사용 횟수 일괄 증가"""
        for term_id in term_ids:
            cls.increment_usage(term_id)


class TranslationRepository:
    """번역 기록 Repository"""

    TABLE = "translations"

    @classmethod
    def get_all(cls, status: Optional[str] = None, limit: int = 50) -> list[dict]:
        """모든 번역 기록 조회"""
        query = get_client().table(cls.TABLE).select("*")

        if status:
            query = query.eq("status", status)

        response = query.order("created_at", desc=True).limit(limit).execute()
        return response.data or []

    @classmethod
    def get_by_id(cls, translation_id: str) -> Optional[dict]:
        """ID로 번역 기록 조회"""
        response = get_client().table(cls.TABLE).select("*").eq("id", translation_id).execute()
        return response.data[0] if response.data else None

    @classmethod
    def get_by_arxiv_id(cls, arxiv_id: str) -> Optional[dict]:
        """ArXiv ID로 번역 기록 조회"""
        response = get_client().table(cls.TABLE).select("*").eq("arxiv_id", arxiv_id).execute()
        return response.data[0] if response.data else None

    @classmethod
    def create(
        cls,
        paper_title: str,
        output_path: str,
        paper_url: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        domain: Optional[str] = None,
        total_chunks: Optional[int] = None
    ) -> Optional[dict]:
        """번역 기록 생성"""
        data = {
            "paper_title": paper_title,
            "output_path": output_path,
            "paper_url": paper_url,
            "arxiv_id": arxiv_id,
            "domain": domain,
            "total_chunks": total_chunks,
            "status": "in_progress"
        }

        response = get_client().table(cls.TABLE).insert(data).execute()
        return response.data[0] if response.data else None

    @classmethod
    def update_status(cls, translation_id: str, status: str, error_message: Optional[str] = None) -> Optional[dict]:
        """번역 상태 업데이트"""
        update_data = {"status": status}
        if error_message:
            update_data["error_message"] = error_message

        response = get_client().table(cls.TABLE).update(update_data).eq("id", translation_id).execute()
        return response.data[0] if response.data else None

    @classmethod
    def update_hashes(cls, translation_id: str, original_hash: str, current_hash: Optional[str] = None) -> Optional[dict]:
        """MD 해시 업데이트"""
        update_data = {"original_md_hash": original_hash}
        if current_hash:
            update_data["current_md_hash"] = current_hash
        else:
            update_data["current_md_hash"] = original_hash

        response = get_client().table(cls.TABLE).update(update_data).eq("id", translation_id).execute()
        return response.data[0] if response.data else None

    @classmethod
    def update_current_hash(cls, translation_id: str, current_hash: str) -> Optional[dict]:
        """현재 MD 해시만 업데이트"""
        response = get_client().table(cls.TABLE).update(
            {"current_md_hash": current_hash}
        ).eq("id", translation_id).execute()
        return response.data[0] if response.data else None

    @classmethod
    def delete(cls, translation_id: str) -> bool:
        """번역 기록 삭제 (cascade로 history도 삭제됨)"""
        response = get_client().table(cls.TABLE).delete().eq("id", translation_id).execute()
        return bool(response.data)

    @classmethod
    def get_changed_translations(cls) -> list[dict]:
        """original_hash != current_hash인 번역 기록 조회"""
        response = get_client().table(cls.TABLE).select("*").neq(
            "original_md_hash", "current_md_hash"
        ).execute()
        return response.data or []

    @classmethod
    def get_by_filename(cls, filename: str) -> list[dict]:
        """파일명으로 번역 기록 조회 (output_path에서 파일명 매칭)"""
        response = get_client().table(cls.TABLE).select("*").ilike(
            "output_path", f"%{filename}%"
        ).execute()
        return response.data or []

    @classmethod
    def update(cls, translation_id: str, update_data: dict) -> Optional[dict]:
        """번역 기록 업데이트 (일반)"""
        if not update_data:
            return None

        response = get_client().table(cls.TABLE).update(update_data).eq("id", translation_id).execute()
        return response.data[0] if response.data else None


class TranslationHistoryRepository:
    """번역 히스토리 (청크별) Repository"""

    TABLE = "translation_history"

    @classmethod
    def get_by_translation(cls, translation_id: str) -> list[dict]:
        """번역 ID로 모든 청크 조회"""
        response = get_client().table(cls.TABLE).select("*").eq(
            "translation_id", translation_id
        ).order("chunk_index").execute()
        return response.data or []

    @classmethod
    def get_chunk(cls, translation_id: str, chunk_index: int) -> Optional[dict]:
        """특정 청크 조회"""
        response = get_client().table(cls.TABLE).select("*").eq(
            "translation_id", translation_id
        ).eq("chunk_index", chunk_index).execute()
        return response.data[0] if response.data else None

    @classmethod
    def create(
        cls,
        translation_id: str,
        chunk_index: int,
        original_text: str,
        translated_text: str,
        section_title: Optional[str] = None,
        terms_applied: Optional[list[dict]] = None,
        tokens_used: Optional[int] = None
    ) -> Optional[dict]:
        """청크 히스토리 생성"""
        data = {
            "translation_id": translation_id,
            "chunk_index": chunk_index,
            "original_text": original_text,
            "translated_text": translated_text,
            "section_title": section_title,
            "terms_applied": terms_applied,
            "tokens_used": tokens_used
        }

        response = get_client().table(cls.TABLE).insert(data).execute()
        return response.data[0] if response.data else None

    @classmethod
    def bulk_create(cls, chunks: list[dict]) -> list[dict]:
        """여러 청크 일괄 생성"""
        response = get_client().table(cls.TABLE).insert(chunks).execute()
        return response.data or []

    @classmethod
    def update_translated_text(cls, history_id: str, translated_text: str) -> Optional[dict]:
        """번역문 업데이트"""
        response = get_client().table(cls.TABLE).update(
            {"translated_text": translated_text}
        ).eq("id", history_id).execute()
        return response.data[0] if response.data else None

    @classmethod
    def delete_by_translation(cls, translation_id: str) -> bool:
        """번역 ID로 모든 청크 삭제"""
        response = get_client().table(cls.TABLE).delete().eq("translation_id", translation_id).execute()
        return bool(response.data)


class TermChangeRepository:
    """용어 변경 로그 Repository"""

    TABLE = "term_changes"

    @classmethod
    def create(cls, data: dict) -> Optional[dict]:
        """변경 로그 생성 (일반)"""
        # 필드 매핑 (sync_manager 호환)
        insert_data = {
            "change_type": data.get("change_type", "update"),
            "source_text": data.get("source_text"),
            "old_target_text": data.get("old_target"),
            "new_target_text": data.get("new_target"),
            "detected_from": data.get("change_reason", "user_feedback"),
        }

        response = get_client().table(cls.TABLE).insert(insert_data).execute()
        return response.data[0] if response.data else None

    @classmethod
    def get_all(cls, limit: int = 100) -> list[dict]:
        """모든 변경 로그 조회"""
        response = get_client().table(cls.TABLE).select("*").order(
            "created_at", desc=True
        ).limit(limit).execute()
        return response.data or []

    @classmethod
    def get_by_translation(cls, translation_id: str) -> list[dict]:
        """번역 ID로 변경 로그 조회"""
        response = get_client().table(cls.TABLE).select("*").eq(
            "translation_id", translation_id
        ).order("created_at", desc=True).execute()
        return response.data or []

    @classmethod
    def get_by_terminology(cls, terminology_id: str) -> list[dict]:
        """용어 ID로 변경 로그 조회"""
        response = get_client().table(cls.TABLE).select("*").eq(
            "terminology_id", terminology_id
        ).order("created_at", desc=True).execute()
        return response.data or []

    @classmethod
    def log_add(
        cls,
        source_text: str,
        new_target_text: str,
        terminology_id: Optional[str] = None,
        translation_id: Optional[str] = None,
        detected_from: str = "manual"
    ) -> Optional[dict]:
        """용어 추가 로그"""
        data = {
            "change_type": "add",
            "source_text": source_text,
            "new_target_text": new_target_text,
            "terminology_id": terminology_id,
            "translation_id": translation_id,
            "detected_from": detected_from
        }

        response = get_client().table(cls.TABLE).insert(data).execute()
        return response.data[0] if response.data else None

    @classmethod
    def log_update(
        cls,
        source_text: str,
        old_target_text: str,
        new_target_text: str,
        terminology_id: Optional[str] = None,
        translation_id: Optional[str] = None,
        detected_from: str = "manual"
    ) -> Optional[dict]:
        """용어 수정 로그"""
        data = {
            "change_type": "update",
            "source_text": source_text,
            "old_target_text": old_target_text,
            "new_target_text": new_target_text,
            "terminology_id": terminology_id,
            "translation_id": translation_id,
            "detected_from": detected_from
        }

        response = get_client().table(cls.TABLE).insert(data).execute()
        return response.data[0] if response.data else None

    @classmethod
    def log_delete(
        cls,
        source_text: str,
        old_target_text: str,
        terminology_id: Optional[str] = None,
        translation_id: Optional[str] = None,
        detected_from: str = "manual"
    ) -> Optional[dict]:
        """용어 삭제 로그"""
        data = {
            "change_type": "delete",
            "source_text": source_text,
            "old_target_text": old_target_text,
            "terminology_id": terminology_id,
            "translation_id": translation_id,
            "detected_from": detected_from
        }

        response = get_client().table(cls.TABLE).insert(data).execute()
        return response.data[0] if response.data else None
