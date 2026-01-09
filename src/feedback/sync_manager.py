"""
Sync Manager 모듈
번역 결과 변경 사항을 DB에 동기화
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from enum import Enum

from src.db.repositories import (
    TerminologyRepository,
    TranslationRepository,
    TermChangeRepository,
)
from src.feedback.diff_analyzer import (
    DiffAnalyzer,
    DiffResult,
    TermChange,
)

logger = logging.getLogger(__name__)


class SyncAction(Enum):
    """동기화 액션 유형"""
    UPDATE_TERM = "update_term"         # 용어 업데이트
    ADD_TERM = "add_term"               # 새 용어 추가
    LOG_CHANGE = "log_change"           # 변경 로그 기록
    UPDATE_HASH = "update_hash"         # 해시 업데이트
    SKIP = "skip"                       # 건너뜀


@dataclass
class SyncItem:
    """동기화 항목"""
    action: SyncAction
    description: str
    data: dict = field(default_factory=dict)
    applied: bool = False
    error: Optional[str] = None


@dataclass
class SyncResult:
    """동기화 결과"""
    file_path: str
    success: bool
    items: list[SyncItem] = field(default_factory=list)
    terms_updated: int = 0
    terms_added: int = 0
    changes_logged: int = 0
    hash_updated: bool = False
    error: Optional[str] = None
    synced_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SyncManager:
    """DB 동기화 관리자"""

    def __init__(
        self,
        auto_sync: bool = False,
        confirm_callback: Optional[Callable[[list[SyncItem]], bool]] = None,
        min_confidence: float = 0.7,
        use_llm_analysis: bool = True
    ):
        """
        Args:
            auto_sync: 자동 동기화 여부 (False면 확인 필요)
            confirm_callback: 사용자 확인 콜백 함수
            min_confidence: 최소 확신도 (이 이상만 반영)
            use_llm_analysis: LLM 분석 사용 여부
        """
        self.auto_sync = auto_sync
        self.confirm_callback = confirm_callback
        self.min_confidence = min_confidence
        self.use_llm_analysis = use_llm_analysis

        self._analyzer = DiffAnalyzer(
            use_llm_analysis=use_llm_analysis,
            min_confidence=min_confidence
        )

    def analyze_changes(
        self,
        file_path: str,
        original_content: Optional[str] = None
    ) -> DiffResult:
        """
        파일 변경 분석

        Args:
            file_path: 분석할 파일 경로
            original_content: 원본 콘텐츠 (없으면 DB에서 조회)

        Returns:
            DiffResult 객체
        """
        return self._analyzer.analyze_file(file_path, original_content)

    def prepare_sync_items(
        self,
        diff_result: DiffResult,
        translation_id: Optional[str] = None
    ) -> list[SyncItem]:
        """
        동기화 항목 준비

        Args:
            diff_result: Diff 분석 결과
            translation_id: 번역 기록 ID

        Returns:
            SyncItem 리스트
        """
        items = []

        if not diff_result.has_changes:
            return items

        # 1. 용어 변경 처리
        for term_change in diff_result.term_changes:
            if term_change.confidence < self.min_confidence:
                items.append(SyncItem(
                    action=SyncAction.SKIP,
                    description=f"확신도 부족: {term_change.source_text} ({term_change.confidence:.0%})",
                    data={"term_change": term_change}
                ))
                continue

            # 기존 용어 조회
            existing_terms = TerminologyRepository.search(term_change.source_text)

            if existing_terms:
                # 용어 업데이트
                existing = existing_terms[0]
                items.append(SyncItem(
                    action=SyncAction.UPDATE_TERM,
                    description=f"용어 업데이트: '{term_change.old_target}' → '{term_change.new_target}'",
                    data={
                        "term_id": existing.get("id"),
                        "source_text": term_change.source_text,
                        "old_target": existing.get("target_text"),
                        "new_target": term_change.new_target,
                        "confidence": term_change.confidence,
                    }
                ))
            else:
                # 새 용어 추가
                items.append(SyncItem(
                    action=SyncAction.ADD_TERM,
                    description=f"새 용어 추가: '{term_change.source_text}' → '{term_change.new_target}'",
                    data={
                        "source_text": term_change.source_text,
                        "target_text": term_change.new_target,
                        "domain": "General",  # 기본 도메인
                        "confidence": term_change.confidence,
                    }
                ))

            # 변경 로그 기록
            items.append(SyncItem(
                action=SyncAction.LOG_CHANGE,
                description=f"변경 로그: {term_change.source_text}",
                data={
                    "source_text": term_change.source_text,
                    "old_target": term_change.old_target,
                    "new_target": term_change.new_target,
                    "confidence": term_change.confidence,
                    "file_path": diff_result.file_path,
                }
            ))

        # 2. 해시 업데이트
        if translation_id:
            items.append(SyncItem(
                action=SyncAction.UPDATE_HASH,
                description=f"해시 업데이트: {diff_result.current_hash[:16]}...",
                data={
                    "translation_id": translation_id,
                    "new_hash": diff_result.current_hash,
                    "old_hash": diff_result.original_hash,
                }
            ))

        return items

    def apply_sync_items(self, items: list[SyncItem]) -> tuple[int, int, int]:
        """
        동기화 항목 적용

        Args:
            items: SyncItem 리스트

        Returns:
            (업데이트된 용어 수, 추가된 용어 수, 로깅된 변경 수)
        """
        updated = 0
        added = 0
        logged = 0

        for item in items:
            if item.action == SyncAction.SKIP:
                item.applied = True
                continue

            try:
                if item.action == SyncAction.UPDATE_TERM:
                    self._update_term(item)
                    updated += 1
                    item.applied = True

                elif item.action == SyncAction.ADD_TERM:
                    self._add_term(item)
                    added += 1
                    item.applied = True

                elif item.action == SyncAction.LOG_CHANGE:
                    self._log_change(item)
                    logged += 1
                    item.applied = True

                elif item.action == SyncAction.UPDATE_HASH:
                    self._update_hash(item)
                    item.applied = True

            except Exception as e:
                item.error = str(e)
                logger.error(f"동기화 실패: {item.description} - {e}")

        return updated, added, logged

    def _update_term(self, item: SyncItem):
        """용어 업데이트"""
        term_id = item.data.get("term_id")
        new_target = item.data.get("new_target")

        if not term_id or not new_target:
            raise ValueError("term_id와 new_target이 필요합니다")

        result = TerminologyRepository.update(term_id, target_text=new_target)

        if not result:
            raise Exception(f"용어 업데이트 실패: {term_id}")

        logger.info(f"용어 업데이트: {item.data.get('source_text')} → {new_target}")

    def _add_term(self, item: SyncItem):
        """새 용어 추가"""
        source_text = item.data.get("source_text")
        target_text = item.data.get("target_text")
        domain = item.data.get("domain", "General")
        confidence = item.data.get("confidence", 0.8)

        result = TerminologyRepository.create(
            source_text=source_text,
            target_text=target_text,
            domain=domain,
            confidence=confidence,
            is_user_defined=True
        )

        if not result:
            raise Exception(f"용어 추가 실패: {source_text}")

        logger.info(f"새 용어 추가: {source_text} → {target_text}")

    def _log_change(self, item: SyncItem):
        """변경 로그 기록"""
        log_data = {
            "source_text": item.data.get("source_text"),
            "old_target": item.data.get("old_target"),
            "new_target": item.data.get("new_target"),
            "change_reason": f"사용자 수정 (확신도: {item.data.get('confidence', 0):.0%})",
            "changed_at": datetime.now().isoformat(),
        }

        result = TermChangeRepository.create(log_data)

        if not result:
            logger.warning(f"변경 로그 기록 실패: {log_data['source_text']}")
        else:
            logger.info(f"변경 로그 기록: {log_data['source_text']}")

    def _update_hash(self, item: SyncItem):
        """해시 업데이트"""
        translation_id = item.data.get("translation_id")
        new_hash = item.data.get("new_hash")

        if not translation_id or not new_hash:
            raise ValueError("translation_id와 new_hash가 필요합니다")

        result = TranslationRepository.update(translation_id, {
            "current_md_hash": new_hash,
            "updated_at": datetime.now().isoformat(),
        })

        if not result:
            logger.warning(f"해시 업데이트 실패: {translation_id}")
        else:
            logger.info(f"해시 업데이트: {new_hash[:16]}...")

    def sync_file(
        self,
        file_path: str,
        original_content: Optional[str] = None,
        translation_id: Optional[str] = None,
        dry_run: bool = False
    ) -> SyncResult:
        """
        파일 동기화

        Args:
            file_path: 동기화할 파일 경로
            original_content: 원본 콘텐츠
            translation_id: 번역 기록 ID
            dry_run: True면 실제 변경 없이 미리보기만

        Returns:
            SyncResult 객체
        """
        try:
            # 1. 변경 분석
            diff_result = self.analyze_changes(file_path, original_content)

            if not diff_result.has_changes:
                return SyncResult(
                    file_path=file_path,
                    success=True,
                    error="변경 사항 없음"
                )

            # 2. 동기화 항목 준비
            items = self.prepare_sync_items(diff_result, translation_id)

            if not items:
                return SyncResult(
                    file_path=file_path,
                    success=True,
                    items=[],
                    error="동기화할 항목 없음"
                )

            # 3. Dry run이면 여기서 반환
            if dry_run:
                return SyncResult(
                    file_path=file_path,
                    success=True,
                    items=items,
                    error="Dry run 모드"
                )

            # 4. 자동 동기화가 아니면 확인 요청
            if not self.auto_sync:
                if self.confirm_callback:
                    confirmed = self.confirm_callback(items)
                    if not confirmed:
                        return SyncResult(
                            file_path=file_path,
                            success=False,
                            items=items,
                            error="사용자가 동기화를 취소했습니다"
                        )
                else:
                    # 콜백이 없으면 기본적으로 진행하지 않음
                    return SyncResult(
                        file_path=file_path,
                        success=False,
                        items=items,
                        error="확인이 필요합니다 (auto_sync=False)"
                    )

            # 5. 동기화 적용
            updated, added, logged = self.apply_sync_items(items)

            # 6. 결과 반환
            return SyncResult(
                file_path=file_path,
                success=True,
                items=items,
                terms_updated=updated,
                terms_added=added,
                changes_logged=logged,
                hash_updated=any(
                    i.action == SyncAction.UPDATE_HASH and i.applied
                    for i in items
                )
            )

        except Exception as e:
            logger.error(f"동기화 실패: {file_path} - {e}")
            return SyncResult(
                file_path=file_path,
                success=False,
                error=str(e)
            )

    def sync_files(
        self,
        file_paths: list[str],
        dry_run: bool = False
    ) -> list[SyncResult]:
        """
        여러 파일 동기화

        Args:
            file_paths: 파일 경로 목록
            dry_run: Dry run 모드

        Returns:
            SyncResult 리스트
        """
        results = []

        for file_path in file_paths:
            result = self.sync_file(file_path, dry_run=dry_run)
            results.append(result)

        return results

    def get_sync_summary(self, results: list[SyncResult]) -> dict:
        """
        동기화 결과 요약

        Args:
            results: SyncResult 리스트

        Returns:
            요약 딕셔너리
        """
        total_files = len(results)
        success_count = sum(1 for r in results if r.success)
        failed_count = total_files - success_count

        total_terms_updated = sum(r.terms_updated for r in results)
        total_terms_added = sum(r.terms_added for r in results)
        total_changes_logged = sum(r.changes_logged for r in results)

        return {
            "total_files": total_files,
            "success": success_count,
            "failed": failed_count,
            "terms_updated": total_terms_updated,
            "terms_added": total_terms_added,
            "changes_logged": total_changes_logged,
            "results": [
                {
                    "file": r.file_path,
                    "success": r.success,
                    "updated": r.terms_updated,
                    "added": r.terms_added,
                    "error": r.error
                }
                for r in results
            ]
        }

    def format_sync_preview(self, items: list[SyncItem]) -> str:
        """
        동기화 미리보기 포맷

        Args:
            items: SyncItem 리스트

        Returns:
            포맷된 문자열
        """
        lines = [
            "=" * 50,
            "동기화 미리보기",
            "=" * 50,
        ]

        # 액션별 그룹화
        by_action = {}
        for item in items:
            action = item.action.value
            if action not in by_action:
                by_action[action] = []
            by_action[action].append(item)

        action_labels = {
            "update_term": "📝 용어 업데이트",
            "add_term": "➕ 새 용어 추가",
            "log_change": "📋 변경 로그",
            "update_hash": "🔄 해시 업데이트",
            "skip": "⏭️ 건너뜀",
        }

        for action, action_items in by_action.items():
            label = action_labels.get(action, action)
            lines.append(f"\n{label} ({len(action_items)}건)")
            lines.append("-" * 40)

            for item in action_items[:5]:  # 최대 5개만 표시
                lines.append(f"  • {item.description}")

            if len(action_items) > 5:
                lines.append(f"  ... 외 {len(action_items) - 5}건")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)


# 편의 함수
def sync_translation_file(
    file_path: str,
    auto_sync: bool = False,
    dry_run: bool = False
) -> SyncResult:
    """번역 파일 동기화 (단축 함수)"""
    manager = SyncManager(auto_sync=auto_sync)
    return manager.sync_file(file_path, dry_run=dry_run)


def preview_sync(file_path: str) -> str:
    """동기화 미리보기 (단축 함수)"""
    manager = SyncManager(auto_sync=False, use_llm_analysis=False)
    diff_result = manager.analyze_changes(file_path)

    if not diff_result.has_changes:
        return "변경 사항 없음"

    items = manager.prepare_sync_items(diff_result)
    return manager.format_sync_preview(items)


def get_changed_files(output_dir: str = "./translations") -> list[str]:
    """변경된 파일 목록 조회 (단축 함수)"""
    output_path = Path(output_dir)
    changed_files = []

    if not output_path.exists():
        return changed_files

    for md_file in output_path.glob("*.md"):
        # DB에서 해시 조회
        filename = md_file.stem
        translations = TranslationRepository.get_by_filename(filename)

        if translations:
            stored_hash = translations[0].get("current_md_hash")
            if stored_hash:
                # 현재 파일 해시와 비교
                current_content = md_file.read_text(encoding="utf-8")
                current_hash = DiffAnalyzer.calculate_hash(current_content)

                if current_hash != stored_hash:
                    changed_files.append(str(md_file))
        else:
            # DB에 없는 파일도 변경된 것으로 간주
            changed_files.append(str(md_file))

    return changed_files
