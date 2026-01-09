"""
피드백 모듈
번역 결과 분석 및 DB 동기화 기능 제공
"""
from .diff_analyzer import (
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

from .sync_manager import (
    SyncManager,
    SyncResult,
    SyncItem,
    SyncAction,
    sync_translation_file,
    preview_sync,
    get_changed_files,
)

__all__ = [
    # Diff Analyzer
    "DiffAnalyzer",
    "DiffResult",
    "TextChange",
    "TermChange",
    "ChangeType",
    "analyze_file",
    "compare_files",
    "compare_content",
    "get_file_hash",
    # Sync Manager
    "SyncManager",
    "SyncResult",
    "SyncItem",
    "SyncAction",
    "sync_translation_file",
    "preview_sync",
    "get_changed_files",
]
