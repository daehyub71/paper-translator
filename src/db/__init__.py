"""
데이터베이스 모듈
Supabase 클라이언트 및 Repository 제공
"""
from .supabase_client import get_supabase, get_client, SupabaseClient
from .repositories import (
    TerminologyRepository,
    TranslationRepository,
    TranslationHistoryRepository,
    TermChangeRepository,
)

__all__ = [
    "get_supabase",
    "get_client",
    "SupabaseClient",
    "TerminologyRepository",
    "TranslationRepository",
    "TranslationHistoryRepository",
    "TermChangeRepository",
]
