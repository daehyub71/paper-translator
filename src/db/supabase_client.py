"""
Supabase 클라이언트 관리 모듈
싱글톤 패턴으로 연결 관리
"""
import os
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()


class SupabaseClient:
    """Supabase 클라이언트 싱글톤"""

    _instance: Optional["SupabaseClient"] = None
    _client: Optional[Client] = None

    def __new__(cls) -> "SupabaseClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """클라이언트 초기화"""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError(
                "SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다. "
                ".env 파일을 확인하세요."
            )

        self._client = create_client(url, key)

    @property
    def client(self) -> Client:
        """Supabase 클라이언트 반환"""
        if self._client is None:
            self._initialize()
        return self._client

    def table(self, table_name: str):
        """테이블 접근 단축 메서드"""
        return self.client.table(table_name)


# 편의를 위한 전역 인스턴스
_supabase: Optional[SupabaseClient] = None


def get_supabase() -> SupabaseClient:
    """Supabase 클라이언트 인스턴스 반환"""
    global _supabase
    if _supabase is None:
        _supabase = SupabaseClient()
    return _supabase


def get_client() -> Client:
    """Supabase Client 객체 직접 반환"""
    return get_supabase().client
