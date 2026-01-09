"""
설정 관리 모듈
환경변수와 settings.yaml을 통합 관리
"""
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings:
    """애플리케이션 설정 클래스"""

    _instance: Optional["Settings"] = None
    _config: dict[str, Any] = {}

    def __new__(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """settings.yaml 로드"""
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}

    # OpenAI 설정
    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def openai_model(self) -> str:
        return os.getenv("OPENAI_MODEL", self._config.get("translation", {}).get("model", "gpt-4o-mini"))

    # Supabase 설정
    @property
    def supabase_url(self) -> str:
        return os.getenv("SUPABASE_URL", "")

    @property
    def supabase_key(self) -> str:
        return os.getenv("SUPABASE_KEY", "")

    @property
    def supabase_service_role_key(self) -> str:
        return os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    @property
    def supabase_database_url(self) -> str:
        return os.getenv("SUPABASE_DATABASE_URL", "")

    # Upstash Redis 설정
    @property
    def upstash_url(self) -> str:
        return os.getenv("UPSTASH_URL", "")

    @property
    def upstash_token(self) -> str:
        return os.getenv("UPSTASH_TOKEN", "")

    # InsightBot 설정
    @property
    def insightbot_api_url(self) -> str:
        return os.getenv("INSIGHTBOT_API_URL", "http://localhost:8000")

    # 번역 설정
    @property
    def translation_temperature(self) -> float:
        return self._config.get("translation", {}).get("temperature", 0.1)

    @property
    def max_tokens_per_chunk(self) -> int:
        return self._config.get("translation", {}).get("max_tokens_per_chunk", 4000)

    # 청킹 설정
    @property
    def chunking_strategy(self) -> str:
        return self._config.get("chunking", {}).get("strategy", "hybrid")

    @property
    def max_chunk_tokens(self) -> int:
        return self._config.get("chunking", {}).get("max_chunk_tokens", 3000)

    @property
    def overlap_tokens(self) -> int:
        return self._config.get("chunking", {}).get("overlap_tokens", 200)

    # PDF 파싱 설정
    @property
    def keep_formulas(self) -> bool:
        return self._config.get("pdf_parsing", {}).get("keep_formulas", True)

    @property
    def translate_tables(self) -> bool:
        return self._config.get("pdf_parsing", {}).get("translate_tables", True)

    @property
    def exclude_references(self) -> bool:
        return self._config.get("pdf_parsing", {}).get("exclude_references", True)

    # 용어 설정
    @property
    def pre_process_limit(self) -> int:
        return self._config.get("terminology", {}).get("pre_process_limit", 30)

    @property
    def post_process_threshold(self) -> float:
        return self._config.get("terminology", {}).get("post_process_threshold", 0.8)

    # 출력 설정
    @property
    def output_directory(self) -> str:
        return self._config.get("output", {}).get("directory", "./translations")

    @property
    def filename_format(self) -> str:
        return self._config.get("output", {}).get("filename_format", "{date}_{title}")

    # 피드백 설정
    @property
    def auto_sync(self) -> bool:
        return self._config.get("feedback", {}).get("auto_sync", False)

    # 캐시 설정
    @property
    def cache_enabled(self) -> bool:
        return self._config.get("cache", {}).get("enabled", True)

    @property
    def cache_ttl_seconds(self) -> int:
        return self._config.get("cache", {}).get("ttl_seconds", 86400)


# 싱글톤 인스턴스
settings = Settings()
