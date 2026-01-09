"""
API 모듈
외부 시스템(InsightBot 등)과의 연동 인터페이스 제공
"""
from .interface import (
    # Input/Output Schemas
    TranslationRequest,
    TranslationResponse,
    TranslationStatus,
    TranslationProgress,
    TranslationError,
    # Main Interface
    PaperTranslatorAPI,
    # Convenience Functions
    translate,
    translate_async,
    get_translation_status,
    cancel_translation,
)

from .insightbot import (
    # State Types
    InsightBotState,
    PaperSource,
    TranslationResult,
    # Node Functions
    parse_paper_source,
    validate_translation_request,
    execute_translation,
    format_translation_response,
    # Graph Builders
    create_translation_subgraph,
    compile_translation_subgraph,
    # Node Wrapper
    TranslationNodeWrapper,
    # Convenience Functions
    translate_in_insightbot,
    get_translation_node,
    get_translation_subgraph,
)

__all__ = [
    # === Interface Module ===
    # Schemas
    "TranslationRequest",
    "TranslationResponse",
    "TranslationStatus",
    "TranslationProgress",
    "TranslationError",
    # API Class
    "PaperTranslatorAPI",
    # Functions
    "translate",
    "translate_async",
    "get_translation_status",
    "cancel_translation",
    # === InsightBot Module ===
    # State Types
    "InsightBotState",
    "PaperSource",
    "TranslationResult",
    # Node Functions
    "parse_paper_source",
    "validate_translation_request",
    "execute_translation",
    "format_translation_response",
    # Graph Builders
    "create_translation_subgraph",
    "compile_translation_subgraph",
    # Node Wrapper
    "TranslationNodeWrapper",
    # Convenience Functions
    "translate_in_insightbot",
    "get_translation_node",
    "get_translation_subgraph",
]
