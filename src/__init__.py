"""
Paper Translator 메인 모듈
ArXiv 논문을 한국어로 번역하는 AI 파이프라인
"""
from .state import (
    TranslationState,
    TranslationConfig,
    TranslationStats,
    PaperMetadata,
    OutputInfo,
    create_initial_state,
    get_state_summary,
)

from .graph import (
    create_translation_graph,
    compile_graph,
    run_translation,
    run_translation_sync,
    translate_paper,
    translate_arxiv,
)


__all__ = [
    # State
    "TranslationState",
    "TranslationConfig",
    "TranslationStats",
    "PaperMetadata",
    "OutputInfo",
    "create_initial_state",
    "get_state_summary",
    # Graph
    "create_translation_graph",
    "compile_graph",
    "run_translation",
    "run_translation_sync",
    "translate_paper",
    "translate_arxiv",
]

__version__ = "0.1.0"
