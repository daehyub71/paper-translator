"""
프로세서 모듈
텍스트 청킹, 전처리, 번역, 후처리 기능 제공
"""
from .chunker import (
    TextChunker,
    Chunk,
    chunk_text,
    chunk_sections,
)
from .pre_processor import (
    PreProcessor,
    ProcessedChunk,
    preprocess_chunk,
    preprocess_chunks,
    build_terminology_prompt,
)
from .translator import (
    Translator,
    TranslatedChunk,
    TranslationStatus,
    TranslationStats,
    translate_chunk,
    translate_chunks,
    estimate_translation_cost,
)
from .post_processor import (
    PostProcessor,
    PostProcessedChunk,
    PostProcessStats,
    TermValidation,
    TermMatchStatus,
    postprocess_chunk,
    postprocess_chunks,
    validate_terminology,
)

__all__ = [
    # Chunker
    "TextChunker",
    "Chunk",
    "chunk_text",
    "chunk_sections",
    # Pre-processor
    "PreProcessor",
    "ProcessedChunk",
    "preprocess_chunk",
    "preprocess_chunks",
    "build_terminology_prompt",
    # Translator
    "Translator",
    "TranslatedChunk",
    "TranslationStatus",
    "TranslationStats",
    "translate_chunk",
    "translate_chunks",
    "estimate_translation_cost",
    # Post-processor
    "PostProcessor",
    "PostProcessedChunk",
    "PostProcessStats",
    "TermValidation",
    "TermMatchStatus",
    "postprocess_chunk",
    "postprocess_chunks",
    "validate_terminology",
]
