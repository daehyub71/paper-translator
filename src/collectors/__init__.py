"""
논문 수집 모듈
ArXiv API, Semantic Scholar API를 통한 논문 검색 및 수집
"""
from .arxiv_collector import (
    ArxivCollector,
    ArxivPaper,
    search_arxiv,
    get_trending_papers,
)

from .semantic_scholar_collector import (
    SemanticScholarCollector,
    SemanticScholarPaper,
    search_semantic_scholar,
    get_highly_cited_papers,
)

__all__ = [
    # ArXiv
    "ArxivCollector",
    "ArxivPaper",
    "search_arxiv",
    "get_trending_papers",
    # Semantic Scholar
    "SemanticScholarCollector",
    "SemanticScholarPaper",
    "search_semantic_scholar",
    "get_highly_cited_papers",
]
