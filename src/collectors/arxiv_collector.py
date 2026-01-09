"""
ArXiv 논문 수집기
ArXiv API를 통한 논문 검색 및 수집
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Literal
from enum import Enum

import arxiv

logger = logging.getLogger(__name__)


class ArxivCategory(Enum):
    """ArXiv 카테고리"""
    # Computer Science
    CS_AI = "cs.AI"          # Artificial Intelligence
    CS_CL = "cs.CL"          # Computation and Language (NLP)
    CS_CV = "cs.CV"          # Computer Vision
    CS_LG = "cs.LG"          # Machine Learning
    CS_NE = "cs.NE"          # Neural and Evolutionary Computing
    CS_IR = "cs.IR"          # Information Retrieval
    # Statistics
    STAT_ML = "stat.ML"      # Machine Learning
    # Electrical Engineering
    EESS_AS = "eess.AS"      # Audio and Speech Processing


# 도메인별 카테고리 매핑
DOMAIN_CATEGORIES = {
    "NLP": [ArxivCategory.CS_CL, ArxivCategory.CS_AI],
    "CV": [ArxivCategory.CS_CV, ArxivCategory.CS_AI],
    "ML": [ArxivCategory.CS_LG, ArxivCategory.STAT_ML],
    "RL": [ArxivCategory.CS_AI, ArxivCategory.CS_LG],
    "Speech": [ArxivCategory.EESS_AS, ArxivCategory.CS_CL],
    "General": [ArxivCategory.CS_AI, ArxivCategory.CS_LG],
}


@dataclass
class ArxivPaper:
    """ArXiv 논문 정보"""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: datetime
    updated: datetime
    pdf_url: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    primary_category: str = ""

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "categories": self.categories,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "pdf_url": self.pdf_url,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "primary_category": self.primary_category,
        }


class ArxivCollector:
    """ArXiv 논문 수집기"""

    def __init__(self, max_results: int = 10):
        """
        Args:
            max_results: 기본 검색 결과 수
        """
        self.max_results = max_results
        self.client = arxiv.Client()

    def _result_to_paper(self, result: arxiv.Result) -> ArxivPaper:
        """arxiv.Result를 ArxivPaper로 변환"""
        return ArxivPaper(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title,
            authors=[author.name for author in result.authors],
            abstract=result.summary,
            categories=result.categories,
            published=result.published,
            updated=result.updated,
            pdf_url=result.pdf_url,
            comment=result.comment,
            journal_ref=result.journal_ref,
            primary_category=result.primary_category,
        )

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance",
        sort_order: Literal["ascending", "descending"] = "descending",
        categories: Optional[list[str]] = None,
    ) -> list[ArxivPaper]:
        """
        ArXiv 검색

        Args:
            query: 검색 쿼리 (제목, 초록, 저자 등)
            max_results: 최대 결과 수
            sort_by: 정렬 기준
            sort_order: 정렬 순서
            categories: 카테고리 필터 (예: ['cs.CL', 'cs.AI'])

        Returns:
            ArxivPaper 리스트
        """
        max_results = max_results or self.max_results

        # 카테고리 필터 추가
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query = f"({query}) AND ({cat_query})"

        # 정렬 기준 매핑
        sort_criterion = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }[sort_by]

        sort_order_enum = {
            "ascending": arxiv.SortOrder.Ascending,
            "descending": arxiv.SortOrder.Descending,
        }[sort_order]

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=sort_order_enum,
        )

        papers = []
        try:
            for result in self.client.results(search):
                papers.append(self._result_to_paper(result))
        except Exception as e:
            logger.error(f"ArXiv 검색 실패: {e}")
            raise

        logger.info(f"ArXiv 검색 완료: {len(papers)}개 논문 발견 (쿼리: {query})")
        return papers

    def search_by_domain(
        self,
        query: str,
        domain: str = "General",
        max_results: Optional[int] = None,
        sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance",
    ) -> list[ArxivPaper]:
        """
        도메인별 검색

        Args:
            query: 검색 쿼리
            domain: 도메인 (NLP, CV, ML, RL, Speech, General)
            max_results: 최대 결과 수
            sort_by: 정렬 기준

        Returns:
            ArxivPaper 리스트
        """
        categories = DOMAIN_CATEGORIES.get(domain, DOMAIN_CATEGORIES["General"])
        cat_values = [cat.value for cat in categories]

        return self.search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            categories=cat_values,
        )

    def get_recent(
        self,
        domain: str = "General",
        days: int = 7,
        max_results: Optional[int] = None,
    ) -> list[ArxivPaper]:
        """
        최근 논문 조회

        Args:
            domain: 도메인
            days: 최근 며칠 이내
            max_results: 최대 결과 수

        Returns:
            ArxivPaper 리스트
        """
        categories = DOMAIN_CATEGORIES.get(domain, DOMAIN_CATEGORIES["General"])
        cat_values = [cat.value for cat in categories]
        cat_query = " OR ".join([f"cat:{cat}" for cat in cat_values])

        return self.search(
            query=cat_query,
            max_results=max_results,
            sort_by="submittedDate",
            sort_order="descending",
        )

    def get_trending(
        self,
        domain: str = "General",
        max_results: Optional[int] = None,
    ) -> list[ArxivPaper]:
        """
        인기 논문 조회 (최근 업데이트 기준)

        Args:
            domain: 도메인
            max_results: 최대 결과 수

        Returns:
            ArxivPaper 리스트
        """
        # 인기 키워드로 검색
        trending_keywords = {
            "NLP": "large language model OR transformer OR GPT OR LLM",
            "CV": "diffusion OR vision transformer OR ViT OR image generation",
            "ML": "deep learning OR neural network OR optimization",
            "RL": "reinforcement learning OR policy gradient OR Q-learning",
            "General": "deep learning OR transformer OR neural network",
        }

        query = trending_keywords.get(domain, trending_keywords["General"])
        return self.search_by_domain(
            query=query,
            domain=domain,
            max_results=max_results,
            sort_by="submittedDate",
        )

    def get_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        ArXiv ID로 논문 조회

        Args:
            arxiv_id: ArXiv ID (예: "1706.03762")

        Returns:
            ArxivPaper 또는 None
        """
        search = arxiv.Search(id_list=[arxiv_id])

        try:
            for result in self.client.results(search):
                return self._result_to_paper(result)
        except Exception as e:
            logger.error(f"ArXiv 논문 조회 실패 ({arxiv_id}): {e}")

        return None


# 편의 함수
def search_arxiv(
    query: str,
    domain: str = "General",
    max_results: int = 10,
) -> list[ArxivPaper]:
    """ArXiv 검색 (단축 함수)"""
    collector = ArxivCollector(max_results=max_results)
    return collector.search_by_domain(query, domain, max_results)


def get_trending_papers(
    domain: str = "General",
    max_results: int = 10,
) -> list[ArxivPaper]:
    """인기 논문 조회 (단축 함수)"""
    collector = ArxivCollector(max_results=max_results)
    return collector.get_trending(domain, max_results)
