"""
Semantic Scholar 논문 수집기
Semantic Scholar API를 통한 인용수 기반 논문 검색
"""
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal

import requests

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Rate limit 초과 에러"""
    pass


# Semantic Scholar API 설정
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_RATE_LIMIT_DELAY = 3.0  # 초 (무료 API 제한 - 더 보수적으로)
S2_MAX_RETRIES = 3  # 최대 재시도 횟수
S2_RETRY_DELAY = 5.0  # 재시도 대기 시간 (초)


# 분야별 검색 키워드
DOMAIN_FIELDS = {
    "NLP": "Natural Language Processing",
    "CV": "Computer Vision",
    "ML": "Machine Learning",
    "RL": "Reinforcement Learning",
    "Speech": "Speech Recognition",
    "General": "Artificial Intelligence",
}


@dataclass
class SemanticScholarPaper:
    """Semantic Scholar 논문 정보"""
    paper_id: str
    title: str
    authors: list[str]
    abstract: Optional[str]
    year: Optional[int]
    citation_count: int
    influential_citation_count: int
    venue: Optional[str]
    url: str
    arxiv_id: Optional[str] = None
    pdf_url: Optional[str] = None
    fields_of_study: list[str] = field(default_factory=list)
    publication_date: Optional[str] = None

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "venue": self.venue,
            "url": self.url,
            "arxiv_id": self.arxiv_id,
            "pdf_url": self.pdf_url,
            "fields_of_study": self.fields_of_study,
            "publication_date": self.publication_date,
        }


class SemanticScholarCollector:
    """Semantic Scholar 논문 수집기"""

    # API 필드
    PAPER_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "year",
        "citationCount",
        "influentialCitationCount",
        "venue",
        "url",
        "externalIds",
        "fieldsOfStudy",
        "publicationDate",
        "authors.name",
        "openAccessPdf",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,
    ):
        """
        Args:
            api_key: Semantic Scholar API 키 (선택, 없으면 rate limit 적용)
                     환경변수 SEMANTIC_SCHOLAR_API_KEY로도 설정 가능
            max_results: 기본 검색 결과 수
        """
        # API 키: 인자 > 환경변수 > None
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.max_results = max_results
        self.session = requests.Session()

        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key
            logger.info("Semantic Scholar API 키 사용 중")

    def _make_request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> dict:
        """API 요청 (재시도 로직 포함)"""
        url = f"{S2_API_BASE}/{endpoint}"

        # 요청 전 rate limit 대기 (무료 API)
        if not self.api_key:
            time.sleep(S2_RATE_LIMIT_DELAY)

        for attempt in range(S2_MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=30)

                # 429 Too Many Requests - 재시도
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", S2_RETRY_DELAY))
                    wait_time = max(retry_after, S2_RETRY_DELAY * (attempt + 1))
                    logger.warning(
                        f"Rate limit 도달. {wait_time}초 후 재시도... ({attempt + 1}/{S2_MAX_RETRIES})"
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt < S2_MAX_RETRIES - 1:
                    logger.warning(f"요청 실패, 재시도 중... ({attempt + 1}/{S2_MAX_RETRIES}): {e}")
                    time.sleep(S2_RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Semantic Scholar API 요청 실패: {e}")
                    raise

        # 모든 재시도 실패 - rate limit인 경우 별도 에러
        raise RateLimitError(
            f"Semantic Scholar API rate limit 초과. "
            f"잠시 후 다시 시도하거나, API 키를 사용하세요. "
            f"(https://www.semanticscholar.org/product/api#api-key-form)"
        )

    def _parse_paper(self, data: dict) -> SemanticScholarPaper:
        """API 응답을 SemanticScholarPaper로 변환"""
        external_ids = data.get("externalIds", {}) or {}
        open_access_pdf = data.get("openAccessPdf", {}) or {}
        authors = data.get("authors", []) or []

        return SemanticScholarPaper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            authors=[a.get("name", "") for a in authors if a],
            abstract=data.get("abstract"),
            year=data.get("year"),
            citation_count=data.get("citationCount", 0) or 0,
            influential_citation_count=data.get("influentialCitationCount", 0) or 0,
            venue=data.get("venue"),
            url=data.get("url", ""),
            arxiv_id=external_ids.get("ArXiv"),
            pdf_url=open_access_pdf.get("url"),
            fields_of_study=data.get("fieldsOfStudy", []) or [],
            publication_date=data.get("publicationDate"),
        )

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        fields_of_study: Optional[list[str]] = None,
        min_citations: int = 0,
    ) -> list[SemanticScholarPaper]:
        """
        논문 검색

        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            year_from: 시작 연도
            year_to: 종료 연도
            fields_of_study: 연구 분야 필터
            min_citations: 최소 인용수

        Returns:
            SemanticScholarPaper 리스트
        """
        max_results = max_results or self.max_results

        params = {
            "query": query,
            "limit": min(max_results * 2, 100),  # 필터링 위해 더 많이 가져옴
            "fields": ",".join(self.PAPER_FIELDS),
        }

        # 연도 필터
        if year_from or year_to:
            year_filter = ""
            if year_from:
                year_filter = f"{year_from}-"
            if year_to:
                year_filter += str(year_to)
            elif year_from:
                year_filter += str(datetime.now().year)
            params["year"] = year_filter

        # 분야 필터
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        try:
            response = self._make_request("paper/search", params)
            papers_data = response.get("data", [])

            papers = []
            for paper_data in papers_data:
                paper = self._parse_paper(paper_data)

                # 인용수 필터
                if paper.citation_count >= min_citations:
                    papers.append(paper)

                if len(papers) >= max_results:
                    break

            logger.info(f"Semantic Scholar 검색 완료: {len(papers)}개 논문 발견")
            return papers

        except Exception as e:
            logger.error(f"Semantic Scholar 검색 실패: {e}")
            return []

    def search_by_domain(
        self,
        query: str,
        domain: str = "General",
        max_results: Optional[int] = None,
        year_from: Optional[int] = None,
        min_citations: int = 0,
    ) -> list[SemanticScholarPaper]:
        """
        도메인별 검색

        Args:
            query: 검색 쿼리
            domain: 도메인 (NLP, CV, ML, RL, Speech, General)
            max_results: 최대 결과 수
            year_from: 시작 연도
            min_citations: 최소 인용수

        Returns:
            SemanticScholarPaper 리스트
        """
        domain_field = DOMAIN_FIELDS.get(domain, DOMAIN_FIELDS["General"])
        full_query = f"{query} {domain_field}"

        return self.search(
            query=full_query,
            max_results=max_results,
            year_from=year_from,
            min_citations=min_citations,
        )

    def get_highly_cited(
        self,
        domain: str = "General",
        max_results: Optional[int] = None,
        year_from: Optional[int] = None,
        min_citations: int = 100,
    ) -> list[SemanticScholarPaper]:
        """
        고인용 논문 조회

        Args:
            domain: 도메인
            max_results: 최대 결과 수
            year_from: 시작 연도
            min_citations: 최소 인용수 (기본: 100)

        Returns:
            SemanticScholarPaper 리스트 (인용수 내림차순)
        """
        # 도메인별 인기 키워드
        domain_keywords = {
            "NLP": "transformer language model",
            "CV": "vision transformer image",
            "ML": "deep learning neural network",
            "RL": "reinforcement learning policy",
            "General": "deep learning artificial intelligence",
        }

        query = domain_keywords.get(domain, domain_keywords["General"])
        papers = self.search_by_domain(
            query=query,
            domain=domain,
            max_results=(max_results or self.max_results) * 3,  # 정렬 후 자르기
            year_from=year_from,
            min_citations=min_citations,
        )

        # 인용수 기준 정렬
        papers.sort(key=lambda p: p.citation_count, reverse=True)

        return papers[: max_results or self.max_results]

    def get_influential(
        self,
        domain: str = "General",
        max_results: Optional[int] = None,
        year_from: Optional[int] = None,
    ) -> list[SemanticScholarPaper]:
        """
        영향력 있는 논문 조회 (influential citation 기준)

        Args:
            domain: 도메인
            max_results: 최대 결과 수
            year_from: 시작 연도

        Returns:
            SemanticScholarPaper 리스트
        """
        domain_keywords = {
            "NLP": "attention transformer BERT GPT",
            "CV": "ResNet vision transformer diffusion",
            "ML": "deep learning optimization",
            "RL": "reinforcement learning DQN PPO",
            "General": "neural network deep learning",
        }

        query = domain_keywords.get(domain, domain_keywords["General"])
        papers = self.search_by_domain(
            query=query,
            domain=domain,
            max_results=(max_results or self.max_results) * 3,
            year_from=year_from,
        )

        # influential citation 기준 정렬
        papers.sort(key=lambda p: p.influential_citation_count, reverse=True)

        return papers[: max_results or self.max_results]

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[SemanticScholarPaper]:
        """
        ArXiv ID로 논문 조회

        Args:
            arxiv_id: ArXiv ID (예: "1706.03762")

        Returns:
            SemanticScholarPaper 또는 None
        """
        try:
            params = {"fields": ",".join(self.PAPER_FIELDS)}
            response = self._make_request(f"paper/arXiv:{arxiv_id}", params)
            return self._parse_paper(response)

        except Exception as e:
            logger.error(f"Semantic Scholar 논문 조회 실패 ({arxiv_id}): {e}")
            return None

    def get_paper_citations(
        self,
        paper_id: str,
        max_results: int = 10,
    ) -> list[SemanticScholarPaper]:
        """
        논문의 인용 논문 조회

        Args:
            paper_id: Semantic Scholar paper ID
            max_results: 최대 결과 수

        Returns:
            인용 논문 리스트
        """
        try:
            params = {
                "fields": ",".join(self.PAPER_FIELDS),
                "limit": max_results,
            }
            response = self._make_request(f"paper/{paper_id}/citations", params)

            papers = []
            for item in response.get("data", []):
                citing_paper = item.get("citingPaper", {})
                if citing_paper:
                    papers.append(self._parse_paper(citing_paper))

            return papers

        except Exception as e:
            logger.error(f"인용 논문 조회 실패: {e}")
            return []


# 편의 함수
def search_semantic_scholar(
    query: str,
    domain: str = "General",
    max_results: int = 10,
    min_citations: int = 0,
) -> list[SemanticScholarPaper]:
    """Semantic Scholar 검색 (단축 함수)"""
    collector = SemanticScholarCollector(max_results=max_results)
    return collector.search_by_domain(
        query=query,
        domain=domain,
        max_results=max_results,
        min_citations=min_citations,
    )


def get_highly_cited_papers(
    domain: str = "General",
    max_results: int = 10,
    min_citations: int = 100,
    year_from: Optional[int] = None,
) -> list[SemanticScholarPaper]:
    """고인용 논문 조회 (단축 함수)"""
    collector = SemanticScholarCollector(max_results=max_results)
    return collector.get_highly_cited(
        domain=domain,
        max_results=max_results,
        year_from=year_from,
        min_citations=min_citations,
    )
