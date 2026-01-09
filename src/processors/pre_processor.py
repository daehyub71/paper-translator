"""
Pre-processor 모듈
번역 전 청크에 용어 프롬프트를 주입하는 전처리 로직
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from src.db.repositories import TerminologyRepository
from src.utils import settings
from .chunker import Chunk


@dataclass
class ProcessedChunk:
    """전처리된 청크"""
    chunk: Chunk                                    # 원본 청크
    matched_terms: list[dict] = field(default_factory=list)  # 매칭된 용어 목록
    terminology_prompt: str = ""                    # 용어 프롬프트
    context_hint: str = ""                          # 컨텍스트 힌트 (선택적)


class PreProcessor:
    """번역 전처리기"""

    # 용어 프롬프트 템플릿
    TERMINOLOGY_PROMPT_TEMPLATE = """다음 전문용어들은 아래 번역 규칙을 따라 번역해주세요:

{term_list}

위 용어들이 텍스트에 나타나면 반드시 지정된 번역어를 사용하세요."""

    # 개별 용어 포맷
    TERM_FORMAT = "- {source} → {target}"

    # 용어 포맷 (설명 포함)
    TERM_FORMAT_WITH_DESC = "- {source} → {target} ({description})"

    def __init__(
        self,
        pre_process_limit: Optional[int] = None,
        domain: Optional[str] = None,
        include_descriptions: bool = False
    ):
        """
        Args:
            pre_process_limit: 프롬프트에 포함할 최대 용어 수 (기본: settings에서)
            domain: 필터링할 도메인 (None이면 전체)
            include_descriptions: 용어 설명 포함 여부
        """
        self.pre_process_limit = pre_process_limit or settings.pre_process_limit
        self.domain = domain
        self.include_descriptions = include_descriptions

        # 캐시: 전체 용어 목록 (반복 조회 방지)
        self._term_cache: Optional[list[dict]] = None

    def get_all_terms(self, force_refresh: bool = False) -> list[dict]:
        """
        DB에서 전체 용어 목록 조회 (캐시됨)

        Args:
            force_refresh: True면 캐시 무시하고 새로 조회

        Returns:
            용어 목록
        """
        if self._term_cache is None or force_refresh:
            self._term_cache = TerminologyRepository.get_all(
                domain=self.domain,
                limit=1000  # 충분히 큰 수
            )
        return self._term_cache

    def get_terms_by_domain(self, domain: str) -> list[dict]:
        """
        특정 도메인의 용어 조회

        Args:
            domain: 도메인 (예: 'NLP', 'CV', 'RL', 'General')

        Returns:
            해당 도메인의 용어 목록
        """
        return TerminologyRepository.get_all(domain=domain, limit=500)

    def find_matching_terms(
        self,
        text: str,
        limit: Optional[int] = None
    ) -> list[dict]:
        """
        텍스트에서 매칭되는 용어 추출

        Args:
            text: 검색할 텍스트
            limit: 최대 반환 개수

        Returns:
            매칭된 용어 목록 (usage_count 내림차순)
        """
        limit = limit or self.pre_process_limit
        all_terms = self.get_all_terms()

        matched = []
        text_lower = text.lower()

        for term in all_terms:
            source = term.get("source_text", "")
            if source and source.lower() in text_lower:
                # 매칭 위치와 횟수 계산
                pattern = re.compile(re.escape(source), re.IGNORECASE)
                matches = pattern.findall(text)
                if matches:
                    term_copy = term.copy()
                    term_copy["match_count"] = len(matches)
                    matched.append(term_copy)

        # usage_count와 match_count 기준 정렬
        matched.sort(
            key=lambda x: (x.get("usage_count", 0), x.get("match_count", 0)),
            reverse=True
        )

        return matched[:limit]

    def build_terminology_prompt(
        self,
        terms: list[dict],
        include_descriptions: Optional[bool] = None
    ) -> str:
        """
        용어 목록으로 프롬프트 구성

        Args:
            terms: 용어 목록
            include_descriptions: 설명 포함 여부 (None이면 인스턴스 설정 사용)

        Returns:
            포맷된 용어 프롬프트
        """
        if not terms:
            return ""

        include_desc = include_descriptions if include_descriptions is not None else self.include_descriptions

        term_lines = []
        for term in terms:
            source = term.get("source_text", "")
            target = term.get("target_text", "")
            description = term.get("description", "")

            if include_desc and description:
                line = self.TERM_FORMAT_WITH_DESC.format(
                    source=source,
                    target=target,
                    description=description
                )
            else:
                line = self.TERM_FORMAT.format(source=source, target=target)

            term_lines.append(line)

        term_list = "\n".join(term_lines)
        return self.TERMINOLOGY_PROMPT_TEMPLATE.format(term_list=term_list)

    def build_context_hint(self, chunk: Chunk, paper_title: str = "") -> str:
        """
        청크에 대한 컨텍스트 힌트 생성

        Args:
            chunk: 처리할 청크
            paper_title: 논문 제목

        Returns:
            컨텍스트 힌트 문자열
        """
        hints = []

        if paper_title:
            hints.append(f"논문 제목: {paper_title}")

        if chunk.section_title and chunk.section_title != "Main Content":
            hints.append(f"현재 섹션: {chunk.section_title}")

        if chunk.section_index > 0:
            hints.append(f"섹션 내 {chunk.section_index + 1}번째 부분")

        return " | ".join(hints) if hints else ""

    def process_chunk(
        self,
        chunk: Chunk,
        paper_title: str = "",
        additional_terms: Optional[list[dict]] = None
    ) -> ProcessedChunk:
        """
        단일 청크 전처리

        Args:
            chunk: 처리할 청크
            paper_title: 논문 제목 (컨텍스트용)
            additional_terms: 추가 용어 목록 (수동 지정)

        Returns:
            ProcessedChunk 객체
        """
        # 매칭 용어 추출
        matched_terms = self.find_matching_terms(chunk.content)

        # 추가 용어 병합
        if additional_terms:
            existing_sources = {t.get("source_text", "").lower() for t in matched_terms}
            for term in additional_terms:
                if term.get("source_text", "").lower() not in existing_sources:
                    matched_terms.append(term)

        # 제한 적용
        matched_terms = matched_terms[:self.pre_process_limit]

        # 용어 프롬프트 생성
        terminology_prompt = self.build_terminology_prompt(matched_terms)

        # 컨텍스트 힌트 생성
        context_hint = self.build_context_hint(chunk, paper_title)

        return ProcessedChunk(
            chunk=chunk,
            matched_terms=matched_terms,
            terminology_prompt=terminology_prompt,
            context_hint=context_hint
        )

    def process_chunks(
        self,
        chunks: list[Chunk],
        paper_title: str = "",
        shared_terms: Optional[list[dict]] = None
    ) -> list[ProcessedChunk]:
        """
        여러 청크 일괄 전처리

        Args:
            chunks: 처리할 청크 목록
            paper_title: 논문 제목
            shared_terms: 모든 청크에 공통으로 적용할 용어

        Returns:
            ProcessedChunk 목록
        """
        processed = []

        for chunk in chunks:
            processed_chunk = self.process_chunk(
                chunk=chunk,
                paper_title=paper_title,
                additional_terms=shared_terms
            )
            processed.append(processed_chunk)

        return processed

    def get_unique_terms_from_chunks(
        self,
        chunks: list[Chunk],
        limit: Optional[int] = None
    ) -> list[dict]:
        """
        여러 청크에서 고유 용어 추출 (전체 문서 분석용)

        Args:
            chunks: 분석할 청크 목록
            limit: 최대 반환 개수

        Returns:
            고유 용어 목록
        """
        limit = limit or self.pre_process_limit * 2

        # 전체 텍스트 합치기
        full_text = " ".join(chunk.content for chunk in chunks)

        # 매칭 용어 추출
        return self.find_matching_terms(full_text, limit=limit)

    def get_processing_summary(
        self,
        processed_chunks: list[ProcessedChunk]
    ) -> dict:
        """
        전처리 요약 정보

        Args:
            processed_chunks: 전처리된 청크 목록

        Returns:
            요약 딕셔너리
        """
        all_terms = []
        for pc in processed_chunks:
            all_terms.extend(pc.matched_terms)

        # 중복 제거 (source_text 기준)
        unique_terms = {}
        for term in all_terms:
            source = term.get("source_text", "")
            if source not in unique_terms:
                unique_terms[source] = term

        return {
            "total_chunks": len(processed_chunks),
            "chunks_with_terms": sum(1 for pc in processed_chunks if pc.matched_terms),
            "total_term_matches": len(all_terms),
            "unique_terms": len(unique_terms),
            "top_terms": list(unique_terms.values())[:10],
        }


# 편의 함수
def preprocess_chunk(
    chunk: Chunk,
    domain: Optional[str] = None,
    paper_title: str = ""
) -> ProcessedChunk:
    """단일 청크 전처리 (단축 함수)"""
    preprocessor = PreProcessor(domain=domain)
    return preprocessor.process_chunk(chunk, paper_title)


def preprocess_chunks(
    chunks: list[Chunk],
    domain: Optional[str] = None,
    paper_title: str = ""
) -> list[ProcessedChunk]:
    """여러 청크 전처리 (단축 함수)"""
    preprocessor = PreProcessor(domain=domain)
    return preprocessor.process_chunks(chunks, paper_title)


def build_terminology_prompt(terms: list[dict]) -> str:
    """용어 목록으로 프롬프트 생성 (단축 함수)"""
    preprocessor = PreProcessor()
    return preprocessor.build_terminology_prompt(terms)
