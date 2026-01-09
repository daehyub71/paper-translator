"""
InsightBot 그래프 연동 모듈
InsightBot LangGraph 워크플로우에서 Paper Translator를 호출하기 위한 노드 래퍼
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TypedDict, Any, Literal
from enum import Enum

from langgraph.graph import StateGraph, END

from src.api.interface import (
    TranslationRequest,
    TranslationResponse,
    TranslationStatus,
    TranslationProgress,
    PaperTranslatorAPI,
)

logger = logging.getLogger(__name__)


# === InsightBot 상태 정의 ===

class PaperSource(TypedDict, total=False):
    """논문 소스 정보"""
    url: Optional[str]
    arxiv_id: Optional[str]
    file_path: Optional[str]
    title: Optional[str]
    domain: str


class TranslationResult(TypedDict, total=False):
    """번역 결과"""
    request_id: str
    status: str
    success: bool
    output_path: Optional[str]
    output_hash: Optional[str]
    title: Optional[str]
    title_ko: Optional[str]
    arxiv_id: Optional[str]
    stats: Optional[dict]
    error: Optional[dict]
    translated_at: Optional[str]


class InsightBotState(TypedDict, total=False):
    """
    InsightBot 통합 상태

    InsightBot 그래프에서 Paper Translator 노드와 상호작용할 때 사용하는 상태

    Attributes:
        messages: 대화 메시지 목록 (InsightBot 기본)
        paper_source: 번역할 논문 소스 정보
        translation_request: 번역 요청 (내부 사용)
        translation_result: 번역 결과
        translation_in_progress: 번역 진행 중 여부
        translation_progress: 진행 상황
        should_translate: 번역 필요 여부 (조건부 엣지용)
        user_confirmed: 사용자 확인 여부
    """
    messages: list[dict]
    paper_source: Optional[PaperSource]
    translation_request: Optional[dict]
    translation_result: Optional[TranslationResult]
    translation_in_progress: bool
    translation_progress: Optional[dict]
    should_translate: bool
    user_confirmed: bool


# === 노드 함수 정의 ===

def parse_paper_source(state: InsightBotState) -> InsightBotState:
    """
    논문 소스 파싱 노드

    사용자 메시지에서 논문 소스(URL, ArXiv ID 등)를 추출합니다.

    Args:
        state: InsightBot 상태

    Returns:
        업데이트된 상태 (paper_source 설정)
    """
    messages = state.get("messages", [])

    if not messages:
        return state

    # 마지막 사용자 메시지에서 소스 추출
    last_message = messages[-1] if messages else {}
    content = last_message.get("content", "")

    paper_source: PaperSource = {
        "domain": "General"
    }

    # URL 패턴 감지
    if "arxiv.org" in content or content.startswith("http"):
        import re
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, content)
        if urls:
            paper_source["url"] = urls[0]

            # ArXiv ID 추출 시도
            arxiv_pattern = r'(\d{4}\.\d{4,5})'
            arxiv_match = re.search(arxiv_pattern, urls[0])
            if arxiv_match:
                paper_source["arxiv_id"] = arxiv_match.group(1)

    # ArXiv ID 직접 입력 감지
    elif any(char.isdigit() for char in content):
        import re
        arxiv_pattern = r'(\d{4}\.\d{4,5})'
        arxiv_match = re.search(arxiv_pattern, content)
        if arxiv_match:
            paper_source["arxiv_id"] = arxiv_match.group(1)

    # 도메인 감지
    content_lower = content.lower()
    if "nlp" in content_lower or "언어" in content_lower or "language" in content_lower:
        paper_source["domain"] = "NLP"
    elif "cv" in content_lower or "vision" in content_lower or "이미지" in content_lower:
        paper_source["domain"] = "CV"
    elif "rl" in content_lower or "reinforcement" in content_lower or "강화" in content_lower:
        paper_source["domain"] = "RL"

    state["paper_source"] = paper_source
    state["should_translate"] = bool(paper_source.get("url") or paper_source.get("arxiv_id"))

    logger.info(f"논문 소스 파싱 완료: {paper_source}")

    return state


def validate_translation_request(state: InsightBotState) -> InsightBotState:
    """
    번역 요청 검증 노드

    논문 소스가 유효한지 확인하고 번역 요청을 준비합니다.

    Args:
        state: InsightBot 상태

    Returns:
        업데이트된 상태 (translation_request 설정)
    """
    paper_source = state.get("paper_source")

    if not paper_source:
        state["should_translate"] = False
        return state

    # 소스 결정 (우선순위: arxiv_id > url > file_path)
    source = (
        paper_source.get("arxiv_id") or
        paper_source.get("url") or
        paper_source.get("file_path")
    )

    if not source:
        state["should_translate"] = False
        return state

    # 번역 요청 생성
    state["translation_request"] = {
        "source": source,
        "domain": paper_source.get("domain", "General"),
        "exclude_references": True,
        "extract_tables": True,
    }

    state["should_translate"] = True
    logger.info(f"번역 요청 준비 완료: {source}")

    return state


def execute_translation(state: InsightBotState) -> InsightBotState:
    """
    번역 실행 노드

    Paper Translator API를 호출하여 번역을 실행합니다.

    Args:
        state: InsightBot 상태

    Returns:
        업데이트된 상태 (translation_result 설정)
    """
    translation_request = state.get("translation_request")

    if not translation_request:
        state["translation_result"] = {
            "success": False,
            "status": "failed",
            "error": {"code": "NO_REQUEST", "message": "번역 요청이 없습니다"}
        }
        return state

    state["translation_in_progress"] = True

    try:
        # API 호출
        api = PaperTranslatorAPI()

        request = TranslationRequest(
            source=translation_request["source"],
            domain=translation_request.get("domain", "General"),
            exclude_references=translation_request.get("exclude_references", True),
            extract_tables=translation_request.get("extract_tables", True),
        )

        # 진행 상황 콜백
        def progress_callback(progress: TranslationProgress):
            state["translation_progress"] = progress.to_dict()

        response = api.translate(request, progress_callback=progress_callback)

        # 결과 저장
        state["translation_result"] = {
            "request_id": response.request_id,
            "status": response.status.value,
            "success": response.success,
            "output_path": response.output_path,
            "output_hash": response.output_hash,
            "title": response.title,
            "title_ko": response.title_ko,
            "arxiv_id": response.arxiv_id,
            "stats": response.stats,
            "error": response.error.to_dict() if response.error else None,
            "translated_at": datetime.now().isoformat(),
        }

        logger.info(f"번역 완료: {response.status.value}")

    except Exception as e:
        logger.exception(f"번역 실행 실패: {e}")
        state["translation_result"] = {
            "success": False,
            "status": "failed",
            "error": {"code": "EXECUTION_ERROR", "message": str(e)}
        }

    finally:
        state["translation_in_progress"] = False

    return state


def format_translation_response(state: InsightBotState) -> InsightBotState:
    """
    번역 결과 포맷팅 노드

    번역 결과를 InsightBot 메시지 형식으로 변환합니다.

    Args:
        state: InsightBot 상태

    Returns:
        업데이트된 상태 (messages에 결과 추가)
    """
    translation_result = state.get("translation_result")
    messages = state.get("messages", [])

    if not translation_result:
        messages.append({
            "role": "assistant",
            "content": "번역 결과를 찾을 수 없습니다."
        })
        state["messages"] = messages
        return state

    if translation_result.get("success"):
        # 성공 메시지
        title = translation_result.get("title", "제목 없음")
        title_ko = translation_result.get("title_ko", "")
        output_path = translation_result.get("output_path", "")
        stats = translation_result.get("stats", {})

        content = f"""번역이 완료되었습니다!

**논문**: {title}
{f"**한국어 제목**: {title_ko}" if title_ko else ""}
**출력 파일**: {output_path}

**번역 통계**:
- 총 청크: {stats.get('total_chunks', 'N/A')}개
- 총 토큰: {stats.get('total_tokens', 0):,}
- 예상 비용: {stats.get('estimated_cost_usd', 'N/A')}
- 용어 매칭률: {stats.get('term_match_rate', 'N/A')}
"""
    else:
        # 실패 메시지
        error = translation_result.get("error", {})
        content = f"""번역에 실패했습니다.

**오류 코드**: {error.get('code', 'UNKNOWN')}
**오류 메시지**: {error.get('message', '알 수 없는 오류')}
"""

    messages.append({
        "role": "assistant",
        "content": content
    })

    state["messages"] = messages
    return state


# === 조건부 엣지 함수 ===

def should_translate(state: InsightBotState) -> Literal["translate", "skip"]:
    """번역 필요 여부 확인"""
    if state.get("should_translate") and state.get("user_confirmed", True):
        return "translate"
    return "skip"


def check_translation_result(state: InsightBotState) -> Literal["success", "failed"]:
    """번역 결과 확인"""
    result = state.get("translation_result", {})
    if result.get("success"):
        return "success"
    return "failed"


# === 그래프 빌더 ===

def create_translation_subgraph() -> StateGraph:
    """
    번역 서브그래프 생성

    InsightBot 메인 그래프에 통합할 수 있는 번역 서브그래프

    Returns:
        컴파일 가능한 StateGraph
    """
    workflow = StateGraph(InsightBotState)

    # 노드 추가
    workflow.add_node("parse_source", parse_paper_source)
    workflow.add_node("validate_request", validate_translation_request)
    workflow.add_node("execute_translation", execute_translation)
    workflow.add_node("format_response", format_translation_response)

    # 시작점
    workflow.set_entry_point("parse_source")

    # 엣지 연결
    workflow.add_edge("parse_source", "validate_request")
    workflow.add_conditional_edges(
        "validate_request",
        should_translate,
        {
            "translate": "execute_translation",
            "skip": "format_response"
        }
    )
    workflow.add_edge("execute_translation", "format_response")
    workflow.add_edge("format_response", END)

    return workflow


def compile_translation_subgraph():
    """서브그래프 컴파일"""
    return create_translation_subgraph().compile()


# === 노드 래퍼 (InsightBot 통합용) ===

class TranslationNodeWrapper:
    """
    InsightBot 그래프에서 사용할 수 있는 번역 노드 래퍼

    Examples:
        # InsightBot 그래프에 통합
        from src.api.insightbot import TranslationNodeWrapper

        wrapper = TranslationNodeWrapper()

        # 단일 노드로 사용
        insight_graph.add_node("translate_paper", wrapper.translate_paper_node)

        # 또는 서브그래프로 통합
        translation_subgraph = wrapper.get_subgraph()
        insight_graph.add_node("translation_flow", translation_subgraph)
    """

    def __init__(self, auto_confirm: bool = True):
        """
        Args:
            auto_confirm: 자동 확인 모드 (True면 사용자 확인 없이 번역)
        """
        self.auto_confirm = auto_confirm
        self._api = PaperTranslatorAPI()
        self._subgraph = None

    def translate_paper_node(self, state: InsightBotState) -> InsightBotState:
        """
        단일 번역 노드

        InsightBot 그래프에서 단일 노드로 사용할 수 있는 번역 함수

        Args:
            state: InsightBot 상태 (paper_source 필요)

        Returns:
            업데이트된 상태 (translation_result 포함)
        """
        # 소스 파싱
        state = parse_paper_source(state)

        # 요청 검증
        state = validate_translation_request(state)

        # 번역 실행
        if state.get("should_translate"):
            state["user_confirmed"] = self.auto_confirm
            state = execute_translation(state)

        # 결과 포맷팅
        state = format_translation_response(state)

        return state

    def get_subgraph(self):
        """
        번역 서브그래프 반환

        InsightBot 그래프에 서브그래프로 통합할 때 사용

        Returns:
            컴파일된 서브그래프
        """
        if self._subgraph is None:
            self._subgraph = compile_translation_subgraph()
        return self._subgraph

    def create_translation_request(
        self,
        source: str,
        domain: str = "General"
    ) -> InsightBotState:
        """
        번역 요청 상태 생성 헬퍼

        Args:
            source: 논문 소스 (URL, ArXiv ID 등)
            domain: 도메인

        Returns:
            초기 InsightBot 상태
        """
        return InsightBotState(
            messages=[{
                "role": "user",
                "content": f"논문 번역: {source}"
            }],
            paper_source={
                "arxiv_id": source if source.replace(".", "").replace("/", "").isdigit() else None,
                "url": source if source.startswith("http") else None,
                "domain": domain
            },
            should_translate=True,
            user_confirmed=self.auto_confirm,
            translation_in_progress=False,
        )


# === 편의 함수 ===

def translate_in_insightbot(
    source: str,
    domain: str = "General",
    messages: Optional[list[dict]] = None
) -> InsightBotState:
    """
    InsightBot 컨텍스트에서 번역 실행

    Args:
        source: 논문 소스
        domain: 도메인
        messages: 기존 메시지 (있는 경우)

    Returns:
        번역 결과가 포함된 InsightBot 상태

    Examples:
        # 간단한 번역
        result = translate_in_insightbot("1706.03762", domain="NLP")
        print(result["translation_result"])

        # 기존 대화에 추가
        messages = [{"role": "user", "content": "이 논문 번역해줘"}]
        result = translate_in_insightbot("1706.03762", messages=messages)
    """
    wrapper = TranslationNodeWrapper(auto_confirm=True)

    initial_state: InsightBotState = {
        "messages": messages or [{
            "role": "user",
            "content": f"논문 번역: {source}"
        }],
        "paper_source": {
            "arxiv_id": source if source.replace(".", "").replace("/", "").isdigit() else None,
            "url": source if source.startswith("http") else None,
            "domain": domain
        },
        "should_translate": True,
        "user_confirmed": True,
        "translation_in_progress": False,
    }

    return wrapper.translate_paper_node(initial_state)


def get_translation_node():
    """
    InsightBot 그래프에 추가할 번역 노드 반환

    Returns:
        노드 함수

    Examples:
        # InsightBot 그래프에 추가
        from src.api.insightbot import get_translation_node

        insight_graph = StateGraph(InsightBotState)
        insight_graph.add_node("translate", get_translation_node())
    """
    wrapper = TranslationNodeWrapper(auto_confirm=True)
    return wrapper.translate_paper_node


def get_translation_subgraph():
    """
    InsightBot 그래프에 통합할 번역 서브그래프 반환

    Returns:
        컴파일된 서브그래프

    Examples:
        # InsightBot 그래프에 서브그래프로 추가
        from src.api.insightbot import get_translation_subgraph

        translation_flow = get_translation_subgraph()
        insight_graph.add_node("translation_flow", translation_flow)
    """
    return compile_translation_subgraph()
