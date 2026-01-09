"""
InsightBot 호출 인터페이스 모듈
외부 시스템에서 Paper Translator를 호출하기 위한 API 제공
"""
import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any, Dict
from threading import Lock

logger = logging.getLogger(__name__)


# === 열거형 정의 ===

class TranslationStatus(Enum):
    """번역 상태"""
    PENDING = "pending"           # 대기 중
    RUNNING = "running"           # 실행 중
    COMPLETED = "completed"       # 완료
    FAILED = "failed"             # 실패
    CANCELLED = "cancelled"       # 취소됨


class SourceType(Enum):
    """소스 타입"""
    URL = "url"                   # PDF URL
    ARXIV_ID = "arxiv_id"         # ArXiv ID
    LOCAL_FILE = "local_file"    # 로컬 파일


# === 입력 스키마 ===

@dataclass
class TranslationRequest:
    """
    번역 요청 스키마

    InsightBot이나 외부 시스템에서 Paper Translator를 호출할 때 사용하는 입력 형식

    Attributes:
        source: PDF URL, ArXiv ID, 또는 로컬 파일 경로
        domain: 용어 도메인 (NLP, CV, RL, General)
        exclude_references: References 섹션 제외 여부
        extract_tables: 표 추출 여부
        callback_url: 완료 시 호출할 웹훅 URL (선택)
        metadata: 추가 메타데이터 (선택)

    Examples:
        # ArXiv ID로 번역 요청
        request = TranslationRequest(
            source="1706.03762",
            domain="NLP"
        )

        # URL로 번역 요청
        request = TranslationRequest(
            source="https://arxiv.org/pdf/1706.03762.pdf",
            domain="NLP",
            exclude_references=True
        )

        # 로컬 파일 번역 요청
        request = TranslationRequest(
            source="/path/to/paper.pdf",
            domain="CV"
        )
    """
    source: str
    domain: str = "General"
    exclude_references: bool = True
    extract_tables: bool = True
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 내부 사용
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_source_type(self) -> SourceType:
        """소스 타입 판별"""
        if self.source.startswith("http"):
            return SourceType.URL
        elif self.source.replace(".", "").replace("/", "").isdigit():
            return SourceType.ARXIV_ID
        else:
            return SourceType.LOCAL_FILE

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        요청 유효성 검증

        Returns:
            (유효 여부, 에러 메시지)
        """
        if not self.source:
            return False, "source는 필수입니다"

        if not self.source.strip():
            return False, "source가 비어있습니다"

        source_type = self.get_source_type()

        # 로컬 파일인 경우 존재 여부 확인
        if source_type == SourceType.LOCAL_FILE:
            if not Path(self.source).exists():
                return False, f"파일을 찾을 수 없습니다: {self.source}"

        # 도메인 검증
        valid_domains = {"NLP", "CV", "RL", "General"}
        if self.domain not in valid_domains:
            return False, f"유효하지 않은 도메인: {self.domain}. 허용: {valid_domains}"

        return True, None

    def to_dict(self) -> dict:
        """딕셔너리 변환"""
        return {
            "request_id": self.request_id,
            "source": self.source,
            "source_type": self.get_source_type().value,
            "domain": self.domain,
            "exclude_references": self.exclude_references,
            "extract_tables": self.extract_tables,
            "callback_url": self.callback_url,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


# === 출력 스키마 ===

@dataclass
class TranslationProgress:
    """
    번역 진행 상황

    Attributes:
        current_node: 현재 실행 중인 노드
        completed_nodes: 완료된 노드 목록
        total_nodes: 전체 노드 수
        progress_percent: 진행률 (0-100)
        message: 현재 상태 메시지
    """
    current_node: str = ""
    completed_nodes: list[str] = field(default_factory=list)
    total_nodes: int = 8
    progress_percent: float = 0.0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "current_node": self.current_node,
            "completed_nodes": self.completed_nodes,
            "total_nodes": self.total_nodes,
            "progress_percent": self.progress_percent,
            "message": self.message,
        }


@dataclass
class TranslationError:
    """
    번역 에러 정보

    Attributes:
        code: 에러 코드
        message: 에러 메시지
        node: 에러 발생 노드
        details: 상세 정보
    """
    code: str
    message: str
    node: Optional[str] = None
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "node": self.node,
            "details": self.details,
        }


@dataclass
class TranslationResponse:
    """
    번역 응답 스키마

    InsightBot이나 외부 시스템으로 반환되는 결과 형식

    Attributes:
        request_id: 요청 ID
        status: 현재 상태
        output_path: 번역 결과 파일 경로
        output_hash: MD5 해시
        title: 논문 제목
        title_ko: 한국어 제목 (있는 경우)
        arxiv_id: ArXiv ID (있는 경우)
        stats: 번역 통계
        progress: 진행 상황
        error: 에러 정보 (실패 시)
        created_at: 요청 생성 시간
        completed_at: 완료 시간

    Examples:
        # 성공 응답
        response = TranslationResponse(
            request_id="abc-123",
            status=TranslationStatus.COMPLETED,
            output_path="/output/translated.md",
            title="Attention Is All You Need",
            stats={"total_tokens": 10000, "cost_usd": "$0.15"}
        )

        # 실패 응답
        response = TranslationResponse(
            request_id="abc-123",
            status=TranslationStatus.FAILED,
            error=TranslationError(
                code="PDF_PARSE_ERROR",
                message="PDF 파싱 실패",
                node="parse_pdf"
            )
        )
    """
    request_id: str
    status: TranslationStatus
    output_path: Optional[str] = None
    output_hash: Optional[str] = None
    title: Optional[str] = None
    title_ko: Optional[str] = None
    arxiv_id: Optional[str] = None
    stats: Optional[dict] = None
    progress: Optional[TranslationProgress] = None
    error: Optional[TranslationError] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def success(self) -> bool:
        """성공 여부"""
        return self.status == TranslationStatus.COMPLETED

    @property
    def is_running(self) -> bool:
        """실행 중 여부"""
        return self.status in (TranslationStatus.PENDING, TranslationStatus.RUNNING)

    def to_dict(self) -> dict:
        """딕셔너리 변환"""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "success": self.success,
            "output_path": self.output_path,
            "output_hash": self.output_hash,
            "title": self.title,
            "title_ko": self.title_ko,
            "arxiv_id": self.arxiv_id,
            "stats": self.stats,
            "progress": self.progress.to_dict() if self.progress else None,
            "error": self.error.to_dict() if self.error else None,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# === API 클래스 ===

class PaperTranslatorAPI:
    """
    Paper Translator API

    InsightBot이나 외부 시스템에서 Paper Translator를 호출하기 위한 인터페이스

    Features:
        - 동기/비동기 번역 실행
        - 진행 상황 추적
        - 작업 취소
        - 콜백 지원

    Examples:
        # 동기 실행
        api = PaperTranslatorAPI()
        response = api.translate(TranslationRequest(
            source="1706.03762",
            domain="NLP"
        ))

        # 비동기 실행
        import asyncio
        response = asyncio.run(api.translate_async(TranslationRequest(
            source="1706.03762",
            domain="NLP"
        )))

        # 진행 상황 콜백과 함께
        def on_progress(progress: TranslationProgress):
            print(f"Progress: {progress.progress_percent}%")

        response = api.translate(request, progress_callback=on_progress)
    """

    # 노드 순서 (진행률 계산용)
    NODE_ORDER = [
        "fetch_pdf",
        "parse_pdf",
        "chunk_text",
        "pre_process",
        "translate_chunks",
        "post_process",
        "generate_markdown",
        "save_output",
    ]

    def __init__(self, max_workers: int = 4):
        """
        Args:
            max_workers: 비동기 작업용 스레드 풀 크기
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running_tasks: Dict[str, dict] = {}
        self._lock = Lock()

    def translate(
        self,
        request: TranslationRequest,
        progress_callback: Optional[Callable[[TranslationProgress], None]] = None,
    ) -> TranslationResponse:
        """
        동기 방식 번역 실행

        Args:
            request: 번역 요청
            progress_callback: 진행 상황 콜백

        Returns:
            TranslationResponse
        """
        # 요청 검증
        valid, error_msg = request.validate()
        if not valid:
            return TranslationResponse(
                request_id=request.request_id,
                status=TranslationStatus.FAILED,
                error=TranslationError(
                    code="VALIDATION_ERROR",
                    message=error_msg,
                ),
                created_at=request.created_at,
            )

        # 실행 중 상태 등록
        with self._lock:
            self._running_tasks[request.request_id] = {
                "status": TranslationStatus.RUNNING,
                "cancelled": False,
            }

        try:
            # 진행 상황 추적
            progress = TranslationProgress()
            completed_nodes = []

            def internal_progress_callback(node_name: str, status: str):
                nonlocal progress, completed_nodes

                if node_name and node_name not in completed_nodes:
                    completed_nodes.append(node_name)

                progress.current_node = node_name
                progress.completed_nodes = completed_nodes.copy()
                progress.progress_percent = (len(completed_nodes) / len(self.NODE_ORDER)) * 100
                progress.message = f"{node_name} 실행 중..." if status == "running" else f"{node_name} 완료"

                if progress_callback:
                    progress_callback(progress)

                # 취소 확인
                with self._lock:
                    task_info = self._running_tasks.get(request.request_id, {})
                    if task_info.get("cancelled"):
                        raise InterruptedError("Translation cancelled")

            # 번역 실행
            from src.graph import run_translation

            final_state = run_translation(
                source=request.source,
                domain=request.domain,
                exclude_references=request.exclude_references,
                extract_tables=request.extract_tables,
                progress_callback=internal_progress_callback,
            )

            # 결과 생성
            if final_state.get("status") == "failed":
                return TranslationResponse(
                    request_id=request.request_id,
                    status=TranslationStatus.FAILED,
                    error=TranslationError(
                        code="TRANSLATION_ERROR",
                        message=final_state.get("error", "Unknown error"),
                        node=final_state.get("current_node"),
                    ),
                    progress=progress,
                    created_at=request.created_at,
                )

            # 성공
            output = final_state.get("output", {})
            metadata = final_state.get("metadata", {})
            stats = final_state.get("stats", {})

            return TranslationResponse(
                request_id=request.request_id,
                status=TranslationStatus.COMPLETED,
                output_path=output.get("file_path"),
                output_hash=output.get("md5_hash"),
                title=metadata.get("title"),
                title_ko=metadata.get("title_ko"),
                arxiv_id=metadata.get("arxiv_id"),
                stats=stats,
                progress=progress,
                created_at=request.created_at,
                completed_at=datetime.now().isoformat(),
            )

        except InterruptedError:
            return TranslationResponse(
                request_id=request.request_id,
                status=TranslationStatus.CANCELLED,
                error=TranslationError(
                    code="CANCELLED",
                    message="번역이 취소되었습니다",
                ),
                created_at=request.created_at,
            )

        except Exception as e:
            logger.exception(f"번역 실패: {e}")
            return TranslationResponse(
                request_id=request.request_id,
                status=TranslationStatus.FAILED,
                error=TranslationError(
                    code="INTERNAL_ERROR",
                    message=str(e),
                ),
                created_at=request.created_at,
            )

        finally:
            # 실행 상태 정리
            with self._lock:
                self._running_tasks.pop(request.request_id, None)

    async def translate_async(
        self,
        request: TranslationRequest,
        progress_callback: Optional[Callable[[TranslationProgress], None]] = None,
    ) -> TranslationResponse:
        """
        비동기 방식 번역 실행

        Args:
            request: 번역 요청
            progress_callback: 진행 상황 콜백

        Returns:
            TranslationResponse
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.translate(request, progress_callback)
        )

    def get_status(self, request_id: str) -> Optional[TranslationStatus]:
        """
        작업 상태 조회

        Args:
            request_id: 요청 ID

        Returns:
            현재 상태 (없으면 None)
        """
        with self._lock:
            task_info = self._running_tasks.get(request_id)
            if task_info:
                return task_info.get("status")
            return None

    def cancel(self, request_id: str) -> bool:
        """
        작업 취소

        Args:
            request_id: 요청 ID

        Returns:
            취소 성공 여부
        """
        with self._lock:
            if request_id in self._running_tasks:
                self._running_tasks[request_id]["cancelled"] = True
                self._running_tasks[request_id]["status"] = TranslationStatus.CANCELLED
                logger.info(f"번역 취소 요청: {request_id}")
                return True
            return False

    def is_running(self, request_id: str) -> bool:
        """작업 실행 중 여부 확인"""
        with self._lock:
            return request_id in self._running_tasks

    def shutdown(self):
        """리소스 정리"""
        self._executor.shutdown(wait=False)


# === 편의 함수 ===

# 싱글톤 API 인스턴스
_api_instance: Optional[PaperTranslatorAPI] = None
_api_lock = Lock()


def _get_api() -> PaperTranslatorAPI:
    """싱글톤 API 인스턴스 반환"""
    global _api_instance
    with _api_lock:
        if _api_instance is None:
            _api_instance = PaperTranslatorAPI()
        return _api_instance


def translate(
    source: str,
    domain: str = "General",
    exclude_references: bool = True,
    extract_tables: bool = True,
    progress_callback: Optional[Callable[[TranslationProgress], None]] = None,
    **kwargs
) -> TranslationResponse:
    """
    논문 번역 (편의 함수)

    InsightBot에서 가장 간단하게 호출할 수 있는 인터페이스

    Args:
        source: PDF URL, ArXiv ID, 또는 로컬 파일 경로
        domain: 용어 도메인 (NLP, CV, RL, General)
        exclude_references: References 섹션 제외 여부
        extract_tables: 표 추출 여부
        progress_callback: 진행 상황 콜백
        **kwargs: 추가 메타데이터

    Returns:
        TranslationResponse

    Examples:
        # ArXiv 논문 번역
        result = translate("1706.03762", domain="NLP")

        # URL로 번역
        result = translate("https://arxiv.org/pdf/1706.03762.pdf")

        # 진행 상황 확인
        def on_progress(p):
            print(f"{p.progress_percent:.0f}% - {p.message}")

        result = translate("1706.03762", progress_callback=on_progress)

        # 결과 확인
        if result.success:
            print(f"번역 완료: {result.output_path}")
        else:
            print(f"실패: {result.error.message}")
    """
    request = TranslationRequest(
        source=source,
        domain=domain,
        exclude_references=exclude_references,
        extract_tables=extract_tables,
        metadata=kwargs,
    )

    return _get_api().translate(request, progress_callback)


async def translate_async(
    source: str,
    domain: str = "General",
    exclude_references: bool = True,
    extract_tables: bool = True,
    progress_callback: Optional[Callable[[TranslationProgress], None]] = None,
    **kwargs
) -> TranslationResponse:
    """
    논문 번역 - 비동기 버전

    Args:
        source: PDF URL, ArXiv ID, 또는 로컬 파일 경로
        domain: 용어 도메인
        exclude_references: References 섹션 제외 여부
        extract_tables: 표 추출 여부
        progress_callback: 진행 상황 콜백
        **kwargs: 추가 메타데이터

    Returns:
        TranslationResponse

    Examples:
        import asyncio

        async def main():
            result = await translate_async("1706.03762", domain="NLP")
            print(f"완료: {result.output_path}")

        asyncio.run(main())
    """
    request = TranslationRequest(
        source=source,
        domain=domain,
        exclude_references=exclude_references,
        extract_tables=extract_tables,
        metadata=kwargs,
    )

    return await _get_api().translate_async(request, progress_callback)


def get_translation_status(request_id: str) -> Optional[TranslationStatus]:
    """
    번역 작업 상태 조회

    Args:
        request_id: 요청 ID

    Returns:
        현재 상태 (실행 중이 아니면 None)
    """
    return _get_api().get_status(request_id)


def cancel_translation(request_id: str) -> bool:
    """
    번역 작업 취소

    Args:
        request_id: 요청 ID

    Returns:
        취소 성공 여부
    """
    return _get_api().cancel(request_id)
