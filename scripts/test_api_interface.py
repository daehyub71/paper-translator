#!/usr/bin/env python3
"""
InsightBot API 인터페이스 테스트 스크립트
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api import (
    TranslationRequest,
    TranslationResponse,
    TranslationStatus,
    TranslationProgress,
    TranslationError,
    PaperTranslatorAPI,
    translate,
    translate_async,
    get_translation_status,
    cancel_translation,
)


# 테스트 색상 출력
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_test(name: str, passed: bool, message: str = ""):
    """테스트 결과 출력"""
    status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
    print(f"  [{status}] {name}")
    if message and not passed:
        print(f"        {Colors.RED}{message}{Colors.RESET}")


def print_section(title: str):
    """섹션 헤더 출력"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}{Colors.RESET}\n")


def test_translation_status_enum():
    """TranslationStatus 열거형 테스트"""
    print_section("TranslationStatus 열거형 테스트")

    tests = [
        ("PENDING 상태", TranslationStatus.PENDING.value == "pending"),
        ("RUNNING 상태", TranslationStatus.RUNNING.value == "running"),
        ("COMPLETED 상태", TranslationStatus.COMPLETED.value == "completed"),
        ("FAILED 상태", TranslationStatus.FAILED.value == "failed"),
        ("CANCELLED 상태", TranslationStatus.CANCELLED.value == "cancelled"),
    ]

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_translation_request():
    """TranslationRequest 테스트"""
    print_section("TranslationRequest 스키마 테스트")

    tests = []

    # 기본 생성
    req1 = TranslationRequest(source="1706.03762")
    tests.append(("기본 생성", req1.source == "1706.03762"))
    tests.append(("기본 domain=General", req1.domain == "General"))
    tests.append(("기본 exclude_references=True", req1.exclude_references == True))
    tests.append(("기본 extract_tables=True", req1.extract_tables == True))
    tests.append(("request_id 자동 생성", req1.request_id is not None and len(req1.request_id) > 0))
    tests.append(("created_at 자동 생성", req1.created_at is not None))

    # 소스 타입 판별
    req_url = TranslationRequest(source="https://arxiv.org/pdf/1706.03762.pdf")
    req_arxiv = TranslationRequest(source="1706.03762")
    req_file = TranslationRequest(source="/path/to/paper.pdf")

    from src.api.interface import SourceType
    tests.append(("URL 타입 판별", req_url.get_source_type() == SourceType.URL))
    tests.append(("ArXiv ID 타입 판별", req_arxiv.get_source_type() == SourceType.ARXIV_ID))
    tests.append(("로컬 파일 타입 판별", req_file.get_source_type() == SourceType.LOCAL_FILE))

    # 유효성 검증
    valid_req = TranslationRequest(source="1706.03762", domain="NLP")
    is_valid, error = valid_req.validate()
    tests.append(("유효한 요청 검증", is_valid == True and error is None))

    invalid_req = TranslationRequest(source="", domain="NLP")
    is_valid2, error2 = invalid_req.validate()
    tests.append(("빈 source 검증 실패", is_valid2 == False and error2 is not None))

    invalid_domain_req = TranslationRequest(source="1706.03762", domain="INVALID")
    is_valid3, error3 = invalid_domain_req.validate()
    tests.append(("잘못된 domain 검증 실패", is_valid3 == False))

    # to_dict
    dict_result = req1.to_dict()
    tests.append(("to_dict 변환", "request_id" in dict_result and "source" in dict_result))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_translation_progress():
    """TranslationProgress 테스트"""
    print_section("TranslationProgress 스키마 테스트")

    tests = []

    progress = TranslationProgress(
        current_node="translate_chunks",
        completed_nodes=["fetch_pdf", "parse_pdf", "chunk_text", "pre_process"],
        total_nodes=8,
        progress_percent=50.0,
        message="번역 중..."
    )

    tests.append(("current_node 설정", progress.current_node == "translate_chunks"))
    tests.append(("completed_nodes 설정", len(progress.completed_nodes) == 4))
    tests.append(("progress_percent 설정", progress.progress_percent == 50.0))
    tests.append(("message 설정", progress.message == "번역 중..."))

    dict_result = progress.to_dict()
    tests.append(("to_dict 변환", "current_node" in dict_result))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_translation_error():
    """TranslationError 테스트"""
    print_section("TranslationError 스키마 테스트")

    tests = []

    error = TranslationError(
        code="PDF_PARSE_ERROR",
        message="PDF 파싱 실패",
        node="parse_pdf",
        details={"page": 5}
    )

    tests.append(("code 설정", error.code == "PDF_PARSE_ERROR"))
    tests.append(("message 설정", error.message == "PDF 파싱 실패"))
    tests.append(("node 설정", error.node == "parse_pdf"))
    tests.append(("details 설정", error.details == {"page": 5}))

    dict_result = error.to_dict()
    tests.append(("to_dict 변환", "code" in dict_result and "message" in dict_result))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_translation_response():
    """TranslationResponse 테스트"""
    print_section("TranslationResponse 스키마 테스트")

    tests = []

    # 성공 응답
    success_response = TranslationResponse(
        request_id="test-123",
        status=TranslationStatus.COMPLETED,
        output_path="/output/translated.md",
        output_hash="abc123",
        title="Attention Is All You Need",
        stats={"total_tokens": 10000},
    )

    tests.append(("성공 응답 생성", success_response.status == TranslationStatus.COMPLETED))
    tests.append(("success 프로퍼티 (성공)", success_response.success == True))
    tests.append(("is_running 프로퍼티 (완료)", success_response.is_running == False))
    tests.append(("output_path 설정", success_response.output_path == "/output/translated.md"))

    # 실패 응답
    fail_response = TranslationResponse(
        request_id="test-456",
        status=TranslationStatus.FAILED,
        error=TranslationError(code="ERROR", message="테스트 에러"),
    )

    tests.append(("실패 응답 생성", fail_response.status == TranslationStatus.FAILED))
    tests.append(("success 프로퍼티 (실패)", fail_response.success == False))
    tests.append(("error 설정", fail_response.error is not None))

    # 실행 중 응답
    running_response = TranslationResponse(
        request_id="test-789",
        status=TranslationStatus.RUNNING,
    )
    tests.append(("is_running 프로퍼티 (실행 중)", running_response.is_running == True))

    # to_dict
    dict_result = success_response.to_dict()
    tests.append(("to_dict 변환", "request_id" in dict_result and "status" in dict_result))
    tests.append(("to_dict success 포함", "success" in dict_result))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_api_class():
    """PaperTranslatorAPI 클래스 테스트"""
    print_section("PaperTranslatorAPI 클래스 테스트")

    tests = []

    # API 인스턴스 생성
    api = PaperTranslatorAPI(max_workers=2)
    tests.append(("API 인스턴스 생성", api is not None))
    tests.append(("NODE_ORDER 정의", len(api.NODE_ORDER) == 8))

    # 유효성 검증 실패 케이스
    invalid_request = TranslationRequest(source="", domain="NLP")
    response = api.translate(invalid_request)
    tests.append(("빈 source 검증 실패", response.status == TranslationStatus.FAILED))
    tests.append(("VALIDATION_ERROR 코드", response.error.code == "VALIDATION_ERROR"))

    # 잘못된 도메인 검증 실패
    invalid_domain = TranslationRequest(source="1706.03762", domain="INVALID")
    response2 = api.translate(invalid_domain)
    tests.append(("잘못된 domain 검증 실패", response2.status == TranslationStatus.FAILED))

    # 상태 조회 (존재하지 않는 ID)
    status = api.get_status("non-existent-id")
    tests.append(("존재하지 않는 ID 상태 조회", status is None))

    # 취소 (존재하지 않는 ID)
    cancelled = api.cancel("non-existent-id")
    tests.append(("존재하지 않는 ID 취소 실패", cancelled == False))

    # is_running (존재하지 않는 ID)
    running = api.is_running("non-existent-id")
    tests.append(("존재하지 않는 ID is_running", running == False))

    # 정리
    api.shutdown()
    tests.append(("shutdown 호출", True))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_convenience_functions():
    """편의 함수 테스트"""
    print_section("편의 함수 테스트")

    tests = []

    # translate 함수 (유효성 실패 케이스)
    response = translate(source="", domain="NLP")
    tests.append(("translate 빈 source", response.status == TranslationStatus.FAILED))

    # get_translation_status
    status = get_translation_status("non-existent-id")
    tests.append(("get_translation_status 존재하지 않는 ID", status is None))

    # cancel_translation
    cancelled = cancel_translation("non-existent-id")
    tests.append(("cancel_translation 존재하지 않는 ID", cancelled == False))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_async_interface():
    """비동기 인터페이스 테스트"""
    print_section("비동기 인터페이스 테스트")

    tests = []

    async def run_async_tests():
        nonlocal tests

        # translate_async 함수 (유효성 실패 케이스)
        response = await translate_async(source="", domain="NLP")
        tests.append(("translate_async 빈 source", response.status == TranslationStatus.FAILED))

        # API 인스턴스로 비동기 호출
        api = PaperTranslatorAPI()
        request = TranslationRequest(source="", domain="General")
        response2 = await api.translate_async(request)
        tests.append(("API translate_async 빈 source", response2.status == TranslationStatus.FAILED))

        api.shutdown()

    asyncio.run(run_async_tests())

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_progress_callback():
    """진행 상황 콜백 테스트"""
    print_section("진행 상황 콜백 테스트")

    tests = []

    progress_updates = []

    def progress_callback(progress: TranslationProgress):
        progress_updates.append(progress.to_dict())

    # 유효성 실패는 콜백 호출 없이 바로 반환
    response = translate(source="", progress_callback=progress_callback)
    tests.append(("유효성 실패 시 콜백 미호출", len(progress_updates) == 0))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def main():
    """메인 테스트 실행"""
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(" InsightBot API 인터페이스 테스트 실행")
    print(f"{'=' * 60}{Colors.RESET}")
    print(f" 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_passed = 0
    total_tests = 0

    # 테스트 실행
    test_functions = [
        test_translation_status_enum,
        test_translation_request,
        test_translation_progress,
        test_translation_error,
        test_translation_response,
        test_api_class,
        test_convenience_functions,
        test_async_interface,
        test_progress_callback,
    ]

    for test_func in test_functions:
        try:
            passed, total = test_func()
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"\n{Colors.RED}테스트 함수 실행 실패: {test_func.__name__}")
            print(f"에러: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()

    # 최종 결과
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(" 테스트 결과")
    print(f"{'=' * 60}{Colors.RESET}")

    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    if total_passed == total_tests:
        color = Colors.GREEN
        status = "ALL PASSED"
    elif total_passed > total_tests * 0.7:
        color = Colors.YELLOW
        status = "MOSTLY PASSED"
    else:
        color = Colors.RED
        status = "FAILED"

    print(f"\n {color}{status}: {total_passed}/{total_tests} ({success_rate:.1f}%){Colors.RESET}")
    print()

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    exit(main())
