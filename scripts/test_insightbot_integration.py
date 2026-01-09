#!/usr/bin/env python3
"""
InsightBot 그래프 연동 테스트 스크립트
"""
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.insightbot import (
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
    # Conditional Edge Functions
    should_translate,
    check_translation_result,
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


def test_state_types():
    """상태 타입 테스트"""
    print_section("상태 타입 테스트")

    tests = []

    # PaperSource
    source: PaperSource = {
        "arxiv_id": "1706.03762",
        "domain": "NLP"
    }
    tests.append(("PaperSource 생성", source.get("arxiv_id") == "1706.03762"))
    tests.append(("PaperSource domain", source.get("domain") == "NLP"))

    # TranslationResult
    result: TranslationResult = {
        "request_id": "test-123",
        "status": "completed",
        "success": True,
        "output_path": "/output/test.md"
    }
    tests.append(("TranslationResult 생성", result.get("success") == True))
    tests.append(("TranslationResult status", result.get("status") == "completed"))

    # InsightBotState
    state: InsightBotState = {
        "messages": [{"role": "user", "content": "번역해줘"}],
        "paper_source": source,
        "should_translate": True,
        "user_confirmed": True,
        "translation_in_progress": False,
    }
    tests.append(("InsightBotState 생성", state.get("should_translate") == True))
    tests.append(("InsightBotState messages", len(state.get("messages", [])) == 1))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_parse_paper_source():
    """논문 소스 파싱 테스트"""
    print_section("parse_paper_source 노드 테스트")

    tests = []

    # ArXiv URL 파싱
    state1: InsightBotState = {
        "messages": [{"role": "user", "content": "https://arxiv.org/pdf/1706.03762.pdf 번역해줘"}],
    }
    result1 = parse_paper_source(state1)
    tests.append(("ArXiv URL 파싱", result1.get("paper_source", {}).get("url") is not None))
    tests.append(("ArXiv ID 추출", result1.get("paper_source", {}).get("arxiv_id") == "1706.03762"))
    tests.append(("should_translate=True", result1.get("should_translate") == True))

    # ArXiv ID 직접 입력
    state2: InsightBotState = {
        "messages": [{"role": "user", "content": "1706.03762 논문 번역"}],
    }
    result2 = parse_paper_source(state2)
    tests.append(("ArXiv ID 직접 파싱", result2.get("paper_source", {}).get("arxiv_id") == "1706.03762"))

    # 도메인 감지 (NLP)
    state3: InsightBotState = {
        "messages": [{"role": "user", "content": "1706.03762 NLP 논문 번역해줘"}],
    }
    result3 = parse_paper_source(state3)
    tests.append(("NLP 도메인 감지", result3.get("paper_source", {}).get("domain") == "NLP"))

    # 도메인 감지 (CV)
    state4: InsightBotState = {
        "messages": [{"role": "user", "content": "vision 관련 논문 번역"}],
    }
    result4 = parse_paper_source(state4)
    tests.append(("CV 도메인 감지", result4.get("paper_source", {}).get("domain") == "CV"))

    # 소스 없음
    state5: InsightBotState = {
        "messages": [{"role": "user", "content": "안녕하세요"}],
    }
    result5 = parse_paper_source(state5)
    tests.append(("소스 없음 처리", result5.get("should_translate") == False))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_validate_translation_request():
    """번역 요청 검증 테스트"""
    print_section("validate_translation_request 노드 테스트")

    tests = []

    # 유효한 요청
    state1: InsightBotState = {
        "paper_source": {
            "arxiv_id": "1706.03762",
            "domain": "NLP"
        },
    }
    result1 = validate_translation_request(state1)
    tests.append(("유효한 요청 검증", result1.get("translation_request") is not None))
    tests.append(("요청 source 설정", result1.get("translation_request", {}).get("source") == "1706.03762"))
    tests.append(("요청 domain 설정", result1.get("translation_request", {}).get("domain") == "NLP"))
    tests.append(("should_translate=True", result1.get("should_translate") == True))

    # URL 소스
    state2: InsightBotState = {
        "paper_source": {
            "url": "https://arxiv.org/pdf/1706.03762.pdf",
            "domain": "General"
        },
    }
    result2 = validate_translation_request(state2)
    tests.append(("URL 소스 검증", result2.get("translation_request", {}).get("source") is not None))

    # 소스 없음
    state3: InsightBotState = {
        "paper_source": {
            "domain": "General"
        },
    }
    result3 = validate_translation_request(state3)
    tests.append(("소스 없음 -> should_translate=False", result3.get("should_translate") == False))

    # paper_source 자체가 없음
    state4: InsightBotState = {}
    result4 = validate_translation_request(state4)
    tests.append(("paper_source 없음 처리", result4.get("should_translate") == False))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_format_translation_response():
    """번역 결과 포맷팅 테스트"""
    print_section("format_translation_response 노드 테스트")

    tests = []

    # 성공 결과 포맷팅
    state1: InsightBotState = {
        "messages": [],
        "translation_result": {
            "success": True,
            "title": "Attention Is All You Need",
            "title_ko": "어텐션이 전부입니다",
            "output_path": "/output/attention.md",
            "stats": {
                "total_chunks": 10,
                "total_tokens": 5000,
                "estimated_cost_usd": "$0.05",
                "term_match_rate": "95%"
            }
        }
    }
    result1 = format_translation_response(state1)
    messages1 = result1.get("messages", [])
    tests.append(("성공 메시지 추가", len(messages1) == 1))
    tests.append(("assistant role", messages1[0].get("role") == "assistant"))
    tests.append(("제목 포함", "Attention" in messages1[0].get("content", "")))

    # 실패 결과 포맷팅
    state2: InsightBotState = {
        "messages": [],
        "translation_result": {
            "success": False,
            "error": {
                "code": "PDF_ERROR",
                "message": "PDF 파싱 실패"
            }
        }
    }
    result2 = format_translation_response(state2)
    messages2 = result2.get("messages", [])
    tests.append(("실패 메시지 추가", len(messages2) == 1))
    tests.append(("에러 코드 포함", "PDF_ERROR" in messages2[0].get("content", "")))

    # 결과 없음
    state3: InsightBotState = {
        "messages": [],
    }
    result3 = format_translation_response(state3)
    messages3 = result3.get("messages", [])
    tests.append(("결과 없음 처리", len(messages3) == 1))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_conditional_edges():
    """조건부 엣지 함수 테스트"""
    print_section("조건부 엣지 함수 테스트")

    tests = []

    # should_translate
    state1: InsightBotState = {"should_translate": True, "user_confirmed": True}
    tests.append(("should_translate -> translate", should_translate(state1) == "translate"))

    state2: InsightBotState = {"should_translate": False}
    tests.append(("should_translate -> skip (False)", should_translate(state2) == "skip"))

    state3: InsightBotState = {"should_translate": True, "user_confirmed": False}
    tests.append(("should_translate -> skip (not confirmed)", should_translate(state3) == "skip"))

    # check_translation_result
    state4: InsightBotState = {"translation_result": {"success": True}}
    tests.append(("check_result -> success", check_translation_result(state4) == "success"))

    state5: InsightBotState = {"translation_result": {"success": False}}
    tests.append(("check_result -> failed", check_translation_result(state5) == "failed"))

    state6: InsightBotState = {}
    tests.append(("check_result -> failed (no result)", check_translation_result(state6) == "failed"))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_translation_node_wrapper():
    """TranslationNodeWrapper 테스트"""
    print_section("TranslationNodeWrapper 테스트")

    tests = []

    # 래퍼 생성
    wrapper = TranslationNodeWrapper(auto_confirm=True)
    tests.append(("래퍼 생성", wrapper is not None))
    tests.append(("auto_confirm 설정", wrapper.auto_confirm == True))

    # 번역 요청 상태 생성 헬퍼
    state = wrapper.create_translation_request("1706.03762", domain="NLP")
    tests.append(("요청 상태 생성", state is not None))
    tests.append(("messages 포함", len(state.get("messages", [])) == 1))
    tests.append(("paper_source 포함", state.get("paper_source") is not None))
    tests.append(("arxiv_id 설정", state.get("paper_source", {}).get("arxiv_id") == "1706.03762"))

    # 서브그래프 반환
    subgraph = wrapper.get_subgraph()
    tests.append(("서브그래프 반환", subgraph is not None))

    # 두 번 호출해도 동일한 인스턴스 (캐싱)
    subgraph2 = wrapper.get_subgraph()
    tests.append(("서브그래프 캐싱", subgraph is subgraph2))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_graph_builders():
    """그래프 빌더 테스트"""
    print_section("그래프 빌더 테스트")

    tests = []

    # 서브그래프 생성
    subgraph = create_translation_subgraph()
    tests.append(("서브그래프 생성", subgraph is not None))

    # 서브그래프 컴파일
    compiled = compile_translation_subgraph()
    tests.append(("서브그래프 컴파일", compiled is not None))

    # 편의 함수 - get_translation_node
    node = get_translation_node()
    tests.append(("get_translation_node 반환", callable(node)))

    # 편의 함수 - get_translation_subgraph
    subgraph2 = get_translation_subgraph()
    tests.append(("get_translation_subgraph 반환", subgraph2 is not None))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_end_to_end_flow():
    """End-to-End 플로우 테스트 (유효성 검증까지만)"""
    print_section("End-to-End 플로우 테스트")

    tests = []

    # 전체 파이프라인 (검증 실패 케이스)
    state: InsightBotState = {
        "messages": [{"role": "user", "content": "안녕하세요"}],  # 논문 소스 없음
    }

    # parse_paper_source
    state = parse_paper_source(state)
    tests.append(("파싱 후 should_translate=False", state.get("should_translate") == False))

    # validate_translation_request
    state = validate_translation_request(state)
    tests.append(("검증 후 should_translate=False", state.get("should_translate") == False))

    # format_translation_response (결과 없음)
    state = format_translation_response(state)
    tests.append(("포맷팅 후 messages 추가", len(state.get("messages", [])) == 2))

    # 유효한 소스로 테스트 (실행 없이 검증까지만)
    state2: InsightBotState = {
        "messages": [{"role": "user", "content": "1706.03762 논문 번역해줘 NLP"}],
    }

    state2 = parse_paper_source(state2)
    tests.append(("유효 소스 파싱", state2.get("paper_source", {}).get("arxiv_id") == "1706.03762"))

    state2 = validate_translation_request(state2)
    tests.append(("유효 소스 검증", state2.get("should_translate") == True))
    tests.append(("요청 생성 확인", state2.get("translation_request") is not None))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def main():
    """메인 테스트 실행"""
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(" InsightBot 그래프 연동 테스트 실행")
    print(f"{'=' * 60}{Colors.RESET}")
    print(f" 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_passed = 0
    total_tests = 0

    # 테스트 실행
    test_functions = [
        test_state_types,
        test_parse_paper_source,
        test_validate_translation_request,
        test_format_translation_response,
        test_conditional_edges,
        test_translation_node_wrapper,
        test_graph_builders,
        test_end_to_end_flow,
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
