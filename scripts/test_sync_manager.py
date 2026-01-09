#!/usr/bin/env python3
"""
Sync Manager 테스트 스크립트
"""
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feedback.sync_manager import (
    SyncManager,
    SyncResult,
    SyncItem,
    SyncAction,
    sync_translation_file,
    preview_sync,
    get_changed_files,
)
from src.feedback.diff_analyzer import (
    DiffAnalyzer,
    DiffResult,
    TextChange,
    TermChange,
    ChangeType,
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


def test_sync_action_enum():
    """SyncAction 열거형 테스트"""
    print_section("SyncAction 열거형 테스트")

    # 액션 유형 테스트
    tests = [
        ("UPDATE_TERM 액션", SyncAction.UPDATE_TERM.value == "update_term"),
        ("ADD_TERM 액션", SyncAction.ADD_TERM.value == "add_term"),
        ("LOG_CHANGE 액션", SyncAction.LOG_CHANGE.value == "log_change"),
        ("UPDATE_HASH 액션", SyncAction.UPDATE_HASH.value == "update_hash"),
        ("SKIP 액션", SyncAction.SKIP.value == "skip"),
    ]

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_sync_item_dataclass():
    """SyncItem 데이터클래스 테스트"""
    print_section("SyncItem 데이터클래스 테스트")

    tests = []

    # 기본 생성
    item = SyncItem(
        action=SyncAction.UPDATE_TERM,
        description="용어 업데이트: test",
        data={"term_id": "123", "new_target": "테스트"},
    )
    tests.append(("SyncItem 생성", item is not None))
    tests.append(("액션 확인", item.action == SyncAction.UPDATE_TERM))
    tests.append(("설명 확인", "용어 업데이트" in item.description))
    tests.append(("기본 applied=False", item.applied == False))
    tests.append(("기본 error=None", item.error is None))

    # 데이터 접근
    tests.append(("data 접근", item.data.get("term_id") == "123"))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_sync_result_dataclass():
    """SyncResult 데이터클래스 테스트"""
    print_section("SyncResult 데이터클래스 테스트")

    tests = []

    # 성공 결과
    success_result = SyncResult(
        file_path="test.md",
        success=True,
        items=[
            SyncItem(SyncAction.UPDATE_TERM, "용어 업데이트", applied=True),
            SyncItem(SyncAction.ADD_TERM, "새 용어 추가", applied=True),
        ],
        terms_updated=1,
        terms_added=1,
        changes_logged=2,
        hash_updated=True,
    )
    tests.append(("성공 결과 생성", success_result.success == True))
    tests.append(("파일 경로", success_result.file_path == "test.md"))
    tests.append(("항목 개수", len(success_result.items) == 2))
    tests.append(("용어 업데이트 수", success_result.terms_updated == 1))
    tests.append(("용어 추가 수", success_result.terms_added == 1))
    tests.append(("해시 업데이트", success_result.hash_updated == True))
    tests.append(("synced_at 존재", success_result.synced_at is not None))

    # 실패 결과
    fail_result = SyncResult(
        file_path="fail.md",
        success=False,
        error="테스트 에러",
    )
    tests.append(("실패 결과 생성", fail_result.success == False))
    tests.append(("에러 메시지", fail_result.error == "테스트 에러"))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_sync_manager_init():
    """SyncManager 초기화 테스트"""
    print_section("SyncManager 초기화 테스트")

    tests = []

    # 기본 설정
    manager1 = SyncManager()
    tests.append(("기본 생성", manager1 is not None))
    tests.append(("기본 auto_sync=False", manager1.auto_sync == False))
    tests.append(("기본 min_confidence=0.7", manager1.min_confidence == 0.7))
    tests.append(("기본 use_llm_analysis=True", manager1.use_llm_analysis == True))

    # 커스텀 설정
    manager2 = SyncManager(
        auto_sync=True,
        min_confidence=0.5,
        use_llm_analysis=False,
    )
    tests.append(("커스텀 auto_sync=True", manager2.auto_sync == True))
    tests.append(("커스텀 min_confidence=0.5", manager2.min_confidence == 0.5))
    tests.append(("커스텀 use_llm_analysis=False", manager2.use_llm_analysis == False))

    # 콜백 설정
    def dummy_callback(items):
        return True

    manager3 = SyncManager(confirm_callback=dummy_callback)
    tests.append(("콜백 설정", manager3.confirm_callback is not None))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_prepare_sync_items():
    """prepare_sync_items 테스트"""
    print_section("prepare_sync_items 테스트")

    tests = []
    manager = SyncManager(min_confidence=0.7, use_llm_analysis=False)

    # 변경 없음
    empty_diff = DiffResult(
        file_path="test.md",
        original_hash="abc",
        current_hash="abc",
        has_changes=False,
        text_changes=[],
        term_changes=[],
    )
    items1 = manager.prepare_sync_items(empty_diff)
    tests.append(("변경 없음 -> 빈 항목", len(items1) == 0))

    # 용어 변경 있음 (높은 확신도)
    diff_with_changes = DiffResult(
        file_path="test.md",
        original_hash="abc",
        current_hash="xyz",
        has_changes=True,
        text_changes=[],
        term_changes=[
            TermChange(
                source_text="attention",
                old_target="어텐션",
                new_target="주의 메커니즘",
                confidence=0.9,
            )
        ],
    )
    items2 = manager.prepare_sync_items(diff_with_changes, translation_id="trans123")
    tests.append(("변경 있음 -> 항목 생성", len(items2) > 0))

    # UPDATE_TERM 또는 ADD_TERM 액션 포함 확인
    has_term_action = any(
        i.action in [SyncAction.UPDATE_TERM, SyncAction.ADD_TERM]
        for i in items2
    )
    tests.append(("용어 액션 포함", has_term_action))

    # LOG_CHANGE 액션 포함 확인
    has_log_action = any(i.action == SyncAction.LOG_CHANGE for i in items2)
    tests.append(("로그 액션 포함", has_log_action))

    # UPDATE_HASH 액션 포함 확인 (translation_id 있을 때)
    has_hash_action = any(i.action == SyncAction.UPDATE_HASH for i in items2)
    tests.append(("해시 액션 포함", has_hash_action))

    # 낮은 확신도 -> SKIP
    diff_low_conf = DiffResult(
        file_path="test.md",
        original_hash="abc",
        current_hash="xyz",
        has_changes=True,
        text_changes=[],
        term_changes=[
            TermChange(
                source_text="test",
                old_target="테스트",
                new_target="시험",
                confidence=0.3,  # 낮은 확신도
            )
        ],
    )
    items3 = manager.prepare_sync_items(diff_low_conf)
    has_skip = any(i.action == SyncAction.SKIP for i in items3)
    tests.append(("낮은 확신도 -> SKIP", has_skip))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_format_sync_preview():
    """format_sync_preview 테스트"""
    print_section("format_sync_preview 테스트")

    tests = []
    manager = SyncManager()

    items = [
        SyncItem(SyncAction.UPDATE_TERM, "용어 업데이트: attention → 주의", {"confidence": 0.9}),
        SyncItem(SyncAction.ADD_TERM, "새 용어: transformer → 트랜스포머", {"confidence": 0.85}),
        SyncItem(SyncAction.LOG_CHANGE, "변경 로그: attention", {}),
        SyncItem(SyncAction.SKIP, "건너뜀: test (낮은 확신도)", {"confidence": 0.3}),
    ]

    preview = manager.format_sync_preview(items)

    tests.append(("미리보기 문자열 생성", len(preview) > 0))
    tests.append(("구분선 포함", "=" in preview))
    tests.append(("용어 업데이트 포함", "용어 업데이트" in preview))
    tests.append(("새 용어 추가 포함", "새 용어" in preview or "add_term" in preview.lower()))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_get_sync_summary():
    """get_sync_summary 테스트"""
    print_section("get_sync_summary 테스트")

    tests = []
    manager = SyncManager()

    results = [
        SyncResult(
            file_path="file1.md",
            success=True,
            terms_updated=2,
            terms_added=1,
            changes_logged=3,
        ),
        SyncResult(
            file_path="file2.md",
            success=True,
            terms_updated=1,
            terms_added=0,
            changes_logged=1,
        ),
        SyncResult(
            file_path="file3.md",
            success=False,
            error="에러 발생",
        ),
    ]

    summary = manager.get_sync_summary(results)

    tests.append(("총 파일 수", summary["total_files"] == 3))
    tests.append(("성공 수", summary["success"] == 2))
    tests.append(("실패 수", summary["failed"] == 1))
    tests.append(("총 업데이트", summary["terms_updated"] == 3))
    tests.append(("총 추가", summary["terms_added"] == 1))
    tests.append(("총 로그", summary["changes_logged"] == 4))
    tests.append(("results 리스트", len(summary["results"]) == 3))

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def test_sync_file_dry_run():
    """sync_file dry_run 테스트"""
    print_section("sync_file dry_run 테스트")

    tests = []
    manager = SyncManager(use_llm_analysis=False)

    # 임시 파일 생성
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test\n\nThis is a test with attention mechanism.\n")
        temp_path = f.name

    try:
        # 원본 콘텐츠와 함께 dry run
        original = "# Test\n\nThis is original content.\n"
        result = manager.sync_file(
            temp_path,
            original_content=original,
            dry_run=True
        )

        tests.append(("결과 반환", result is not None))
        tests.append(("파일 경로 포함", result.file_path == temp_path))
        tests.append(("Dry run 성공", result.success == True))
        tests.append(("실제 변경 없음 (terms_updated=0)", result.terms_updated == 0))
        tests.append(("실제 변경 없음 (terms_added=0)", result.terms_added == 0))

    finally:
        import os
        os.unlink(temp_path)

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

    # get_changed_files 테스트 (존재하지 않는 디렉토리)
    changed = get_changed_files("./non_existent_dir")
    tests.append(("get_changed_files 빈 디렉토리", changed == []))

    # preview_sync 테스트 (임시 파일)
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test\n\nOriginal content.\n")
        temp_path = f.name

    try:
        # 동일한 내용으로 미리보기 (변경 없음)
        preview = preview_sync(temp_path)
        tests.append(("preview_sync 실행", preview is not None))
        tests.append(("변경 없음 메시지", "변경" in preview or len(preview) > 0))

    finally:
        import os
        os.unlink(temp_path)

    passed = 0
    for name, result in tests:
        print_test(name, result)
        if result:
            passed += 1

    return passed, len(tests)


def main():
    """메인 테스트 실행"""
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(" Sync Manager 테스트 실행")
    print(f"{'=' * 60}{Colors.RESET}")
    print(f" 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_passed = 0
    total_tests = 0

    # 테스트 실행
    test_functions = [
        test_sync_action_enum,
        test_sync_item_dataclass,
        test_sync_result_dataclass,
        test_sync_manager_init,
        test_prepare_sync_items,
        test_format_sync_preview,
        test_get_sync_summary,
        test_sync_file_dry_run,
        test_convenience_functions,
    ]

    for test_func in test_functions:
        try:
            passed, total = test_func()
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"\n{Colors.RED}테스트 함수 실행 실패: {test_func.__name__}")
            print(f"에러: {e}{Colors.RESET}")

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
