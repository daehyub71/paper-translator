"""
Diff Analyzer 테스트 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feedback import (
    DiffAnalyzer,
    DiffResult,
    TextChange,
    TermChange,
    ChangeType,
    compare_content,
    get_file_hash,
)


def test_hash_calculation():
    """해시 계산 테스트"""
    print("\n1. 해시 계산 테스트")
    print("-" * 50)

    content1 = "Hello, World!"
    content2 = "Hello, World!"
    content3 = "Hello, World!!"

    hash1 = DiffAnalyzer.calculate_hash(content1)
    hash2 = DiffAnalyzer.calculate_hash(content2)
    hash3 = DiffAnalyzer.calculate_hash(content3)

    print(f"  hash1: {hash1}")
    print(f"  hash2: {hash2}")
    print(f"  hash3: {hash3}")

    assert hash1 == hash2, "동일 콘텐츠는 같은 해시"
    assert hash1 != hash3, "다른 콘텐츠는 다른 해시"

    print("  PASS")
    return True


def test_text_change_extraction():
    """텍스트 변경 추출 테스트"""
    print("\n2. 텍스트 변경 추출 테스트")
    print("-" * 50)

    original = """# 테스트 문서

## 서론

트랜스포머 모델은 어텐션 메커니즘을 사용합니다.
딥러닝은 신경망을 기반으로 합니다.

## 방법

셀프 어텐션이 핵심입니다."""

    modified = """# 테스트 문서

## 서론

트랜스포머 모델은 주의 메커니즘을 사용합니다.
딥러닝은 신경망을 기반으로 합니다.
새로운 문장이 추가되었습니다.

## 방법

셀프 어텐션이 핵심입니다."""

    analyzer = DiffAnalyzer(use_llm_analysis=False)
    changes = analyzer.extract_text_changes(original, modified)

    print(f"  변경 사항 수: {len(changes)}")

    for i, change in enumerate(changes):
        print(f"  {i+1}. [{change.change_type.value}] 라인 {change.line_number}")
        if change.original_text:
            print(f"      - {change.original_text[:40]}...")
        if change.new_text:
            print(f"      + {change.new_text[:40]}...")

    assert len(changes) > 0, "변경 사항이 있어야 함"

    # 수정된 라인 확인
    modified_changes = [c for c in changes if c.change_type == ChangeType.MODIFIED]
    added_changes = [c for c in changes if c.change_type == ChangeType.ADDED]

    print(f"\n  수정: {len(modified_changes)}건, 추가: {len(added_changes)}건")

    print("  PASS")
    return True


def test_compare_content():
    """콘텐츠 비교 테스트"""
    print("\n3. 콘텐츠 비교 테스트")
    print("-" * 50)

    original = """# 트랜스포머 논문

어텐션 메커니즘은 시퀀스 모델링에 사용됩니다.
셀프 어텐션은 트랜스포머의 핵심입니다."""

    modified = """# 트랜스포머 논문

주의 메커니즘은 시퀀스 모델링에 사용됩니다.
셀프 주의는 트랜스포머의 핵심입니다."""

    result = compare_content(original, modified, use_llm=False)

    print(f"  변경 여부: {result.has_changes}")
    print(f"  원본 해시: {result.original_hash[:16]}...")
    print(f"  현재 해시: {result.current_hash[:16]}...")
    print(f"  텍스트 변경: {len(result.text_changes)}건")
    print(f"  용어 변경: {len(result.term_changes)}건")

    assert result.has_changes, "변경이 있어야 함"
    assert result.original_hash != result.current_hash

    print("  PASS")
    return True


def test_no_changes():
    """변경 없음 테스트"""
    print("\n4. 변경 없음 테스트")
    print("-" * 50)

    content = """# 동일 문서

내용이 같습니다."""

    result = compare_content(content, content, use_llm=False)

    print(f"  변경 여부: {result.has_changes}")
    print(f"  해시 일치: {result.original_hash == result.current_hash}")

    assert not result.has_changes, "변경이 없어야 함"
    assert result.original_hash == result.current_hash

    print("  PASS")
    return True


def test_change_summary():
    """변경 요약 테스트"""
    print("\n5. 변경 요약 테스트")
    print("-" * 50)

    original = "트랜스포머는 어텐션을 사용합니다."
    modified = "트랜스포머는 주의 메커니즘을 사용합니다."

    analyzer = DiffAnalyzer(use_llm_analysis=False)
    result = analyzer.compare_with_content(original, modified)
    summary = analyzer.get_change_summary(result)

    print(f"  변경 여부: {summary['has_changes']}")
    print(f"  텍스트 변경: {summary.get('total_text_changes', 0)}건")
    print(f"  용어 변경: {summary.get('term_changes_count', 0)}건")

    if summary.get('change_types'):
        print(f"  변경 유형: {summary['change_types']}")

    print("  PASS")
    return True


def test_diff_report():
    """변경 보고서 테스트"""
    print("\n6. 변경 보고서 테스트")
    print("-" * 50)

    original = """# AI 논문

## 서론
딥러닝은 혁신적입니다.
어텐션 메커니즘이 중요합니다.

## 방법
트랜스포머를 사용합니다."""

    modified = """# AI 논문

## 서론
딥러닝은 혁신적입니다.
주의 메커니즘이 중요합니다.
새로운 발견을 소개합니다.

## 방법
트랜스포머를 사용합니다."""

    analyzer = DiffAnalyzer(use_llm_analysis=False)
    result = analyzer.compare_with_content(original, modified)
    report = analyzer.format_diff_report(result)

    print("  보고서 미리보기:")
    print("-" * 40)
    # 보고서 일부만 출력
    lines = report.split("\n")
    for line in lines[:15]:
        print(f"  {line}")
    if len(lines) > 15:
        print("  ...")
    print("-" * 40)

    assert "변경 분석 보고서" in report
    assert "텍스트 변경" in report

    print("  PASS")
    return True


def test_heuristic_term_detection():
    """휴리스틱 용어 변경 감지 테스트"""
    print("\n7. 휴리스틱 용어 변경 감지 테스트")
    print("-" * 50)

    # 이 테스트는 DB에 용어가 있을 때만 작동
    # 데모용으로 수동 TextChange 생성

    analyzer = DiffAnalyzer(use_llm_analysis=False)

    text_changes = [
        TextChange(
            change_type=ChangeType.MODIFIED,
            original_text="트랜스포머는 어텐션 메커니즘을 사용합니다.",
            new_text="트랜스포머는 주의 메커니즘을 사용합니다.",
            line_number=5,
            context="## 서론"
        )
    ]

    # 휴리스틱 분석 (DB 용어 기반)
    term_changes = analyzer.analyze_term_changes_heuristic(text_changes)

    print(f"  감지된 용어 변경: {len(term_changes)}건")
    for tc in term_changes:
        print(f"    - {tc.source_text}: '{tc.old_target}' → '{tc.new_target}'")
        print(f"      확신도: {tc.confidence:.0%}")

    # DB에 용어가 있으면 감지되어야 함
    # 없으면 빈 리스트 반환 (정상)
    print(f"  (DB 용어에 따라 결과가 달라질 수 있음)")

    print("  PASS")
    return True


def test_llm_term_analysis():
    """LLM 용어 변경 분석 테스트 (옵션)"""
    print("\n8. LLM 용어 변경 분석 테스트")
    print("-" * 50)

    if "--run-llm" not in sys.argv:
        print("  [SKIP] LLM 테스트는 --run-llm 플래그로 실행하세요.")
        return True

    original = """트랜스포머는 어텐션 메커니즘을 사용합니다.
셀프 어텐션이 핵심 구성요소입니다."""

    modified = """트랜스포머는 주의 메커니즘을 사용합니다.
자기 주의가 핵심 구성요소입니다."""

    analyzer = DiffAnalyzer(use_llm_analysis=True)
    result = analyzer.compare_with_content(original, modified)

    print(f"  LLM 분석 용어 변경: {len(result.term_changes)}건")
    for tc in result.term_changes:
        print(f"    - {tc.source_text}: '{tc.old_target}' → '{tc.new_target}'")
        print(f"      확신도: {tc.confidence:.0%}")

    print("  PASS")
    return True


def main():
    print("=" * 60)
    print("Diff Analyzer 테스트 시작")
    print("=" * 60)

    results = {
        "해시 계산": test_hash_calculation(),
        "텍스트 변경 추출": test_text_change_extraction(),
        "콘텐츠 비교": test_compare_content(),
        "변경 없음": test_no_changes(),
        "변경 요약": test_change_summary(),
        "변경 보고서": test_diff_report(),
        "휴리스틱 용어 감지": test_heuristic_term_detection(),
        "LLM 용어 분석": test_llm_term_analysis(),
    }

    print("\n" + "=" * 60)
    print("테스트 결과:")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("모든 Diff Analyzer 테스트 통과!")
    else:
        print("일부 테스트 실패")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
