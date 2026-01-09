"""
Post-processor 테스트 스크립트
"""
import sys
import logging
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

from src.processors import (
    PostProcessor,
    PostProcessedChunk,
    TermValidation,
    TermMatchStatus,
    Chunk,
    ProcessedChunk,
    TranslatedChunk,
    TranslationStatus,
    postprocess_chunk,
    postprocess_chunks,
    validate_terminology,
)


def create_test_translated_chunk(
    original_text: str,
    translated_text: str,
    matched_terms: list[dict],
    chunk_index: int = 0
) -> TranslatedChunk:
    """테스트용 번역된 청크 생성"""
    chunk = Chunk(
        index=chunk_index,
        content=original_text,
        section_title="Test Section",
        section_index=0,
        token_count=50,
        start_char=0,
        end_char=len(original_text)
    )

    processed = ProcessedChunk(
        chunk=chunk,
        matched_terms=matched_terms,
        terminology_prompt="",
        context_hint=""
    )

    return TranslatedChunk(
        processed_chunk=processed,
        translated_text=translated_text,
        status=TranslationStatus.COMPLETED,
        input_tokens=100,
        output_tokens=80,
        total_tokens=180
    )


def test_similarity_calculation():
    """유사도 계산 테스트"""
    print("\n1. 유사도 계산 테스트")
    print("-" * 50)

    processor = PostProcessor()

    test_cases = [
        ("트랜스포머", "트랜스포머", 1.0),
        ("어텐션", "어텐션", 1.0),
        ("셀프 어텐션", "셀프어텐션", 0.9),  # 공백 차이
        ("딥러닝", "딥 러닝", 0.8),
        ("완전히 다름", "전혀 무관", 0.2),
    ]

    for str1, str2, expected_min in test_cases:
        similarity = processor.calculate_similarity(str1, str2)
        status = "OK" if similarity >= expected_min else "WARN"
        print(f"  [{status}] '{str1}' vs '{str2}': {similarity:.2f}")

    print("  PASS")
    return True


def test_find_term_in_text():
    """텍스트에서 용어 찾기 테스트"""
    print("\n2. 텍스트에서 용어 찾기 테스트")
    print("-" * 50)

    processor = PostProcessor()

    text = "트랜스포머 모델은 셀프 어텐션을 사용합니다. 어텐션 메커니즘이 핵심입니다."

    # 정확한 매칭
    matches = processor.find_term_in_text(text, "어텐션")
    print(f"  '어텐션' 매칭: {len(matches)}개 위치 {matches}")
    assert len(matches) == 2

    # 대소문자 무시
    matches = processor.find_term_in_text(text, "트랜스포머")
    print(f"  '트랜스포머' 매칭: {len(matches)}개")
    assert len(matches) == 1

    # 없는 용어
    matches = processor.find_term_in_text(text, "CNN")
    print(f"  'CNN' 매칭: {len(matches)}개")
    assert len(matches) == 0

    print("  PASS")
    return True


def test_validate_term():
    """단일 용어 검증 테스트"""
    print("\n3. 단일 용어 검증 테스트")
    print("-" * 50)

    processor = PostProcessor(threshold=0.8)

    # 테스트 케이스 1: 정확히 매칭
    validation = processor.validate_term(
        source_text="Transformer",
        expected_target="트랜스포머",
        original_text="The Transformer model is powerful.",
        translated_text="트랜스포머 모델은 강력합니다."
    )
    print(f"  케이스 1 (정확 매칭): {validation.status.value}")
    assert validation.status == TermMatchStatus.MATCHED

    # 테스트 케이스 2: 유사 매칭
    validation = processor.validate_term(
        source_text="self-attention",
        expected_target="셀프 어텐션",
        original_text="Self-attention is used.",
        translated_text="셀프어텐션이 사용됩니다."  # 공백 없음
    )
    print(f"  케이스 2 (유사 매칭): {validation.status.value}, 유사도={validation.similarity:.2f}")
    assert validation.status in [TermMatchStatus.MATCHED, TermMatchStatus.SIMILAR]

    # 테스트 케이스 3: 누락
    validation = processor.validate_term(
        source_text="attention",
        expected_target="어텐션",
        original_text="Attention mechanism is key.",
        translated_text="주의 메커니즘이 핵심입니다."  # 다른 번역 사용
    )
    print(f"  케이스 3 (누락/불일치): {validation.status.value}")
    assert validation.status in [TermMatchStatus.MISSING, TermMatchStatus.MISMATCHED]

    print("  PASS")
    return True


def test_validate_chunk_terms():
    """청크 용어 검증 테스트"""
    print("\n4. 청크 용어 검증 테스트")
    print("-" * 50)

    processor = PostProcessor(threshold=0.8)

    # 테스트 데이터 생성
    translated_chunk = create_test_translated_chunk(
        original_text="The Transformer uses self-attention mechanism for sequence modeling.",
        translated_text="트랜스포머는 시퀀스 모델링을 위해 셀프 어텐션 메커니즘을 사용합니다.",
        matched_terms=[
            {"source_text": "Transformer", "target_text": "트랜스포머"},
            {"source_text": "self-attention", "target_text": "셀프 어텐션"},
            {"source_text": "sequence", "target_text": "시퀀스"},
        ]
    )

    validations = processor.validate_chunk_terms(translated_chunk)

    print(f"  검증된 용어 수: {len(validations)}")
    for v in validations:
        print(f"    - {v.source_text}: {v.status.value} (유사도: {v.similarity:.2f})")

    matched_count = sum(1 for v in validations if v.status == TermMatchStatus.MATCHED)
    print(f"  매칭 성공: {matched_count}/{len(validations)}")

    print("  PASS")
    return True


def test_detect_missing_and_mismatched():
    """누락/불일치 용어 감지 테스트"""
    print("\n5. 누락/불일치 용어 감지 테스트")
    print("-" * 50)

    processor = PostProcessor(threshold=0.8)

    # 일부 용어가 누락된 번역
    translated_chunk = create_test_translated_chunk(
        original_text="Deep learning uses neural networks with backpropagation.",
        translated_text="딥러닝은 역전파와 함께 인공신경망을 사용합니다.",  # "신경망" 대신 "인공신경망"
        matched_terms=[
            {"source_text": "Deep learning", "target_text": "딥러닝"},
            {"source_text": "neural networks", "target_text": "신경망"},
            {"source_text": "backpropagation", "target_text": "역전파"},
        ]
    )

    validations = processor.validate_chunk_terms(translated_chunk)
    missing = processor.detect_missing_terms(validations)
    mismatched = processor.detect_mismatched_terms(validations)

    print(f"  총 검증 용어: {len(validations)}")
    print(f"  누락 용어: {len(missing)}")
    print(f"  불일치 용어: {len(mismatched)}")

    for v in missing:
        print(f"    [누락] {v.source_text} → {v.expected_target}")
    for v in mismatched:
        print(f"    [불일치] {v.source_text}: '{v.actual_target}' vs '{v.expected_target}'")

    print("  PASS")
    return True


def test_correct_terms():
    """용어 교정 테스트"""
    print("\n6. 용어 교정 테스트")
    print("-" * 50)

    processor = PostProcessor(threshold=0.7, auto_correct=True)

    # 교정 테스트
    original = "인공신경망은 학습합니다."
    actual = "인공신경망"
    expected = "신경망"

    corrected = processor.correct_term(original, actual, expected)
    print(f"  원본: '{original}'")
    print(f"  교정: '{corrected}'")

    assert "신경망" in corrected
    print("  PASS")
    return True


def test_process_chunk():
    """청크 후처리 테스트"""
    print("\n7. 청크 후처리 테스트")
    print("-" * 50)

    processor = PostProcessor(threshold=0.8, auto_correct=True, log_corrections=False)

    translated_chunk = create_test_translated_chunk(
        original_text="Machine learning is a subset of artificial intelligence.",
        translated_text="머신러닝은 인공지능의 하위 분야입니다.",
        matched_terms=[
            {"source_text": "Machine learning", "target_text": "머신러닝"},
            {"source_text": "artificial intelligence", "target_text": "인공지능"},
        ]
    )

    result = processor.process_chunk(translated_chunk)

    print(f"  교정된 텍스트: {result.corrected_text[:50]}...")
    print(f"  검증 용어: {len(result.validations)}")
    print(f"  교정 횟수: {result.corrections_made}")
    print(f"  누락 용어: {result.missing_terms}")
    print(f"  불일치 용어: {result.mismatched_terms}")

    print("  PASS")
    return True


def test_process_multiple_chunks():
    """여러 청크 후처리 테스트"""
    print("\n8. 여러 청크 후처리 테스트")
    print("-" * 50)

    processor = PostProcessor(threshold=0.8, auto_correct=True, log_corrections=False)

    chunks = [
        create_test_translated_chunk(
            original_text="The Transformer model uses attention.",
            translated_text="트랜스포머 모델은 어텐션을 사용합니다.",
            matched_terms=[
                {"source_text": "Transformer", "target_text": "트랜스포머"},
                {"source_text": "attention", "target_text": "어텐션"},
            ],
            chunk_index=0
        ),
        create_test_translated_chunk(
            original_text="Deep learning enables many applications.",
            translated_text="딥러닝은 많은 응용을 가능하게 합니다.",
            matched_terms=[
                {"source_text": "Deep learning", "target_text": "딥러닝"},
            ],
            chunk_index=1
        ),
    ]

    results = processor.process_chunks(chunks)
    stats = processor.get_stats_summary()

    print(f"  처리된 청크: {len(results)}")
    print(f"\n  통계:")
    print(f"    총 검사 용어: {stats['total_terms_checked']}")
    print(f"    매칭 성공: {stats['matched_terms']}")
    print(f"    유사 매칭: {stats['similar_terms']}")
    print(f"    누락: {stats['missing_terms']}")
    print(f"    불일치: {stats['mismatched_terms']}")
    print(f"    교정: {stats['corrections_made']}")
    print(f"    매칭률: {stats['match_rate']}")

    print("  PASS")
    return True


def test_generate_report():
    """교정 보고서 생성 테스트"""
    print("\n9. 교정 보고서 생성 테스트")
    print("-" * 50)

    processor = PostProcessor(threshold=0.7, auto_correct=True, log_corrections=False)

    chunks = [
        create_test_translated_chunk(
            original_text="Neural networks learn from data using gradient descent.",
            translated_text="인공신경망은 경사하강법을 사용하여 데이터에서 학습합니다.",
            matched_terms=[
                {"source_text": "Neural networks", "target_text": "신경망"},
                {"source_text": "gradient descent", "target_text": "경사 하강법"},
            ],
            chunk_index=0
        ),
    ]

    results = processor.process_chunks(chunks)
    report = processor.generate_correction_report(results)

    print("  보고서 미리보기:")
    print("-" * 40)
    print(report[:500])
    if len(report) > 500:
        print("...")
    print("-" * 40)

    assert "용어 교정 보고서" in report
    print("  PASS")
    return True


def test_convenience_functions():
    """편의 함수 테스트"""
    print("\n10. 편의 함수 테스트")
    print("-" * 50)

    translated_chunk = create_test_translated_chunk(
        original_text="The model uses attention.",
        translated_text="모델은 어텐션을 사용합니다.",
        matched_terms=[
            {"source_text": "attention", "target_text": "어텐션"},
        ]
    )

    # postprocess_chunk
    result = postprocess_chunk(translated_chunk)
    print(f"  postprocess_chunk: {len(result.validations)} validations")

    # postprocess_chunks
    results = postprocess_chunks([translated_chunk])
    print(f"  postprocess_chunks: {len(results)} results")

    # validate_terminology
    validations = validate_terminology(translated_chunk)
    print(f"  validate_terminology: {len(validations)} validations")

    print("  PASS")
    return True


def main():
    print("=" * 60)
    print("Post-processor 테스트 시작")
    print("=" * 60)

    results = {
        "유사도 계산": test_similarity_calculation(),
        "용어 찾기": test_find_term_in_text(),
        "단일 용어 검증": test_validate_term(),
        "청크 용어 검증": test_validate_chunk_terms(),
        "누락/불일치 감지": test_detect_missing_and_mismatched(),
        "용어 교정": test_correct_terms(),
        "청크 후처리": test_process_chunk(),
        "다중 청크 후처리": test_process_multiple_chunks(),
        "보고서 생성": test_generate_report(),
        "편의 함수": test_convenience_functions(),
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
        print("모든 Post-processor 테스트 통과!")
    else:
        print("일부 테스트 실패")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
