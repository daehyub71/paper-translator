"""
Translator 테스트 스크립트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.processors import (
    Translator,
    TranslatedChunk,
    TranslationStatus,
    TranslationStats,
    Chunk,
    ProcessedChunk,
    translate_chunk,
    translate_chunks,
    estimate_translation_cost,
)


def test_translator_init():
    """Translator 초기화 테스트"""
    print("\n1. Translator 초기화 테스트")
    print("-" * 50)

    translator = Translator(
        temperature=0.1,
        max_tokens=4000,
        max_retries=3,
        retry_delay=1.0,
        rate_limit_delay=0.5
    )

    print(f"  temperature: {translator.temperature}")
    print(f"  max_tokens: {translator.max_tokens}")
    print(f"  max_retries: {translator.max_retries}")
    print(f"  rate_limit_delay: {translator.rate_limit_delay}")

    return True


def test_build_prompts():
    """프롬프트 빌드 테스트"""
    print("\n2. 프롬프트 빌드 테스트")
    print("-" * 50)

    translator = Translator()

    # 시스템 프롬프트 (용어 없음)
    system_prompt = translator._build_system_prompt()
    print(f"  기본 시스템 프롬프트 길이: {len(system_prompt)} chars")

    # 시스템 프롬프트 (용어 포함)
    terminology = "- Transformer → 트랜스포머\n- attention → 어텐션"
    context = "논문 제목: Attention Is All You Need | 현재 섹션: Introduction"
    system_prompt_full = translator._build_system_prompt(terminology, context)
    print(f"  전체 시스템 프롬프트 길이: {len(system_prompt_full)} chars")

    # 사용자 프롬프트
    user_prompt = translator._build_user_prompt("The Transformer uses self-attention.")
    print(f"  사용자 프롬프트 길이: {len(user_prompt)} chars")

    assert "번역" in system_prompt
    assert "Transformer" in system_prompt_full
    assert "self-attention" in user_prompt

    print("  PASS")
    return True


def test_estimate_cost():
    """비용 추정 테스트"""
    print("\n3. 비용 추정 테스트")
    print("-" * 50)

    # 테스트용 청크 생성
    chunks = []
    for i in range(5):
        chunk = Chunk(
            index=i,
            content=f"This is test content for chunk {i}. " * 50,
            section_title=f"Section {i}",
            section_index=0,
            token_count=200,
            start_char=i * 500,
            end_char=(i + 1) * 500
        )
        processed = ProcessedChunk(
            chunk=chunk,
            matched_terms=[],
            terminology_prompt="- test → 테스트",
            context_hint=f"섹션 {i}"
        )
        chunks.append(processed)

    translator = Translator()
    estimate = translator.estimate_cost(chunks)

    print(f"  청크 수: {estimate['total_chunks']}")
    print(f"  예상 입력 토큰: {estimate['estimated_input_tokens']}")
    print(f"  예상 출력 토큰: {estimate['estimated_output_tokens']}")
    print(f"  예상 총 토큰: {estimate['estimated_total_tokens']}")
    print(f"  예상 비용: {estimate['estimated_total_cost_usd']}")
    print(f"  예상 시간: {estimate['estimated_time_sec']}초")

    assert estimate['total_chunks'] == 5
    assert estimate['estimated_input_tokens'] > 0

    print("  PASS")
    return True


def test_translation_stats():
    """통계 테스트"""
    print("\n4. 통계 테스트")
    print("-" * 50)

    translator = Translator()

    # 초기 통계
    stats = translator.get_stats()
    print(f"  초기 통계: total_chunks={stats.total_chunks}")

    # 요약
    summary = translator.get_stats_summary()
    print(f"  요약: {summary}")

    return True


def test_single_translation():
    """단일 청크 번역 테스트 (실제 API 호출)"""
    print("\n5. 단일 청크 번역 테스트 (API 호출)")
    print("-" * 50)

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIP - OPENAI_API_KEY가 설정되지 않음")
        return True

    # 짧은 테스트 텍스트
    chunk = Chunk(
        index=0,
        content="The Transformer architecture uses self-attention mechanisms to process sequential data efficiently.",
        section_title="Introduction",
        section_index=0,
        token_count=20,
        start_char=0,
        end_char=100
    )

    processed = ProcessedChunk(
        chunk=chunk,
        matched_terms=[
            {"source_text": "Transformer", "target_text": "트랜스포머"},
            {"source_text": "self-attention", "target_text": "셀프 어텐션"},
        ],
        terminology_prompt="- Transformer → 트랜스포머\n- self-attention → 셀프 어텐션",
        context_hint="논문 제목: Attention Is All You Need"
    )

    translator = Translator(temperature=0.1)
    result = translator.translate_chunk(processed)

    print(f"  상태: {result.status.value}")
    print(f"  입력 토큰: {result.input_tokens}")
    print(f"  출력 토큰: {result.output_tokens}")
    print(f"  번역 시간: {result.translation_time:.2f}초")

    if result.status == TranslationStatus.COMPLETED:
        print(f"\n  번역 결과:")
        print(f"  {result.translated_text[:200]}...")
        print("\n  PASS")
        return True
    else:
        print(f"  에러: {result.error_message}")
        return False


def test_batch_translation():
    """배치 번역 테스트 (실제 API 호출)"""
    print("\n6. 배치 번역 테스트 (API 호출)")
    print("-" * 50)

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIP - OPENAI_API_KEY가 설정되지 않음")
        return True

    # 2개의 짧은 청크
    chunks = []
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
    ]

    for i, text in enumerate(texts):
        chunk = Chunk(
            index=i,
            content=text,
            section_title="Introduction",
            section_index=i,
            token_count=15,
            start_char=i * 100,
            end_char=(i + 1) * 100
        )
        processed = ProcessedChunk(
            chunk=chunk,
            matched_terms=[],
            terminology_prompt="- machine learning → 머신러닝\n- deep learning → 딥러닝",
            context_hint=""
        )
        chunks.append(processed)

    # 진행 상황 콜백
    def progress(current, total, message):
        print(f"    [{current}/{total}] {message}")

    translator = Translator(
        temperature=0.1,
        rate_limit_delay=0.5,
        progress_callback=progress
    )

    results = translator.translate_chunks(chunks)
    stats = translator.get_stats_summary()

    print(f"\n  번역 결과:")
    for i, result in enumerate(results):
        status = "OK" if result.status == TranslationStatus.COMPLETED else "FAIL"
        print(f"    청크 {i}: {status}")
        if result.status == TranslationStatus.COMPLETED:
            print(f"      -> {result.translated_text[:50]}...")

    print(f"\n  통계:")
    print(f"    완료: {stats['completed']}/{stats['total_chunks']}")
    print(f"    총 토큰: {stats['total_tokens']}")
    print(f"    총 시간: {stats['total_time_sec']}초")
    print(f"    예상 비용: {stats['estimated_cost_usd']}")

    success = all(r.status == TranslationStatus.COMPLETED for r in results)
    if success:
        print("\n  PASS")
    return success


def test_convenience_functions():
    """편의 함수 테스트"""
    print("\n7. 편의 함수 테스트")
    print("-" * 50)

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIP - OPENAI_API_KEY가 설정되지 않음")
        return True

    chunk = Chunk(
        index=0,
        content="Neural networks learn patterns from data.",
        section_title="Background",
        section_index=0,
        token_count=10,
        start_char=0,
        end_char=50
    )

    processed = ProcessedChunk(
        chunk=chunk,
        matched_terms=[],
        terminology_prompt="- neural network → 신경망",
        context_hint=""
    )

    # translate_chunk
    result = translate_chunk(processed)
    print(f"  translate_chunk: {result.status.value}")

    # estimate_translation_cost
    estimate = estimate_translation_cost([processed])
    print(f"  estimate_translation_cost: {estimate['estimated_total_cost_usd']}")

    return result.status == TranslationStatus.COMPLETED


def main():
    print("=" * 60)
    print("Translator 테스트 시작")
    print("=" * 60)

    results = {
        "초기화": test_translator_init(),
        "프롬프트 빌드": test_build_prompts(),
        "비용 추정": test_estimate_cost(),
        "통계": test_translation_stats(),
        "단일 번역": test_single_translation(),
        "배치 번역": test_batch_translation(),
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
        print("모든 Translator 테스트 통과!")
    else:
        print("일부 테스트 실패")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
