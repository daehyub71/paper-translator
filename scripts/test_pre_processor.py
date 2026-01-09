"""
Pre-processor 테스트 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.processors import (
    PreProcessor,
    ProcessedChunk,
    Chunk,
    preprocess_chunk,
    preprocess_chunks,
    build_terminology_prompt,
)


def test_get_all_terms():
    """전체 용어 조회 테스트"""
    print("\n1. 전체 용어 조회 테스트")
    print("-" * 50)

    preprocessor = PreProcessor()
    terms = preprocessor.get_all_terms()

    print(f"  총 용어 수: {len(terms)}")
    if terms:
        print("  샘플 용어:")
        for term in terms[:5]:
            print(f"    - {term['source_text']} → {term['target_text']}")

    return len(terms) > 0


def test_get_terms_by_domain():
    """도메인별 용어 조회 테스트"""
    print("\n2. 도메인별 용어 조회 테스트")
    print("-" * 50)

    preprocessor = PreProcessor()

    domains = ["NLP", "CV", "RL", "General"]
    for domain in domains:
        terms = preprocessor.get_terms_by_domain(domain)
        print(f"  {domain}: {len(terms)}개 용어")

    return True


def test_find_matching_terms():
    """매칭 용어 추출 테스트"""
    print("\n3. 매칭 용어 추출 테스트")
    print("-" * 50)

    preprocessor = PreProcessor()

    test_text = """
    The Transformer architecture uses self-attention mechanism
    to process sequential data. Unlike LSTM and RNN models,
    transformers can process all tokens in parallel.
    The model is trained using backpropagation with
    gradient descent optimization. We use dropout for
    regularization and batch normalization for stable training.
    """

    matched = preprocessor.find_matching_terms(test_text)
    print(f"  입력 텍스트 길이: {len(test_text)} chars")
    print(f"  매칭된 용어 수: {len(matched)}")

    if matched:
        print("  매칭된 용어:")
        for term in matched[:10]:
            print(f"    - {term['source_text']} → {term['target_text']} "
                  f"(매칭 {term.get('match_count', 0)}회)")

    return len(matched) > 0


def test_build_terminology_prompt():
    """용어 프롬프트 생성 테스트"""
    print("\n4. 용어 프롬프트 생성 테스트")
    print("-" * 50)

    preprocessor = PreProcessor()

    # 샘플 용어
    sample_terms = [
        {"source_text": "Transformer", "target_text": "트랜스포머"},
        {"source_text": "self-attention", "target_text": "셀프 어텐션"},
        {"source_text": "backpropagation", "target_text": "역전파"},
    ]

    prompt = preprocessor.build_terminology_prompt(sample_terms)
    print("  생성된 프롬프트:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)

    # 설명 포함 테스트
    terms_with_desc = [
        {
            "source_text": "attention",
            "target_text": "어텐션",
            "description": "입력의 중요 부분에 집중하는 메커니즘"
        },
    ]
    preprocessor_with_desc = PreProcessor(include_descriptions=True)
    prompt_with_desc = preprocessor_with_desc.build_terminology_prompt(terms_with_desc)
    print("\n  설명 포함 프롬프트:")
    print("-" * 40)
    print(prompt_with_desc)
    print("-" * 40)

    return "Transformer" in prompt and "트랜스포머" in prompt


def test_process_chunk():
    """단일 청크 전처리 테스트"""
    print("\n5. 단일 청크 전처리 테스트")
    print("-" * 50)

    # 테스트용 청크 생성
    test_chunk = Chunk(
        index=0,
        content="""
        The attention mechanism allows the model to focus on
        relevant parts of the input. Self-attention computes
        relationships between all positions in a sequence.
        This is implemented using queries, keys, and values.
        """,
        section_title="Introduction",
        section_index=0,
        token_count=50,
        start_char=0,
        end_char=200,
        has_overlap=False,
    )

    preprocessor = PreProcessor()
    processed = preprocessor.process_chunk(
        chunk=test_chunk,
        paper_title="Attention Is All You Need"
    )

    print(f"  원본 청크 섹션: {processed.chunk.section_title}")
    print(f"  매칭 용어 수: {len(processed.matched_terms)}")
    print(f"  컨텍스트 힌트: {processed.context_hint}")

    if processed.matched_terms:
        print("  매칭된 용어:")
        for term in processed.matched_terms[:5]:
            print(f"    - {term['source_text']} → {term['target_text']}")

    if processed.terminology_prompt:
        print("\n  용어 프롬프트 (일부):")
        print(f"    {processed.terminology_prompt[:200]}...")

    return isinstance(processed, ProcessedChunk)


def test_process_multiple_chunks():
    """여러 청크 전처리 테스트"""
    print("\n6. 여러 청크 전처리 테스트")
    print("-" * 50)

    # 테스트용 청크 목록
    chunks = [
        Chunk(
            index=0,
            content="The Transformer uses self-attention for sequence modeling.",
            section_title="Abstract",
            section_index=0,
            token_count=10,
            start_char=0,
            end_char=50,
        ),
        Chunk(
            index=1,
            content="We train our model using gradient descent with dropout regularization.",
            section_title="Methods",
            section_index=0,
            token_count=12,
            start_char=51,
            end_char=120,
        ),
        Chunk(
            index=2,
            content="The attention weights are computed using softmax function.",
            section_title="Methods",
            section_index=1,
            token_count=10,
            start_char=121,
            end_char=180,
        ),
    ]

    preprocessor = PreProcessor()
    processed_chunks = preprocessor.process_chunks(
        chunks=chunks,
        paper_title="Test Paper"
    )

    print(f"  처리된 청크 수: {len(processed_chunks)}")

    for i, pc in enumerate(processed_chunks):
        print(f"\n  청크 {i} ({pc.chunk.section_title}):")
        print(f"    매칭 용어: {len(pc.matched_terms)}개")
        if pc.matched_terms:
            terms_str = ", ".join(t['source_text'] for t in pc.matched_terms[:3])
            print(f"    용어: {terms_str}...")

    # 요약 정보
    summary = preprocessor.get_processing_summary(processed_chunks)
    print(f"\n  전처리 요약:")
    print(f"    총 청크: {summary['total_chunks']}")
    print(f"    용어 있는 청크: {summary['chunks_with_terms']}")
    print(f"    총 매칭: {summary['total_term_matches']}")
    print(f"    고유 용어: {summary['unique_terms']}")

    return len(processed_chunks) == 3


def test_convenience_functions():
    """편의 함수 테스트"""
    print("\n7. 편의 함수 테스트")
    print("-" * 50)

    # 단일 청크 전처리
    chunk = Chunk(
        index=0,
        content="Neural networks use activation functions like ReLU.",
        section_title="Background",
        section_index=0,
        token_count=10,
        start_char=0,
        end_char=50,
    )

    processed = preprocess_chunk(chunk, paper_title="Test")
    print(f"  preprocess_chunk: {len(processed.matched_terms)} terms matched")

    # 여러 청크 전처리
    chunks = [chunk]
    processed_list = preprocess_chunks(chunks, paper_title="Test")
    print(f"  preprocess_chunks: {len(processed_list)} chunks processed")

    # 프롬프트 생성
    terms = [{"source_text": "test", "target_text": "테스트"}]
    prompt = build_terminology_prompt(terms)
    print(f"  build_terminology_prompt: {len(prompt)} chars")

    return True


def main():
    print("=" * 60)
    print("Pre-processor 테스트 시작")
    print("=" * 60)

    results = {
        "전체 용어 조회": test_get_all_terms(),
        "도메인별 조회": test_get_terms_by_domain(),
        "매칭 용어 추출": test_find_matching_terms(),
        "프롬프트 생성": test_build_terminology_prompt(),
        "단일 청크 전처리": test_process_chunk(),
        "다중 청크 전처리": test_process_multiple_chunks(),
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
        print("모든 Pre-processor 테스트 통과!")
    else:
        print("일부 테스트 실패")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
