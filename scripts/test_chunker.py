"""
청커 테스트 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers import parse_pdf, ParsedSection
from src.processors import TextChunker, Chunk, chunk_text, chunk_sections


def test_token_counting():
    """토큰 카운트 테스트"""
    print("\n1. 토큰 카운트 테스트")
    print("-" * 50)

    chunker = TextChunker()

    test_texts = [
        "Hello, world!",
        "The Transformer model uses self-attention mechanism.",
        "대한민국은 민주공화국이다.",
    ]

    for text in test_texts:
        count = chunker.count_tokens(text)
        print(f"  '{text[:30]}...' -> {count} tokens")

    return True


def test_section_header_detection():
    """섹션 헤더 감지 테스트"""
    print("\n2. 섹션 헤더 감지 테스트")
    print("-" * 50)

    chunker = TextChunker()

    test_text = """
Abstract

This paper presents a novel approach.

1. Introduction

Deep learning has revolutionized...

2.1 Related Work

Previous studies have shown...

3. Methodology

We propose a new method...

4. Experiments

We evaluate our approach...

5. Conclusion

In this paper, we presented...
"""

    headers = chunker.detect_section_headers(test_text)
    print(f"  감지된 헤더 수: {len(headers)}")
    for pos, title in headers:
        print(f"    - pos={pos:3d}: {title}")

    expected_headers = ["Abstract", "Introduction", "Related Work", "Methodology", "Experiments", "Conclusion"]
    detected_titles = [h[1] for h in headers]

    for expected in expected_headers:
        found = any(expected.lower() in title.lower() for title in detected_titles)
        status = "OK" if found else "MISSING"
        print(f"  [{status}] {expected}")

    return len(headers) >= 4  # 최소 4개 이상 감지


def test_token_split():
    """토큰 기반 분할 테스트"""
    print("\n3. 토큰 기반 분할 테스트")
    print("-" * 50)

    chunker = TextChunker(max_chunk_tokens=100, overlap_tokens=20)

    # 긴 텍스트 생성
    long_text = " ".join(["The quick brown fox jumps over the lazy dog."] * 50)
    total_tokens = chunker.count_tokens(long_text)
    print(f"  원본 텍스트 토큰: {total_tokens}")

    chunks = chunker.split_by_tokens(long_text)
    print(f"  분할된 청크 수: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        tokens = chunker.count_tokens(chunk)
        print(f"    청크 {i+1}: {tokens} tokens")

    # 모든 청크가 제한 이하인지 확인
    all_within_limit = all(chunker.count_tokens(c) <= 120 for c in chunks)  # 약간의 여유
    print(f"  모든 청크 토큰 제한 준수: {all_within_limit}")

    return len(chunks) > 1 and all_within_limit


def test_section_chunking():
    """섹션 청킹 테스트"""
    print("\n4. 섹션 청킹 테스트")
    print("-" * 50)

    chunker = TextChunker(max_chunk_tokens=500, overlap_tokens=50)

    # 테스트용 섹션 생성
    section = ParsedSection(
        title="Introduction",
        content=" ".join(["This is a test sentence about machine learning."] * 100),
        page_start=1,
        page_end=2,
        tables=[]
    )

    chunks = chunker.chunk_section(section)
    print(f"  섹션 '{section.title}' -> {len(chunks)} 청크")

    for chunk in chunks:
        print(f"    [{chunk.index}] {chunk.section_title} (part {chunk.section_index}): "
              f"{chunk.token_count} tokens, overlap={chunk.has_overlap}")

    return len(chunks) >= 1


def test_hybrid_chunking_with_paper():
    """실제 논문으로 하이브리드 청킹 테스트"""
    print("\n5. 실제 논문 하이브리드 청킹 테스트")
    print("-" * 50)

    try:
        print("  논문 다운로드 중 (1706.03762)...")
        paper = parse_pdf("1706.03762", exclude_references=True)

        chunker = TextChunker(max_chunk_tokens=2000, overlap_tokens=200, strategy="hybrid")
        chunks = chunker.chunk_paper(paper)

        summary = chunker.get_chunk_summary(chunks)

        print(f"  총 청크 수: {summary['total_chunks']}")
        print(f"  총 토큰 수: {summary['total_tokens']:,}")
        print(f"  평균 청크 토큰: {summary['avg_tokens_per_chunk']}")
        print(f"  고유 섹션 수: {summary['unique_sections']}")
        print(f"  오버랩 청크 수: {summary['chunks_with_overlap']}")
        print(f"  표 포함 청크 수: {summary['chunks_with_tables']}")

        print("\n  청크 상세:")
        for chunk in chunks[:10]:  # 처음 10개만 출력
            print(f"    [{chunk.index:2d}] {chunk.section_title[:25]:25} "
                  f"(part {chunk.section_index}): {chunk.token_count:4d} tokens")

        if len(chunks) > 10:
            print(f"    ... 외 {len(chunks) - 10}개 청크")

        return len(chunks) > 0

    except Exception as e:
        print(f"  오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_strategies():
    """다양한 청킹 전략 비교 테스트"""
    print("\n6. 청킹 전략 비교 테스트")
    print("-" * 50)

    try:
        print("  논문 다운로드 중...")
        paper = parse_pdf("1706.03762", exclude_references=True)

        strategies = ["token", "section", "hybrid"]
        results = {}

        for strategy in strategies:
            chunker = TextChunker(
                max_chunk_tokens=2000,
                overlap_tokens=200,
                strategy=strategy
            )
            chunks = chunker.chunk_paper(paper)
            summary = chunker.get_chunk_summary(chunks)
            results[strategy] = summary

            print(f"\n  [{strategy.upper()}] 전략:")
            print(f"    청크 수: {summary['total_chunks']}")
            print(f"    총 토큰: {summary['total_tokens']:,}")
            print(f"    평균 토큰: {summary['avg_tokens_per_chunk']}")

        return True

    except Exception as e:
        print(f"  오류: {e}")
        return False


def test_convenience_functions():
    """편의 함수 테스트"""
    print("\n7. 편의 함수 테스트")
    print("-" * 50)

    # chunk_text 테스트
    long_text = " ".join(["Testing the chunking function."] * 100)
    chunks = chunk_text(long_text, max_tokens=100)
    print(f"  chunk_text: {len(chunks)} 청크 생성")

    # chunk_sections 테스트 (논문 사용)
    try:
        paper = parse_pdf("1706.03762", exclude_references=True)
        chunks = chunk_sections(paper, strategy="hybrid")
        print(f"  chunk_sections: {len(chunks)} 청크 생성")
        return True
    except Exception as e:
        print(f"  chunk_sections 오류: {e}")
        return False


def main():
    print("=" * 60)
    print("청커 테스트 시작")
    print("=" * 60)

    results = {
        "토큰 카운트": test_token_counting(),
        "섹션 헤더 감지": test_section_header_detection(),
        "토큰 분할": test_token_split(),
        "섹션 청킹": test_section_chunking(),
        "하이브리드 청킹": test_hybrid_chunking_with_paper(),
        "전략 비교": test_different_strategies(),
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
        print("모든 청커 테스트 통과!")
    else:
        print("일부 테스트 실패")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
