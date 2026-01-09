"""
LLM 클라이언트 테스트 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_llm_client, count_tokens, translate_text


def test_initialization():
    """클라이언트 초기화 테스트"""
    print("\n🔧 클라이언트 초기화 테스트")
    print("-" * 50)

    client = get_llm_client()
    print(f"✅ 모델: {client.model}")
    print(f"✅ OpenAI 클라이언트 초기화 성공")

    return True


def test_token_counting():
    """토큰 카운트 테스트"""
    print("\n🔢 토큰 카운트 테스트")
    print("-" * 50)

    test_texts = [
        "Hello, world!",
        "The Transformer model uses self-attention mechanism.",
        "트랜스포머 모델은 셀프 어텐션 메커니즘을 사용합니다.",
    ]

    for text in test_texts:
        tokens = count_tokens(text)
        print(f"✅ \"{text[:30]}...\" → {tokens} 토큰")

    return True


def test_translation():
    """번역 테스트"""
    print("\n🌐 번역 테스트")
    print("-" * 50)

    # 간단한 테스트 텍스트
    test_text = """
    The Transformer architecture has revolutionized natural language processing.
    It uses self-attention mechanisms to process sequential data in parallel,
    making it significantly faster than recurrent neural networks.
    """

    terminology_prompt = """
- Transformer: 트랜스포머
- self-attention: 셀프 어텐션
- natural language processing: 자연어 처리
- recurrent neural networks: 순환 신경망
"""

    print("원문:")
    print(f"  {test_text.strip()[:80]}...")

    result = translate_text(test_text, terminology_prompt)

    print(f"\n번역문:")
    print(f"  {result['translated_text'][:100]}...")
    print(f"\n토큰 사용량:")
    print(f"  - 입력: {result['input_tokens']}")
    print(f"  - 출력: {result['output_tokens']}")
    print(f"  - 총합: {result['total_tokens']}")

    return True


def test_metadata_extraction():
    """메타데이터 추출 테스트"""
    print("\n📋 메타데이터 추출 테스트")
    print("-" * 50)

    # 간단한 논문 앞부분 시뮬레이션
    sample_paper = """
    Attention Is All You Need

    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

    Abstract

    The dominant sequence transduction models are based on complex recurrent or
    convolutional neural networks that include an encoder and a decoder. The best
    performing models also connect the encoder and decoder through an attention
    mechanism. We propose a new simple network architecture, the Transformer,
    based solely on attention mechanisms, dispensing with recurrence and convolutions
    entirely.
    """

    client = get_llm_client()
    result = client.extract_paper_metadata(sample_paper)

    print(f"✅ 제목: {result.get('title', 'N/A')}")
    print(f"✅ 한국어 제목: {result.get('title_ko', 'N/A')}")
    print(f"✅ 저자: {', '.join(result.get('authors', [])[:3])}...")
    print(f"✅ 도메인: {result.get('domain', 'N/A')}")

    return True


def main():
    print("🧪 LLM 클라이언트 테스트 시작")
    print("=" * 60)

    results = {
        "초기화": test_initialization(),
        "토큰 카운트": test_token_counting(),
        "번역": test_translation(),
        "메타데이터 추출": test_metadata_extraction(),
    }

    print("\n" + "=" * 60)
    print("📊 테스트 결과:")
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 모든 LLM 클라이언트 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
