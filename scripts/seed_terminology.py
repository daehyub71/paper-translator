"""
초기 전문용어 데이터 삽입 스크립트
AI/ML 분야 기본 용어를 terminology_mappings 테이블에 삽입
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()


# 초기 용어 데이터
SEED_TERMS = [
    # ============================================
    # Architecture (아키텍처)
    # ============================================
    {"source_text": "Transformer", "target_text": "트랜스포머", "mapping_type": "word", "domain": "NLP"},
    {"source_text": "attention mechanism", "target_text": "어텐션 메커니즘", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "self-attention", "target_text": "셀프 어텐션", "mapping_type": "word", "domain": "NLP"},
    {"source_text": "multi-head attention", "target_text": "멀티헤드 어텐션", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "feed-forward network", "target_text": "피드포워드 네트워크", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "residual connection", "target_text": "잔차 연결", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "layer normalization", "target_text": "레이어 정규화", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "encoder", "target_text": "인코더", "mapping_type": "word", "domain": "General"},
    {"source_text": "decoder", "target_text": "디코더", "mapping_type": "word", "domain": "General"},
    {"source_text": "neural network", "target_text": "신경망", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "deep learning", "target_text": "딥러닝", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "convolutional neural network", "target_text": "합성곱 신경망", "mapping_type": "phrase", "domain": "CV"},
    {"source_text": "recurrent neural network", "target_text": "순환 신경망", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "LSTM", "target_text": "LSTM", "mapping_type": "word", "domain": "NLP"},
    {"source_text": "GRU", "target_text": "GRU", "mapping_type": "word", "domain": "NLP"},

    # ============================================
    # Training (학습)
    # ============================================
    {"source_text": "fine-tuning", "target_text": "미세조정", "mapping_type": "word", "domain": "General"},
    {"source_text": "pre-training", "target_text": "사전학습", "mapping_type": "word", "domain": "General"},
    {"source_text": "transfer learning", "target_text": "전이학습", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "gradient descent", "target_text": "경사 하강법", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "stochastic gradient descent", "target_text": "확률적 경사 하강법", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "backpropagation", "target_text": "역전파", "mapping_type": "word", "domain": "General"},
    {"source_text": "learning rate", "target_text": "학습률", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "batch size", "target_text": "배치 크기", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "epoch", "target_text": "에폭", "mapping_type": "word", "domain": "General"},
    {"source_text": "overfitting", "target_text": "과적합", "mapping_type": "word", "domain": "General"},
    {"source_text": "underfitting", "target_text": "과소적합", "mapping_type": "word", "domain": "General"},
    {"source_text": "regularization", "target_text": "정규화", "mapping_type": "word", "domain": "General"},
    {"source_text": "dropout", "target_text": "드롭아웃", "mapping_type": "word", "domain": "General"},
    {"source_text": "weight decay", "target_text": "가중치 감쇠", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "optimizer", "target_text": "옵티마이저", "mapping_type": "word", "domain": "General"},
    {"source_text": "Adam", "target_text": "Adam", "mapping_type": "word", "domain": "General"},
    {"source_text": "loss function", "target_text": "손실 함수", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "cross-entropy", "target_text": "크로스 엔트로피", "mapping_type": "word", "domain": "General"},

    # ============================================
    # LLM Specific (대규모 언어 모델)
    # ============================================
    {"source_text": "large language model", "target_text": "대규모 언어 모델", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "hallucination", "target_text": "환각 현상", "mapping_type": "word", "domain": "NLP"},
    {"source_text": "prompt engineering", "target_text": "프롬프트 엔지니어링", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "in-context learning", "target_text": "인컨텍스트 학습", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "chain-of-thought", "target_text": "사고의 연쇄", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "retrieval-augmented generation", "target_text": "검색 증강 생성(RAG)", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "RAG", "target_text": "RAG", "mapping_type": "word", "domain": "NLP"},
    {"source_text": "tokenization", "target_text": "토큰화", "mapping_type": "word", "domain": "NLP"},
    {"source_text": "embedding", "target_text": "임베딩", "mapping_type": "word", "domain": "General"},
    {"source_text": "word embedding", "target_text": "단어 임베딩", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "positional encoding", "target_text": "위치 인코딩", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "context window", "target_text": "컨텍스트 윈도우", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "zero-shot", "target_text": "제로샷", "mapping_type": "word", "domain": "NLP"},
    {"source_text": "few-shot", "target_text": "퓨샷", "mapping_type": "word", "domain": "NLP"},
    {"source_text": "instruction tuning", "target_text": "지시 튜닝", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "RLHF", "target_text": "인간 피드백 강화학습(RLHF)", "mapping_type": "word", "domain": "NLP"},
    {"source_text": "reinforcement learning from human feedback", "target_text": "인간 피드백 강화학습", "mapping_type": "phrase", "domain": "NLP"},

    # ============================================
    # Metrics (평가 지표)
    # ============================================
    {"source_text": "accuracy", "target_text": "정확도", "mapping_type": "word", "domain": "General"},
    {"source_text": "precision", "target_text": "정밀도", "mapping_type": "word", "domain": "General"},
    {"source_text": "recall", "target_text": "재현율", "mapping_type": "word", "domain": "General"},
    {"source_text": "F1 score", "target_text": "F1 점수", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "perplexity", "target_text": "퍼플렉시티", "mapping_type": "word", "domain": "NLP"},
    {"source_text": "BLEU score", "target_text": "BLEU 점수", "mapping_type": "phrase", "domain": "NLP"},
    {"source_text": "ROUGE", "target_text": "ROUGE", "mapping_type": "word", "domain": "NLP"},
    {"source_text": "AUC", "target_text": "AUC", "mapping_type": "word", "domain": "General"},
    {"source_text": "ROC curve", "target_text": "ROC 곡선", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "mean squared error", "target_text": "평균 제곱 오차", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "MSE", "target_text": "MSE", "mapping_type": "word", "domain": "General"},

    # ============================================
    # Common Phrases (자주 쓰이는 표현)
    # ============================================
    {"source_text": "state-of-the-art", "target_text": "최신 기술", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "from scratch", "target_text": "처음부터", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "end-to-end", "target_text": "엔드투엔드", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "out-of-the-box", "target_text": "기본 설정으로", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "benchmark", "target_text": "벤치마크", "mapping_type": "word", "domain": "General"},
    {"source_text": "baseline", "target_text": "베이스라인", "mapping_type": "word", "domain": "General"},
    {"source_text": "ablation study", "target_text": "절제 연구", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "downstream task", "target_text": "다운스트림 태스크", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "upstream task", "target_text": "업스트림 태스크", "mapping_type": "phrase", "domain": "General"},
    {"source_text": "scalability", "target_text": "확장성", "mapping_type": "word", "domain": "General"},
    {"source_text": "inference", "target_text": "추론", "mapping_type": "word", "domain": "General"},
    {"source_text": "latency", "target_text": "지연 시간", "mapping_type": "word", "domain": "General"},
    {"source_text": "throughput", "target_text": "처리량", "mapping_type": "word", "domain": "General"},

    # ============================================
    # Computer Vision (컴퓨터 비전)
    # ============================================
    {"source_text": "image classification", "target_text": "이미지 분류", "mapping_type": "phrase", "domain": "CV"},
    {"source_text": "object detection", "target_text": "객체 탐지", "mapping_type": "phrase", "domain": "CV"},
    {"source_text": "semantic segmentation", "target_text": "의미론적 분할", "mapping_type": "phrase", "domain": "CV"},
    {"source_text": "feature extraction", "target_text": "특징 추출", "mapping_type": "phrase", "domain": "CV"},
    {"source_text": "bounding box", "target_text": "바운딩 박스", "mapping_type": "phrase", "domain": "CV"},
    {"source_text": "Vision Transformer", "target_text": "비전 트랜스포머", "mapping_type": "phrase", "domain": "CV"},
    {"source_text": "ViT", "target_text": "ViT", "mapping_type": "word", "domain": "CV"},

    # ============================================
    # Reinforcement Learning (강화학습)
    # ============================================
    {"source_text": "reinforcement learning", "target_text": "강화학습", "mapping_type": "phrase", "domain": "RL"},
    {"source_text": "reward", "target_text": "보상", "mapping_type": "word", "domain": "RL"},
    {"source_text": "policy", "target_text": "정책", "mapping_type": "word", "domain": "RL"},
    {"source_text": "agent", "target_text": "에이전트", "mapping_type": "word", "domain": "RL"},
    {"source_text": "environment", "target_text": "환경", "mapping_type": "word", "domain": "RL"},
    {"source_text": "action", "target_text": "행동", "mapping_type": "word", "domain": "RL"},
    {"source_text": "state", "target_text": "상태", "mapping_type": "word", "domain": "RL"},
    {"source_text": "Q-learning", "target_text": "Q러닝", "mapping_type": "word", "domain": "RL"},
    {"source_text": "PPO", "target_text": "PPO", "mapping_type": "word", "domain": "RL"},
]


def get_supabase_client() -> Client:
    """Supabase 클라이언트 생성"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다.")

    return create_client(url, key)


def seed_terminology(client: Client, terms: list[dict]) -> dict:
    """용어 데이터 삽입"""
    result = {
        "inserted": 0,
        "skipped": 0,
        "errors": []
    }

    for term in terms:
        try:
            # upsert를 사용하여 중복 시 업데이트
            response = client.table("terminology_mappings").upsert(
                term,
                on_conflict="source_text,domain"
            ).execute()

            if response.data:
                result["inserted"] += 1
            else:
                result["skipped"] += 1

        except Exception as e:
            error_msg = f"{term['source_text']}: {str(e)}"
            result["errors"].append(error_msg)

    return result


def get_term_count(client: Client) -> dict:
    """도메인별 용어 수 조회"""
    response = client.table("terminology_mappings").select("domain").execute()

    if not response.data:
        return {}

    counts = {}
    for row in response.data:
        domain = row["domain"]
        counts[domain] = counts.get(domain, 0) + 1

    return counts


def main():
    print("🌱 Paper Translator 초기 용어 데이터 삽입")
    print("=" * 60)

    # Supabase 클라이언트 생성
    try:
        client = get_supabase_client()
        print("✅ Supabase 연결 성공")
    except Exception as e:
        print(f"❌ Supabase 연결 실패: {e}")
        sys.exit(1)

    # 삽입 전 용어 수
    print("\n📊 삽입 전 용어 현황:")
    counts_before = get_term_count(client)
    if counts_before:
        for domain, count in sorted(counts_before.items()):
            print(f"  - {domain}: {count}개")
    else:
        print("  - (용어 없음)")

    total_before = sum(counts_before.values()) if counts_before else 0

    # 용어 데이터 삽입
    print(f"\n📥 {len(SEED_TERMS)}개의 용어 삽입 중...")

    result = seed_terminology(client, SEED_TERMS)

    print(f"\n📋 삽입 결과:")
    print(f"  - 삽입/업데이트: {result['inserted']}개")
    print(f"  - 건너뜀: {result['skipped']}개")
    print(f"  - 오류: {len(result['errors'])}개")

    if result["errors"]:
        print("\n⚠️ 오류 목록:")
        for error in result["errors"][:5]:  # 최대 5개만 표시
            print(f"  - {error}")
        if len(result["errors"]) > 5:
            print(f"  ... 외 {len(result['errors']) - 5}개")

    # 삽입 후 용어 수
    print("\n📊 삽입 후 용어 현황:")
    counts_after = get_term_count(client)
    for domain, count in sorted(counts_after.items()):
        print(f"  - {domain}: {count}개")

    total_after = sum(counts_after.values())
    new_terms = total_after - total_before

    print("\n" + "=" * 60)
    print(f"🎉 완료! 총 {total_after}개 용어 ({new_terms}개 신규)")


if __name__ == "__main__":
    main()
