"""
Repository 테스트 스크립트
각 Repository의 CRUD 기능 검증
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db import (
    TerminologyRepository,
    TranslationRepository,
    TranslationHistoryRepository,
    TermChangeRepository,
)


def test_terminology_repository():
    """TerminologyRepository 테스트"""
    print("\n📚 TerminologyRepository 테스트")
    print("-" * 50)

    # 1. get_all 테스트
    terms = TerminologyRepository.get_all(limit=5)
    print(f"✅ get_all: {len(terms)}개 용어 조회")

    # 2. get_all with domain 테스트
    nlp_terms = TerminologyRepository.get_all(domain="NLP", limit=5)
    print(f"✅ get_all(domain='NLP'): {len(nlp_terms)}개 용어 조회")

    # 3. get_by_source 테스트
    term = TerminologyRepository.get_by_source("Transformer", domain="NLP")
    if term:
        print(f"✅ get_by_source('Transformer'): {term['target_text']}")
    else:
        print("⚠️ get_by_source('Transformer'): 없음")

    # 4. search 테스트
    search_results = TerminologyRepository.search("attention", limit=3)
    print(f"✅ search('attention'): {len(search_results)}개 결과")

    # 5. get_matching_terms 테스트
    sample_text = "The Transformer model uses self-attention mechanism for better performance."
    matching = TerminologyRepository.get_matching_terms(sample_text, limit=5)
    print(f"✅ get_matching_terms: {len(matching)}개 매칭 용어")
    for m in matching[:3]:
        print(f"   - {m['source_text']} → {m['target_text']}")

    return True


def test_translation_repository():
    """TranslationRepository 테스트"""
    print("\n📄 TranslationRepository 테스트")
    print("-" * 50)

    # 1. create 테스트
    translation = TranslationRepository.create(
        paper_title="[TEST] Attention Is All You Need",
        output_path="./translations/test_attention.md",
        paper_url="https://arxiv.org/pdf/1706.03762",
        arxiv_id="1706.03762_test",
        domain="NLP",
        total_chunks=10
    )

    if translation:
        print(f"✅ create: ID={translation['id'][:8]}...")
        translation_id = translation["id"]

        # 2. get_by_id 테스트
        fetched = TranslationRepository.get_by_id(translation_id)
        print(f"✅ get_by_id: {fetched['paper_title'][:30]}...")

        # 3. update_status 테스트
        updated = TranslationRepository.update_status(translation_id, "completed")
        print(f"✅ update_status: {updated['status']}")

        # 4. update_hashes 테스트
        TranslationRepository.update_hashes(translation_id, "abc123hash", "abc123hash")
        print("✅ update_hashes: 성공")

        # 5. delete 테스트
        deleted = TranslationRepository.delete(translation_id)
        print(f"✅ delete: {deleted}")

        return True
    else:
        print("❌ create 실패")
        return False


def test_translation_history_repository():
    """TranslationHistoryRepository 테스트"""
    print("\n📝 TranslationHistoryRepository 테스트")
    print("-" * 50)

    # 먼저 번역 기록 생성
    translation = TranslationRepository.create(
        paper_title="[TEST] History Test Paper",
        output_path="./translations/test_history.md"
    )

    if not translation:
        print("❌ 번역 기록 생성 실패")
        return False

    translation_id = translation["id"]

    try:
        # 1. create 테스트
        history1 = TranslationHistoryRepository.create(
            translation_id=translation_id,
            chunk_index=0,
            original_text="This is the abstract.",
            translated_text="이것은 초록입니다.",
            section_title="Abstract",
            terms_applied=[{"source": "abstract", "target": "초록"}],
            tokens_used=50
        )
        print(f"✅ create: chunk_index={history1['chunk_index']}")

        # 2. bulk_create 테스트
        chunks = [
            {
                "translation_id": translation_id,
                "chunk_index": 1,
                "original_text": "Introduction text.",
                "translated_text": "서론 텍스트.",
                "section_title": "Introduction"
            },
            {
                "translation_id": translation_id,
                "chunk_index": 2,
                "original_text": "Method text.",
                "translated_text": "방법 텍스트.",
                "section_title": "Method"
            }
        ]
        bulk_result = TranslationHistoryRepository.bulk_create(chunks)
        print(f"✅ bulk_create: {len(bulk_result)}개 생성")

        # 3. get_by_translation 테스트
        all_chunks = TranslationHistoryRepository.get_by_translation(translation_id)
        print(f"✅ get_by_translation: {len(all_chunks)}개 청크")

        # 4. get_chunk 테스트
        chunk = TranslationHistoryRepository.get_chunk(translation_id, 0)
        print(f"✅ get_chunk(0): {chunk['section_title']}")

        return True

    finally:
        # 정리: 번역 기록 삭제 (cascade로 history도 삭제됨)
        TranslationRepository.delete(translation_id)
        print("✅ 테스트 데이터 정리 완료")


def test_term_change_repository():
    """TermChangeRepository 테스트"""
    print("\n📋 TermChangeRepository 테스트")
    print("-" * 50)

    # 1. log_add 테스트
    add_log = TermChangeRepository.log_add(
        source_text="test_term",
        new_target_text="테스트 용어",
        detected_from="manual"
    )
    print(f"✅ log_add: {add_log['change_type']}")

    # 2. log_update 테스트
    update_log = TermChangeRepository.log_update(
        source_text="test_term",
        old_target_text="테스트 용어",
        new_target_text="테스트 용어 수정",
        detected_from="markdown_sync"
    )
    print(f"✅ log_update: {update_log['change_type']}")

    # 3. log_delete 테스트
    delete_log = TermChangeRepository.log_delete(
        source_text="test_term",
        old_target_text="테스트 용어 수정",
        detected_from="manual"
    )
    print(f"✅ log_delete: {delete_log['change_type']}")

    # 4. get_all 테스트
    all_logs = TermChangeRepository.get_all(limit=5)
    print(f"✅ get_all: {len(all_logs)}개 로그")

    return True


def main():
    print("🧪 Repository 테스트 시작")
    print("=" * 60)

    results = {
        "TerminologyRepository": test_terminology_repository(),
        "TranslationRepository": test_translation_repository(),
        "TranslationHistoryRepository": test_translation_history_repository(),
        "TermChangeRepository": test_term_change_repository(),
    }

    print("\n" + "=" * 60)
    print("📊 테스트 결과:")
    all_passed = True
    for repo, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"  {repo}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 모든 Repository 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
