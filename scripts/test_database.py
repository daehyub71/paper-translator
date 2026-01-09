"""
데이터베이스 테이블 생성 확인 테스트
Supabase 연결 및 테이블 존재 여부 검증
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


def get_supabase_client() -> Client:
    """Supabase 클라이언트 생성"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다.")

    return create_client(url, key)


def test_table_exists(client: Client, table_name: str) -> dict:
    """테이블 존재 여부 및 기본 조회 테스트"""
    result = {
        "table": table_name,
        "exists": False,
        "can_query": False,
        "row_count": 0,
        "error": None
    }

    try:
        # 테이블에서 데이터 조회 시도
        response = client.table(table_name).select("*").limit(1).execute()
        result["exists"] = True
        result["can_query"] = True
        result["row_count"] = len(response.data) if response.data else 0
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg or "relation" in error_msg:
            result["exists"] = False
        else:
            result["exists"] = True  # 테이블은 있지만 다른 에러
            result["error"] = error_msg

    return result


def test_insert_and_delete(client: Client) -> dict:
    """terminology_mappings 테이블 INSERT/DELETE 테스트"""
    result = {
        "insert": False,
        "select": False,
        "delete": False,
        "error": None
    }

    test_data = {
        "source_text": "__TEST_TERM__",
        "target_text": "__테스트_용어__",
        "mapping_type": "word",
        "domain": "Test"
    }

    try:
        # INSERT 테스트
        insert_response = client.table("terminology_mappings").insert(test_data).execute()
        if insert_response.data:
            result["insert"] = True
            inserted_id = insert_response.data[0]["id"]

            # SELECT 테스트
            select_response = client.table("terminology_mappings").select("*").eq("id", inserted_id).execute()
            if select_response.data:
                result["select"] = True

            # DELETE 테스트 (정리)
            delete_response = client.table("terminology_mappings").delete().eq("id", inserted_id).execute()
            result["delete"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    print("🔍 Paper Translator 데이터베이스 테스트")
    print("=" * 60)

    # Supabase 클라이언트 생성
    try:
        client = get_supabase_client()
        print("✅ Supabase 연결 성공")
        print(f"   URL: {os.getenv('SUPABASE_URL')[:50]}...")
    except Exception as e:
        print(f"❌ Supabase 연결 실패: {e}")
        sys.exit(1)

    # 테이블 목록
    tables = [
        "terminology_mappings",
        "translations",
        "translation_history",
        "term_changes"
    ]

    print("\n📋 테이블 존재 여부 확인:")
    print("-" * 60)

    all_exists = True
    for table in tables:
        result = test_table_exists(client, table)

        if result["exists"] and result["can_query"]:
            status = "✅ 존재"
            detail = f"(조회 가능, {result['row_count']}개 행)"
        elif result["exists"]:
            status = "⚠️ 존재하나 조회 불가"
            detail = f"({result['error']})"
            all_exists = False
        else:
            status = "❌ 없음"
            detail = ""
            all_exists = False

        print(f"  {table:25} {status} {detail}")

    # CRUD 테스트
    print("\n🧪 CRUD 테스트 (terminology_mappings):")
    print("-" * 60)

    crud_result = test_insert_and_delete(client)

    print(f"  INSERT: {'✅ 성공' if crud_result['insert'] else '❌ 실패'}")
    print(f"  SELECT: {'✅ 성공' if crud_result['select'] else '❌ 실패'}")
    print(f"  DELETE: {'✅ 성공' if crud_result['delete'] else '❌ 실패'}")

    if crud_result["error"]:
        print(f"  ⚠️ 에러: {crud_result['error']}")

    # 최종 결과
    print("\n" + "=" * 60)
    if all_exists and crud_result["insert"] and crud_result["select"] and crud_result["delete"]:
        print("🎉 모든 테이블이 정상적으로 생성되었습니다!")
        print("✅ 데이터베이스 설정 완료")
        return 0
    else:
        print("⚠️ 일부 테이블 또는 기능에 문제가 있습니다.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
