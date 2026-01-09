"""
데이터베이스 초기화 스크립트
Supabase PostgreSQL에 테이블 생성
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

load_dotenv()


def get_connection():
    """Supabase PostgreSQL 연결"""
    database_url = os.getenv("SUPABASE_DATABASE_URL")
    if not database_url:
        raise ValueError("SUPABASE_DATABASE_URL 환경변수가 설정되지 않았습니다.")

    return psycopg2.connect(database_url)


def run_migration(conn, migration_file: Path) -> bool:
    """마이그레이션 SQL 파일 실행"""
    print(f"📄 마이그레이션 실행: {migration_file.name}")

    with open(migration_file, "r", encoding="utf-8") as f:
        sql_content = f.read()

    try:
        with conn.cursor() as cur:
            cur.execute(sql_content)
        conn.commit()
        print(f"✅ {migration_file.name} 완료")
        return True
    except Exception as e:
        conn.rollback()
        print(f"❌ {migration_file.name} 실패: {e}")
        return False


def check_tables(conn) -> dict:
    """테이블 존재 여부 확인"""
    tables = ["terminology_mappings", "translations", "translation_history", "term_changes"]
    result = {}

    with conn.cursor() as cur:
        for table in tables:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                );
            """, (table,))
            result[table] = cur.fetchone()[0]

    return result


def main():
    print("🚀 Paper Translator 데이터베이스 초기화")
    print("=" * 50)

    try:
        conn = get_connection()
        print("✅ Supabase PostgreSQL 연결 성공")
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        sys.exit(1)

    # 마이그레이션 실행 전 테이블 확인
    print("\n📋 현재 테이블 상태:")
    tables_before = check_tables(conn)
    for table, exists in tables_before.items():
        status = "✅ 존재" if exists else "❌ 없음"
        print(f"  - {table}: {status}")

    # 마이그레이션 파일 실행
    migrations_dir = Path(__file__).parent / "migrations"
    migration_files = sorted(migrations_dir.glob("*.sql"))

    if not migration_files:
        print("\n⚠️ 마이그레이션 파일이 없습니다.")
        conn.close()
        return

    print(f"\n📦 {len(migration_files)}개의 마이그레이션 파일 발견")

    for migration_file in migration_files:
        success = run_migration(conn, migration_file)
        if not success:
            print("⚠️ 마이그레이션 중단")
            break

    # 마이그레이션 실행 후 테이블 확인
    print("\n📋 마이그레이션 후 테이블 상태:")
    tables_after = check_tables(conn)
    for table, exists in tables_after.items():
        status = "✅ 존재" if exists else "❌ 없음"
        print(f"  - {table}: {status}")

    conn.close()
    print("\n✅ 데이터베이스 초기화 완료!")


if __name__ == "__main__":
    main()
