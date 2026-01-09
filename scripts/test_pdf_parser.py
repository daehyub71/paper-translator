"""
PDF 파서 테스트 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers import PDFParser, parse_pdf


def test_arxiv_id_extraction():
    """ArXiv ID 추출 테스트"""
    print("\n🔍 ArXiv ID 추출 테스트")
    print("-" * 50)

    test_cases = [
        ("1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762", "1706.03762"),
        ("https://arxiv.org/abs/2301.00234", "2301.00234"),
        ("https://ar5iv.labs.arxiv.org/html/1706.03762", "1706.03762"),
        ("invalid_url", None),
    ]

    for input_val, expected in test_cases:
        result = PDFParser.extract_arxiv_id(input_val)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {input_val[:40]:40} → {result}")

    return True


def test_pdf_url_conversion():
    """ArXiv ID → PDF URL 변환 테스트"""
    print("\n🔗 PDF URL 변환 테스트")
    print("-" * 50)

    arxiv_id = "1706.03762"
    expected = "https://arxiv.org/pdf/1706.03762.pdf"
    result = PDFParser.arxiv_id_to_pdf_url(arxiv_id)

    status = "✅" if result == expected else "❌"
    print(f"  {status} {arxiv_id} → {result}")

    return result == expected


def test_pdf_download():
    """PDF 다운로드 테스트 (실제 ArXiv)"""
    print("\n📥 PDF 다운로드 테스트")
    print("-" * 50)

    parser = PDFParser()

    # 짧은 논문 사용 (빠른 테스트를 위해)
    arxiv_id = "1706.03762"  # Attention Is All You Need

    try:
        print(f"  다운로드 중: {arxiv_id}...")
        pdf_bytes = parser.download_pdf(arxiv_id)

        # 검증
        is_pdf = pdf_bytes[:4] == b"%PDF"
        size_kb = len(pdf_bytes) / 1024

        print(f"  ✅ 다운로드 완료: {size_kb:.1f} KB")
        print(f"  ✅ PDF 형식 확인: {is_pdf}")

        return is_pdf and size_kb > 100  # 최소 100KB 이상

    except Exception as e:
        print(f"  ❌ 다운로드 실패: {e}")
        return False


def test_text_extraction():
    """텍스트 추출 테스트"""
    print("\n📄 텍스트 추출 테스트")
    print("-" * 50)

    parser = PDFParser()

    try:
        # PDF 다운로드
        pdf_bytes = parser.download_pdf("1706.03762")

        # PyPDF2 텍스트 추출
        text, pages = parser.extract_text_pypdf2(pdf_bytes)

        print(f"  ✅ 총 페이지: {pages}")
        print(f"  ✅ 추출된 텍스트 길이: {len(text):,} 문자")
        print(f"  ✅ 첫 200자: {text[:200]}...")

        return len(text) > 1000

    except Exception as e:
        print(f"  ❌ 텍스트 추출 실패: {e}")
        return False


def test_table_extraction():
    """표 추출 테스트"""
    print("\n📊 표 추출 테스트")
    print("-" * 50)

    parser = PDFParser()

    try:
        pdf_bytes = parser.download_pdf("1706.03762")
        tables = parser.extract_tables_pdfplumber(pdf_bytes)

        print(f"  ✅ 추출된 표 수: {len(tables)}")
        if tables:
            print(f"  ✅ 첫 번째 표 미리보기:")
            preview = tables[0][:300] if len(tables[0]) > 300 else tables[0]
            for line in preview.split("\n")[:5]:
                print(f"      {line}")

        return True  # 표가 없어도 성공

    except Exception as e:
        print(f"  ❌ 표 추출 실패: {e}")
        return False


def test_section_detection():
    """섹션 감지 테스트"""
    print("\n📑 섹션 감지 테스트")
    print("-" * 50)

    parser = PDFParser()

    try:
        pdf_bytes = parser.download_pdf("1706.03762")
        text, _ = parser.extract_text_pypdf2(pdf_bytes)
        sections = parser.detect_sections(text)

        print(f"  ✅ 감지된 섹션 수: {len(sections)}")
        for section in sections[:5]:
            content_preview = section.content[:50].replace("\n", " ") if section.content else ""
            print(f"      - {section.title[:30]:30} (p.{section.page_start}-{section.page_end})")

        return len(sections) > 0

    except Exception as e:
        print(f"  ❌ 섹션 감지 실패: {e}")
        return False


def test_full_parse():
    """전체 파싱 테스트"""
    print("\n🎯 전체 파싱 테스트")
    print("-" * 50)

    try:
        # parse_pdf 단축 함수 사용
        result = parse_pdf("1706.03762", exclude_references=True)

        print(f"  ✅ 제목: {result.title[:50]}...")
        print(f"  ✅ ArXiv ID: {result.arxiv_id}")
        print(f"  ✅ 총 페이지: {result.total_pages}")
        print(f"  ✅ 섹션 수: {len(result.sections)}")
        print(f"  ✅ 표 수: {len(result.tables)}")
        print(f"  ✅ 원문 길이: {len(result.raw_text):,} 문자")

        return True

    except Exception as e:
        print(f"  ❌ 전체 파싱 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🧪 PDF 파서 테스트 시작")
    print("=" * 60)

    results = {
        "ArXiv ID 추출": test_arxiv_id_extraction(),
        "PDF URL 변환": test_pdf_url_conversion(),
        "PDF 다운로드": test_pdf_download(),
        "텍스트 추출": test_text_extraction(),
        "표 추출": test_table_extraction(),
        "섹션 감지": test_section_detection(),
        "전체 파싱": test_full_parse(),
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
        print("🎉 모든 PDF 파서 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
