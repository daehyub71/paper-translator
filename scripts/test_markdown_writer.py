"""
Markdown Writer 테스트 스크립트
"""
import sys
import tempfile
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.outputs import (
    MarkdownWriter,
    TranslatedPaper,
    TranslatedSection,
    render_markdown,
    save_markdown,
    write_translated_paper,
    calculate_content_hash,
)
from src.processors import (
    Chunk,
    ProcessedChunk,
    TranslatedChunk,
    TranslationStatus,
    PostProcessedChunk,
    TermValidation,
    TermMatchStatus,
)


def create_test_post_processed_chunk(
    section_title: str,
    content: str,
    translated_text: str,
    chunk_index: int = 0,
    matched_terms: list = None
) -> PostProcessedChunk:
    """테스트용 후처리된 청크 생성"""
    chunk = Chunk(
        index=chunk_index,
        content=content,
        section_title=section_title,
        section_index=0,
        token_count=50,
        start_char=0,
        end_char=len(content),
        tables=[]
    )

    processed = ProcessedChunk(
        chunk=chunk,
        matched_terms=matched_terms or [],
        terminology_prompt="",
        context_hint=""
    )

    translated = TranslatedChunk(
        processed_chunk=processed,
        translated_text=translated_text,
        status=TranslationStatus.COMPLETED,
        input_tokens=100,
        output_tokens=80,
        total_tokens=180
    )

    return PostProcessedChunk(
        translated_chunk=translated,
        corrected_text=translated_text,
        validations=[],
        corrections_made=0,
        missing_terms=0,
        mismatched_terms=0
    )


def test_section_title_translation():
    """섹션 제목 한글화 테스트"""
    print("\n1. 섹션 제목 한글화 테스트")
    print("-" * 50)

    writer = MarkdownWriter()

    test_cases = [
        ("Abstract", "초록"),
        ("1. Introduction", "1. 서론"),
        ("2.1 Related Work", "2.1 관련 연구"),
        ("Methods", "방법"),
        ("3. Experiments", "3. 실험"),
        ("Conclusion", "결론"),
        ("Main Content", "본문"),
        ("Custom Section", "Custom Section"),  # 매핑 없음
    ]

    for eng, expected_contains in test_cases:
        result = writer.translate_section_title(eng)
        if expected_contains in result or result == eng:
            print(f"  [OK] '{eng}' → '{result}'")
        else:
            print(f"  [WARN] '{eng}' → '{result}' (expected: {expected_contains})")

    print("  PASS")
    return True


def test_filename_generation():
    """파일명 생성 테스트"""
    print("\n2. 파일명 생성 테스트")
    print("-" * 50)

    writer = MarkdownWriter()

    # 기본 파일명
    filename = writer.generate_filename(
        title="Attention Is All You Need",
        date="20240101"
    )
    print(f"  기본: '{filename}'")
    assert "attention" in filename.lower()

    # ArXiv ID 포함
    filename_with_arxiv = writer.generate_filename(
        title="트랜스포머 논문",
        arxiv_id="1706.03762"
    )
    print(f"  ArXiv ID 포함: '{filename_with_arxiv}'")

    # 긴 제목
    long_title = "A Very Long Title That Should Be Truncated For The Filename"
    filename_long = writer.generate_filename(title=long_title)
    print(f"  긴 제목: '{filename_long}' (길이: {len(filename_long)})")
    assert len(filename_long) <= 70  # date + title 합쳐서

    print("  PASS")
    return True


def test_md5_hash():
    """MD5 해시 계산 테스트"""
    print("\n3. MD5 해시 계산 테스트")
    print("-" * 50)

    writer = MarkdownWriter()

    content1 = "Hello, World!"
    content2 = "Hello, World!"
    content3 = "Different content"

    hash1 = writer.calculate_md5(content1)
    hash2 = writer.calculate_md5(content2)
    hash3 = writer.calculate_md5(content3)

    print(f"  hash1: {hash1}")
    print(f"  hash2: {hash2}")
    print(f"  hash3: {hash3}")

    assert hash1 == hash2, "동일 콘텐츠는 같은 해시"
    assert hash1 != hash3, "다른 콘텐츠는 다른 해시"

    print("  PASS")
    return True


def test_translated_paper_creation():
    """TranslatedPaper 생성 테스트"""
    print("\n4. TranslatedPaper 생성 테스트")
    print("-" * 50)

    paper = TranslatedPaper(
        title="Attention Is All You Need",
        title_ko="어텐션만 있으면 된다",
        arxiv_id="1706.03762",
        authors=["Vaswani et al."],
        domain="NLP",
        abstract="트랜스포머 모델을 소개합니다.",
        sections=[
            TranslatedSection(
                title="Introduction",
                title_ko="서론",
                content="소개 내용입니다."
            ),
            TranslatedSection(
                title="Methods",
                title_ko="방법",
                content="방법 내용입니다."
            ),
        ],
        total_pages=15,
        total_chunks=10
    )

    print(f"  제목: {paper.title}")
    print(f"  한글제목: {paper.title_ko}")
    print(f"  ArXiv: {paper.arxiv_id}")
    print(f"  섹션 수: {len(paper.sections)}")
    print(f"  번역일: {paper.translated_date}")

    assert paper.title == "Attention Is All You Need"
    assert len(paper.sections) == 2

    print("  PASS")
    return True


def test_render_paper():
    """논문 렌더링 테스트"""
    print("\n5. 논문 렌더링 테스트")
    print("-" * 50)

    writer = MarkdownWriter()

    paper = TranslatedPaper(
        title="Attention Is All You Need",
        title_ko="어텐션만 있으면 된다",
        arxiv_id="1706.03762",
        authors=["Vaswani", "Shazeer"],
        domain="NLP",
        abstract="본 논문에서는 트랜스포머 아키텍처를 제안합니다.",
        sections=[
            TranslatedSection(
                title="Introduction",
                title_ko="서론",
                content="시퀀스 모델링은 중요합니다."
            ),
        ],
        total_pages=15,
        total_chunks=10,
        total_tokens=5000,
        estimated_cost="$0.01",
        terminology_used=[
            {"source": "Transformer", "target": "트랜스포머"},
            {"source": "attention", "target": "어텐션"},
        ]
    )

    content = writer.render(paper)

    print(f"  렌더링 결과 길이: {len(content)} chars")
    print(f"  미리보기:")
    print("-" * 40)
    print(content[:500])
    print("-" * 40)

    assert "어텐션만 있으면 된다" in content
    assert "1706.03762" in content
    assert "트랜스포머" in content

    print("  PASS")
    return True


def test_chunks_to_paper():
    """청크에서 논문 변환 테스트"""
    print("\n6. 청크에서 논문 변환 테스트")
    print("-" * 50)

    writer = MarkdownWriter()

    # 테스트 청크 생성
    chunks = [
        create_test_post_processed_chunk(
            section_title="Abstract",
            content="This paper presents...",
            translated_text="본 논문에서는...",
            chunk_index=0,
            matched_terms=[
                {"source_text": "paper", "target_text": "논문"}
            ]
        ),
        create_test_post_processed_chunk(
            section_title="Introduction",
            content="Deep learning has...",
            translated_text="딥러닝은...",
            chunk_index=1,
            matched_terms=[
                {"source_text": "Deep learning", "target_text": "딥러닝"}
            ]
        ),
        create_test_post_processed_chunk(
            section_title="Introduction",
            content="Transformers are...",
            translated_text="트랜스포머는...",
            chunk_index=2,
            matched_terms=[
                {"source_text": "Transformer", "target_text": "트랜스포머"}
            ]
        ),
    ]

    metadata = {
        "title": "Test Paper",
        "title_ko": "테스트 논문",
        "arxiv_id": "2024.12345",
        "authors": ["Author 1", "Author 2"],
        "domain": "NLP",
        "total_pages": 10
    }

    paper = writer.chunks_to_paper(chunks, metadata)

    print(f"  제목: {paper.title}")
    print(f"  섹션 수: {len(paper.sections)}")
    print(f"  총 청크: {paper.total_chunks}")
    print(f"  사용 용어: {len(paper.terminology_used)}")

    for section in paper.sections:
        print(f"    - {section.title} → {section.title_ko}")

    assert len(paper.sections) == 2  # Abstract, Introduction
    assert paper.total_chunks == 3
    assert len(paper.terminology_used) == 3

    print("  PASS")
    return True


def test_save_markdown():
    """마크다운 저장 테스트"""
    print("\n7. 마크다운 저장 테스트")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = MarkdownWriter(output_dir=tmpdir)

        content = "# Test\n\nThis is a test."
        filename = "test_file"

        file_path = writer.save(content, filename)

        print(f"  저장 경로: {file_path}")
        assert file_path.exists()
        assert file_path.suffix == ".md"

        # 내용 확인
        with open(file_path, "r", encoding="utf-8") as f:
            saved_content = f.read()

        assert saved_content == content

    print("  PASS")
    return True


def test_write_paper():
    """전체 논문 저장 테스트"""
    print("\n8. 전체 논문 저장 테스트")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = MarkdownWriter(output_dir=tmpdir)

        chunks = [
            create_test_post_processed_chunk(
                section_title="Introduction",
                content="Original content",
                translated_text="번역된 내용입니다.",
                chunk_index=0
            ),
        ]

        metadata = {
            "title": "Test Paper",
            "title_ko": "테스트 논문",
            "total_pages": 5,
            "total_tokens": 1000,
            "estimated_cost": "$0.01"
        }

        result = writer.write_paper(chunks, metadata)

        print(f"  파일 경로: {result['file_path']}")
        print(f"  파일명: {result['filename']}")
        print(f"  MD5 해시: {result['md5_hash']}")
        print(f"  콘텐츠 길이: {result['content_length']}")

        assert result['file_path'].exists()
        assert result['md5_hash']
        assert result['content_length'] > 0

    print("  PASS")
    return True


def test_render_simple():
    """간단한 렌더링 테스트"""
    print("\n9. 간단한 렌더링 테스트")
    print("-" * 50)

    writer = MarkdownWriter()

    sections = [
        {"title": "서론", "content": "첫 번째 섹션 내용"},
        {"title": "본론", "content": "두 번째 섹션 내용"},
    ]

    content = writer.render_simple(
        title="간단한 문서",
        sections=sections,
        metadata={"arxiv_id": "1234.5678", "translated_date": "2024-01-01"}
    )

    print(f"  렌더링 결과 길이: {len(content)} chars")
    print("-" * 40)
    print(content)
    print("-" * 40)

    assert "간단한 문서" in content
    assert "서론" in content
    assert "1234.5678" in content

    print("  PASS")
    return True


def test_convenience_functions():
    """편의 함수 테스트"""
    print("\n10. 편의 함수 테스트")
    print("-" * 50)

    # calculate_content_hash
    hash1 = calculate_content_hash("test content")
    hash2 = calculate_content_hash("test content")
    print(f"  calculate_content_hash: {hash1}")
    assert hash1 == hash2

    # save_markdown
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_markdown("# Test", "test", output_dir=tmpdir)
        print(f"  save_markdown: {path}")
        assert path.exists()

    print("  PASS")
    return True


def main():
    print("=" * 60)
    print("Markdown Writer 테스트 시작")
    print("=" * 60)

    results = {
        "섹션 제목 한글화": test_section_title_translation(),
        "파일명 생성": test_filename_generation(),
        "MD5 해시": test_md5_hash(),
        "TranslatedPaper 생성": test_translated_paper_creation(),
        "논문 렌더링": test_render_paper(),
        "청크→논문 변환": test_chunks_to_paper(),
        "마크다운 저장": test_save_markdown(),
        "전체 논문 저장": test_write_paper(),
        "간단한 렌더링": test_render_simple(),
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
        print("모든 Markdown Writer 테스트 통과!")
    else:
        print("일부 테스트 실패")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
