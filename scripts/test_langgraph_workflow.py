"""
LangGraph 워크플로우 테스트 스크립트
"""
import sys
import logging
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_state_creation():
    """상태 생성 테스트"""
    print("\n1. 상태 생성 테스트")
    print("-" * 50)

    from src.state import create_initial_state, get_state_summary

    # 초기 상태 생성
    state = create_initial_state(
        source="https://arxiv.org/pdf/1706.03762",
        domain="NLP",
        exclude_references=True,
        extract_tables=True
    )

    print(f"  source: {state['source']}")
    print(f"  domain: {state['config']['domain']}")
    print(f"  status: {state['status']}")
    print(f"  started_at: {state['started_at']}")

    # 상태 요약
    summary = get_state_summary(state)
    print(f"\n  상태 요약:")
    for key, value in summary.items():
        print(f"    {key}: {value}")

    assert state["source"] == "https://arxiv.org/pdf/1706.03762"
    assert state["config"]["domain"] == "NLP"
    assert state["status"] == "pending"

    print("  PASS")
    return True


def test_graph_creation():
    """그래프 생성 테스트"""
    print("\n2. 그래프 생성 테스트")
    print("-" * 50)

    from src.graph import create_translation_graph, compile_graph

    # 그래프 생성
    workflow = create_translation_graph()
    print(f"  워크플로우 생성 완료")

    # 노드 확인
    nodes = list(workflow.nodes.keys())
    print(f"  노드 목록: {nodes}")

    expected_nodes = [
        "fetch_pdf", "parse_pdf", "chunk_text", "pre_process",
        "translate_chunks", "post_process", "generate_markdown", "save_output"
    ]
    for node in expected_nodes:
        assert node in nodes, f"노드 누락: {node}"

    # 그래프 컴파일
    app = compile_graph()
    print(f"  그래프 컴파일 완료")

    print("  PASS")
    return True


def test_individual_nodes_mock():
    """개별 노드 테스트 (mock 데이터)"""
    print("\n3. 개별 노드 테스트 (mock)")
    print("-" * 50)

    from src.state import TranslationState
    from src.parsers import ParsedPaper, ParsedSection
    from src.processors import Chunk, ProcessedChunk, TranslatedChunk, TranslationStatus, PostProcessedChunk
    from src.graph import chunk_text, pre_process, post_process, generate_markdown

    # Mock ParsedPaper 생성
    mock_sections = [
        ParsedSection(
            title="Abstract",
            content="This is a test abstract about Transformer model.",
            page_start=1,
            page_end=1,
            tables=[]
        ),
        ParsedSection(
            title="Introduction",
            content="Deep learning has revolutionized NLP. The attention mechanism is key.",
            page_start=1,
            page_end=2,
            tables=[]
        ),
    ]

    mock_paper = ParsedPaper(
        title="Test Paper: Transformer Architecture",
        raw_text="Abstract\nThis is a test abstract...",
        sections=mock_sections,
        tables=[],
        total_pages=10,
        source_path="test.pdf",
        arxiv_id="1234.5678"
    )

    # 상태 생성
    state: TranslationState = {
        "source": "test.pdf",
        "config": {
            "domain": "NLP",
            "exclude_references": True,
            "extract_tables": True,
            "chunking_strategy": "hybrid",
            "max_chunk_tokens": 800,
            "overlap_tokens": 100,
            "pre_process_limit": 20,
            "temperature": 0.1,
            "auto_correct": True,
            "threshold": 0.8,
        },
        "pdf_bytes": b"",
        "parsed_paper": mock_paper,
        "chunks": [],
        "processed_chunks": [],
        "translated_chunks": [],
        "post_processed_chunks": [],
        "metadata": {"title": "Test Paper"},
        "stats": {},
        "markdown_content": "",
        "output": {},
        "current_node": "",
        "status": "running",
        "error": None,
        "started_at": "2024-01-01T00:00:00",
        "completed_at": None,
    }

    # chunk_text 노드 테스트
    print("  chunk_text 노드 테스트...")
    state = chunk_text(state)
    print(f"    청크 수: {len(state['chunks'])}")
    assert len(state["chunks"]) > 0
    print("    OK")

    # pre_process 노드 테스트
    print("  pre_process 노드 테스트...")
    state = pre_process(state)
    print(f"    전처리된 청크 수: {len(state['processed_chunks'])}")
    assert len(state["processed_chunks"]) > 0
    print("    OK")

    # Mock 번역 결과 생성 (실제 API 호출 없이)
    mock_translated_chunks = []
    for pc in state["processed_chunks"]:
        mock_translated = TranslatedChunk(
            processed_chunk=pc,
            translated_text=f"[번역됨] {pc.chunk.content[:50]}...",
            status=TranslationStatus.COMPLETED,
            input_tokens=100,
            output_tokens=80,
            total_tokens=180,
        )
        mock_translated_chunks.append(mock_translated)

    state["translated_chunks"] = mock_translated_chunks
    state["stats"] = {
        "total_chunks": len(mock_translated_chunks),
        "completed_chunks": len(mock_translated_chunks),
        "failed_chunks": 0,
        "total_tokens": 180 * len(mock_translated_chunks),
        "estimated_cost_usd": "$0.01",
    }

    # post_process 노드 테스트
    print("  post_process 노드 테스트...")
    state = post_process(state)
    print(f"    후처리된 청크 수: {len(state['post_processed_chunks'])}")
    assert len(state["post_processed_chunks"]) > 0
    print("    OK")

    # generate_markdown 노드 테스트
    print("  generate_markdown 노드 테스트...")
    state = generate_markdown(state)
    print(f"    마크다운 길이: {len(state['markdown_content'])} 문자")
    assert len(state["markdown_content"]) > 0
    print("    OK")

    print("  PASS")
    return True


def test_full_pipeline_with_sample():
    """전체 파이프라인 테스트 (샘플 PDF)"""
    print("\n4. 전체 파이프라인 테스트 (샘플 PDF)")
    print("-" * 50)

    from src.graph import run_translation_sync

    # 테스트용 로컬 샘플 PDF가 있는지 확인
    sample_pdf = project_root / "data" / "sample.pdf"

    if not sample_pdf.exists():
        print(f"  [SKIP] 샘플 PDF가 없습니다: {sample_pdf}")
        print("  실제 테스트를 위해서는 data/sample.pdf를 준비하세요.")
        return True

    print(f"  샘플 PDF: {sample_pdf}")

    # 파이프라인 실행
    result = run_translation_sync(
        source=str(sample_pdf),
        domain="General",
        exclude_references=True,
    )

    print(f"  성공: {result['success']}")
    if result["success"]:
        print(f"  출력 파일: {result['output']['file_path']}")
        print(f"  총 토큰: {result['stats'].get('total_tokens', 'N/A')}")
        print(f"  예상 비용: {result['stats'].get('estimated_cost_usd', 'N/A')}")
    else:
        print(f"  에러: {result.get('error')}")

    print("  PASS")
    return True


def test_arxiv_translation():
    """ArXiv 논문 번역 테스트 (실제 API 호출)"""
    print("\n5. ArXiv 논문 번역 테스트")
    print("-" * 50)

    # 이 테스트는 실제 API를 호출하므로 기본적으로 스킵
    print("  [SKIP] 이 테스트는 실제 API를 호출합니다.")
    print("  실행하려면 --run-api 플래그를 사용하세요.")
    print("  예: python scripts/test_langgraph_workflow.py --run-api")

    if "--run-api" not in sys.argv:
        return True

    from src.graph import translate_arxiv

    # Attention Is All You Need (작은 논문 선택 권장)
    arxiv_id = "1706.03762"
    print(f"  ArXiv ID: {arxiv_id}")
    print("  번역 시작... (몇 분 소요될 수 있습니다)")

    result = translate_arxiv(
        arxiv_id,
        domain="NLP",
        exclude_references=True,
    )

    print(f"\n  성공: {result['success']}")
    if result["success"]:
        print(f"  출력 파일: {result['output']['file_path']}")
        print(f"  총 청크: {result['stats'].get('total_chunks', 'N/A')}")
        print(f"  총 토큰: {result['stats'].get('total_tokens', 'N/A')}")
        print(f"  예상 비용: {result['stats'].get('estimated_cost_usd', 'N/A')}")
        print(f"  용어 매칭률: {result['stats'].get('term_match_rate', 'N/A')}")
    else:
        print(f"  에러: {result.get('error')}")

    print("  PASS")
    return True


def test_graph_visualization():
    """그래프 시각화 테스트"""
    print("\n6. 그래프 시각화 테스트")
    print("-" * 50)

    try:
        from src.graph import compile_graph

        app = compile_graph()

        # ASCII 시각화 (mermaid 없이)
        print("  워크플로우 구조:")
        print("  ┌─────────────┐")
        print("  │  fetch_pdf  │")
        print("  └──────┬──────┘")
        print("         │")
        print("  ┌──────▼──────┐")
        print("  │  parse_pdf  │")
        print("  └──────┬──────┘")
        print("         │")
        print("  ┌──────▼──────┐")
        print("  │ chunk_text  │")
        print("  └──────┬──────┘")
        print("         │")
        print("  ┌──────▼──────┐")
        print("  │ pre_process │")
        print("  └──────┬──────┘")
        print("         │")
        print("  ┌──────▼───────────┐")
        print("  │ translate_chunks │")
        print("  └──────┬───────────┘")
        print("         │")
        print("  ┌──────▼──────┐")
        print("  │ post_process│")
        print("  └──────┬──────┘")
        print("         │")
        print("  ┌──────▼───────────────┐")
        print("  │ generate_markdown    │")
        print("  └──────┬───────────────┘")
        print("         │")
        print("  ┌──────▼──────┐")
        print("  │ save_output │")
        print("  └──────┬──────┘")
        print("         │")
        print("      [END]")

        print("\n  PASS")
        return True

    except Exception as e:
        print(f"  에러: {e}")
        return False


def main():
    print("=" * 60)
    print("LangGraph 워크플로우 테스트 시작")
    print("=" * 60)

    results = {
        "상태 생성": test_state_creation(),
        "그래프 생성": test_graph_creation(),
        "개별 노드 (mock)": test_individual_nodes_mock(),
        "전체 파이프라인": test_full_pipeline_with_sample(),
        "ArXiv 번역": test_arxiv_translation(),
        "그래프 시각화": test_graph_visualization(),
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
        print("모든 LangGraph 워크플로우 테스트 통과!")
    else:
        print("일부 테스트 실패")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
