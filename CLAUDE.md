# CLAUDE.md

Paper Translator 프로젝트 개발 가이드

## 프로젝트 개요

Paper Translator는 ArXiv 논문을 한국어로 번역하는 AI 기반 번역 시스템입니다. LangGraph를 사용한 8-노드 워크플로우로 구성되어 있으며, Supabase를 데이터베이스로 사용합니다.

## 빠른 시작

```bash
# 가상환경 활성화
source venv/bin/activate

# 테스트 실행
pytest tests/ -v

# CLI 실행
python -m src.main translate --help
```

## 프로젝트 구조

```
paper-translator/
├── src/                        # 소스 코드
│   ├── api/                    # 외부 연동 인터페이스
│   │   ├── interface.py        # TranslationRequest/Response 스키마
│   │   └── insightbot.py       # InsightBot LangGraph 연동
│   ├── db/                     # 데이터베이스 레이어
│   │   ├── supabase_client.py  # Supabase 연결 관리
│   │   └── repositories.py     # Repository 패턴 CRUD
│   ├── feedback/               # 피드백 루프
│   │   ├── diff_analyzer.py    # 마크다운 변경 감지
│   │   └── sync_manager.py     # DB 동기화 관리
│   ├── outputs/                # 출력 생성
│   │   └── markdown_writer.py  # Jinja2 마크다운 렌더링
│   ├── parsers/                # 입력 파싱
│   │   └── pdf_parser.py       # PDF 텍스트/표 추출
│   ├── processors/             # 번역 파이프라인
│   │   ├── chunker.py          # 하이브리드 청킹
│   │   ├── pre_processor.py    # 용어 프롬프트 주입
│   │   ├── translator.py       # LLM 번역
│   │   └── post_processor.py   # 용어 검증/교정
│   ├── utils/                  # 유틸리티
│   │   ├── config.py           # 설정 로더
│   │   └── llm_client.py       # OpenAI 클라이언트
│   ├── graph.py                # LangGraph 워크플로우 정의
│   ├── state.py                # TranslationState TypedDict
│   └── main.py                 # Typer CLI
├── tests/                      # 테스트
├── scripts/                    # 스크립트
├── templates/                  # Jinja2 템플릿
├── config/                     # 설정 파일
└── docs/                       # 문서
```

## 핵심 모듈

### LangGraph 워크플로우 (src/graph.py)

8개 노드로 구성된 번역 파이프라인:

1. `fetch_pdf` - PDF 다운로드/로드
2. `parse_pdf` - 텍스트/표/수식 추출
3. `chunk_text` - 하이브리드 청킹
4. `pre_process` - 용어 프롬프트 주입
5. `translate_chunks` - LLM 번역
6. `post_process` - 용어 검증/교정
7. `generate_markdown` - 마크다운 생성
8. `save_output` - 파일 저장/DB 기록

```python
from src.graph import create_translation_graph, run_translation

# 그래프 생성
graph = create_translation_graph()

# 번역 실행
result = run_translation(
    source="https://arxiv.org/pdf/1706.03762.pdf",
    domain="NLP"
)
```

### 상태 관리 (src/state.py)

`TranslationState` TypedDict로 노드 간 데이터 전달:

```python
from src.state import TranslationState, create_initial_state

state = create_initial_state(
    source="1706.03762",
    source_type="arxiv_id",
    domain="NLP"
)
```

### Repository 패턴 (src/db/repositories.py)

4개의 Repository 클래스:

- `TerminologyRepository` - 전문용어 CRUD
- `TranslationRepository` - 번역 기록 CRUD
- `TranslationHistoryRepository` - 청크 히스토리 CRUD
- `TermChangeRepository` - 용어 변경 로그 CRUD

```python
from src.db.repositories import TerminologyRepository

repo = TerminologyRepository()
terms = repo.get_by_domain("NLP")
```

### API 인터페이스 (src/api/)

외부 시스템 연동용 인터페이스:

```python
from src.api import translate, TranslationRequest, TranslationResponse

# 간단한 번역
result = translate(source="1706.03762", domain="NLP")

# InsightBot 연동
from src.api.insightbot import TranslationNodeWrapper
wrapper = TranslationNodeWrapper(auto_confirm=True)
```

## 개발 명령어

### 테스트

```bash
# 전체 테스트
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html

# 특정 테스트
pytest tests/test_translator.py -v

# 특정 테스트 함수
pytest tests/test_translator.py::TestTranslator::test_translate_chunk -v
```

### CLI

```bash
# 번역
python -m src.main translate --url "https://arxiv.org/pdf/1706.03762.pdf"
python -m src.main translate --arxiv-id "1706.03762" --domain NLP
python -m src.main translate --file "./paper.pdf"

# 용어 관리
python -m src.main terms list --domain NLP
python -m src.main terms add --source "attention" --target "어텐션"
python -m src.main terms export --output ./terms.json

# 피드백 동기화
python -m src.main sync --file ./translations/paper.md
python -m src.main sync --all
```

### 스크립트

```bash
# 초기 용어 데이터 삽입
python scripts/seed_terminology.py
```

## 설정

### 환경 변수 (.env)

```bash
# 필수
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# 선택
OPENAI_MODEL=gpt-4o-mini
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_DATABASE_URL=postgresql://...
UPSTASH_URL=https://xxxxx.upstash.io
UPSTASH_TOKEN=AXbUAAIncDI...
```

### 설정 파일 (config/settings.yaml)

```yaml
translation:
  max_chunk_tokens: 1000
  overlap_tokens: 100
  model: gpt-4o-mini

terminology:
  pre_process_limit: 50
  post_process_threshold: 0.8

output:
  directory: ./translations
  filename_format: "{date}_{title}"
```

## 데이터베이스

### Supabase 테이블

1. `terminology_mappings` - 전문용어 매핑
2. `translations` - 번역 기록
3. `translation_history` - 청크별 히스토리
4. `term_changes` - 용어 변경 로그

### 마이그레이션

Supabase Dashboard에서 SQL 실행 또는:

```bash
# 스키마는 docs/DEVELOPMENT_PLAN.md 참조
```

## 트러블슈팅

### "OPENAI_API_KEY not found"

```bash
# .env 파일 확인
cat .env | grep OPENAI

# 환경 변수 로드 확인
python -c "from src.utils.config import get_config; print(get_config())"
```

### "Supabase connection failed"

```bash
# Supabase URL/KEY 확인
python -c "from src.db.supabase_client import get_client; print(get_client())"
```

### "PDF download failed"

```bash
# ArXiv URL 접근 가능 여부 확인
curl -I https://arxiv.org/pdf/1706.03762.pdf

# requests 버전 확인
pip show requests
```

### 테스트 실패

```bash
# 의존성 재설치
pip install -r requirements.txt

# pytest 캐시 삭제
rm -rf .pytest_cache

# 특정 테스트 디버깅
pytest tests/test_translator.py -v -s --tb=long
```

## 코드 스타일

- Python 3.9+ 호환
- Type hints 사용 권장
- docstring: Google 스타일
- 한글 주석 허용

## 주요 의존성

- `langgraph>=0.2.0` - 워크플로우 오케스트레이션
- `langchain-openai>=0.2.0` - OpenAI 연동
- `pypdf2>=3.0.0` - PDF 텍스트 추출
- `pdfplumber>=0.10.0` - PDF 표 추출
- `supabase>=2.0.0` - 데이터베이스
- `typer>=0.12.0` - CLI 프레임워크
- `rich>=13.0.0` - CLI 출력 포맷팅
- `jinja2>=3.1.0` - 마크다운 템플릿
- `tiktoken>=0.5.0` - 토큰 카운팅

## 참고 문서

- [DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) - 개발 계획서
- [TASK_LIST.md](docs/TASK_LIST.md) - 작업 목록
- [README.md](README.md) - 사용자 가이드
