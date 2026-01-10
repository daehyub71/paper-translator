# Paper Translator

ArXiv 논문을 한국어로 번역하는 AI 기반 번역 시스템

## 개요

Paper Translator는 ArXiv 등의 AI 연구 논문 PDF를 한국어로 번역하고, 전문용어 매핑을 통해 일관된 고품질 번역을 제공하는 자동화 시스템입니다.

### 주요 기능

| 기능 | 설명 |
|-----|------|
| 논문 검색 | ArXiv API와 Semantic Scholar로 논문 검색 (인용수 기반 필터링) |
| PDF 전체 번역 | ArXiv 논문 PDF를 다운로드하여 전체 내용 번역 |
| 전문용어 매핑 | 단어/문장 단위 용어 DB 관리, 도메인별 분류 |
| 하이브리드 용어 적용 | Pre-translation(프롬프트 주입) + Post-translation(후처리 검증) |
| 피드백 루프 | 마크다운 수정 시 변경 감지 → DB 자동 업데이트 |
| 번역 히스토리 | 번역 이력 저장, 일괄 수정 지원 |
| 다중 트리거 | CLI 독립 실행 + InsightBot 연동 |

## 기술 스택

| 카테고리 | 기술 |
|---------|------|
| Language | Python 3.9+ |
| LLM | OpenAI GPT-4o-mini |
| Orchestration | LangGraph |
| PDF Parsing | PyPDF2 / pdfplumber |
| Paper Discovery | arxiv (ArXiv API) / requests (Semantic Scholar API) |
| Database | Supabase (PostgreSQL) |
| Template | Jinja2 (Markdown 생성) |
| CLI | Typer + Rich |

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/your-repo/paper-translator.git
cd paper-translator
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 필요한 값을 설정합니다:

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_DATABASE_URL=postgresql://postgres.xxxxx:password@aws-0-ap-northeast-2.pooler.supabase.com:6543/postgres
```

### 5. 초기 용어 데이터 삽입 (선택)

```bash
python scripts/seed_terminology.py
```

## 사용법

### CLI 명령어

#### 논문 검색

ArXiv 또는 Semantic Scholar에서 논문을 검색합니다:

```bash
# ArXiv에서 NLP 트렌딩 논문 검색
python -m src.main discover --source arxiv --domain NLP --trending

# ArXiv에서 키워드로 검색
python -m src.main discover --source arxiv --query "transformer" --domain NLP

# Semantic Scholar에서 고인용 논문 검색
python -m src.main discover --source semantic-scholar --domain ML --highly-cited

# 최소 인용수 필터링
python -m src.main discover --source semantic-scholar --query "BERT" --min-citations 100

# 상세 정보 표시
python -m src.main discover --source arxiv --domain CV --trending --verbose

# 페이지네이션 - 2페이지 조회 (페이지당 10개)
python -m src.main discover --source arxiv --domain NLP --trending --page 2

# 페이지당 개수 지정 (5개씩, 3페이지)
python -m src.main discover --source arxiv --domain NLP --trending --per-page 5 --page 3
```

**검색 옵션:**

| 옵션 | 설명 |
|-----|------|
| `--source` | 검색 소스: `arxiv`, `semantic-scholar` (또는 `s2`) |
| `--query` | 검색어 |
| `--domain` | 도메인 필터: NLP, CV, ML, RL, Speech, General |
| `--page`, `-p` | 페이지 번호 (기본: 1) |
| `--per-page` | 페이지당 결과 수 (기본: 10) |
| `--max-results`, `-n` | 최대 전체 결과 수 (기본: 100) |
| `--min-citations` | 최소 인용수 (Semantic Scholar 전용) |
| `--year-from` | 시작 연도 필터 |
| `--trending` | 트렌딩/최신 논문 조회 |
| `--highly-cited` | 고인용 논문 조회 (Semantic Scholar 전용) |
| `--verbose` | 상세 논문 정보 표시 |
| `--json` | JSON 형식 출력 |

**출력 기능:**
- ArXiv와 Semantic Scholar 모두 **ArXiv ID** 컬럼 표시 (번역 시 편리한 참조용)
- 페이지 정보 하단 표시 (예: "페이지 1/5 (총 50개 결과)")
- 다음 페이지가 있을 경우 다음 페이지 명령어 안내

#### 논문 번역

```bash
# URL로 번역
python -m src.main translate --url "https://arxiv.org/pdf/1706.03762.pdf"

# ArXiv ID로 번역
python -m src.main translate --arxiv-id "1706.03762"

# 로컬 PDF 파일 번역
python -m src.main translate --file "./paper.pdf"

# 도메인 지정 (NLP, CV, RL 등)
python -m src.main translate --url "https://arxiv.org/pdf/1706.03762.pdf" --domain NLP
```

#### 용어 관리

```bash
# 용어 목록 조회
python -m src.main terms list
python -m src.main terms list --domain NLP

# 용어 추가
python -m src.main terms add --source "transformer" --target "트랜스포머" --domain NLP

# 용어 수정
python -m src.main terms update <term-id> --target "새로운 번역"

# 용어 삭제
python -m src.main terms delete <term-id> --yes

# 용어 내보내기
python -m src.main terms export --output ./terms.json

# 용어 가져오기
python -m src.main terms import --file ./terms.json
```

#### 피드백 동기화

번역된 마크다운 파일을 수정한 후 용어 DB에 반영:

```bash
# 단일 파일 동기화
python -m src.main sync --file ./translations/paper.md

# 변경된 모든 파일 동기화
python -m src.main sync --all
```

### Python API

```python
from src.api import translate, TranslationRequest

# 간단한 번역
result = translate(
    source="https://arxiv.org/pdf/1706.03762.pdf",
    domain="NLP"
)
print(result.output_path)

# 상세 옵션으로 번역
request = TranslationRequest(
    source="1706.03762",
    domain="NLP",
    output_dir="./my_translations"
)
response = translate(request)
```

### InsightBot 연동

```python
from src.api.insightbot import (
    TranslationNodeWrapper,
    create_translation_subgraph
)

# 노드 래퍼 사용
wrapper = TranslationNodeWrapper(auto_confirm=True)
result = await wrapper.translate_paper_node(state)

# 서브그래프로 통합
subgraph = create_translation_subgraph()
```

## 프로젝트 구조

```
paper-translator/
├── src/
│   ├── api/                    # 외부 연동 인터페이스
│   │   ├── interface.py        # TranslationRequest/Response
│   │   └── insightbot.py       # InsightBot 연동
│   ├── collectors/             # 논문 검색
│   │   ├── arxiv_collector.py  # ArXiv API 클라이언트
│   │   └── semantic_scholar_collector.py  # Semantic Scholar API
│   ├── db/                     # 데이터베이스
│   │   ├── supabase_client.py  # Supabase 클라이언트
│   │   └── repositories.py     # CRUD 레포지토리
│   ├── feedback/               # 피드백 루프
│   │   ├── diff_analyzer.py    # 변경 감지
│   │   └── sync_manager.py     # DB 동기화
│   ├── outputs/                # 출력
│   │   └── markdown_writer.py  # 마크다운 생성
│   ├── parsers/                # PDF 파싱
│   │   └── pdf_parser.py       # PDF 추출
│   ├── processors/             # 번역 처리
│   │   ├── chunker.py          # 텍스트 청킹
│   │   ├── pre_processor.py    # 전처리 (용어 주입)
│   │   ├── translator.py       # LLM 번역
│   │   └── post_processor.py   # 후처리 (용어 검증)
│   ├── utils/                  # 유틸리티
│   │   ├── config.py           # 설정 관리
│   │   └── llm_client.py       # LLM 클라이언트
│   ├── graph.py                # LangGraph 워크플로우
│   ├── state.py                # 상태 정의
│   └── main.py                 # CLI 진입점
├── scripts/                    # 스크립트
│   └── seed_terminology.py     # 초기 용어 삽입
├── templates/                  # 템플릿
│   └── paper_template.md.j2    # 마크다운 템플릿
├── tests/                      # 테스트
│   ├── test_pdf_parser.py
│   ├── test_chunker.py
│   ├── test_translator.py
│   ├── test_diff_analyzer.py
│   ├── test_repositories.py
│   └── test_integration.py
├── docs/                       # 문서
│   ├── DEVELOPMENT_PLAN.md
│   └── TASK_LIST.md
├── config/
│   └── settings.yaml           # 설정 파일
├── requirements.txt
├── .env.example
└── README.md
```

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   [CLI 독립 실행]                      [InsightBot 연동]                  │
│   $ python -m src.main translate       InsightBot → Paper Translator    │
│     --url "https://arxiv.org/..."      (선택한 논문 자동 전달)            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐  │
│  │   fetch    │ →  │   parse    │ →  │   chunk    │ →  │    pre     │  │
│  │   _pdf     │    │   _pdf     │    │   _text    │    │  _process  │  │
│  │            │    │            │    │            │    │            │  │
│  │ PDF 다운로드│    │텍스트 추출 │    │하이브리드   │    │용어 프롬프트│  │
│  │            │    │표/수식 처리│    │청킹        │    │주입        │  │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘  │
│                                                               ↓         │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐  │
│  │   save     │ ←  │  generate  │ ←  │   post     │ ←  │ translate  │  │
│  │  _output   │    │ _markdown  │    │ _process   │    │  _chunks   │  │
│  │            │    │            │    │            │    │            │  │
│  │ 파일 저장  │    │ MD 템플릿  │    │용어 검증   │    │ LLM 번역   │  │
│  │ DB 기록    │    │ 적용       │    │후처리 교정 │    │ (청크별)   │  │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT LAYER                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ./translations/                                                        │
│     └── 2024-01-08_Attention_Is_All_You_Need.md                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html

# 특정 테스트 파일
pytest tests/test_translator.py -v
```

## 환경 변수

| 변수명 | 필수 | 설명 |
|-------|------|------|
| OPENAI_API_KEY | O | OpenAI API 키 |
| OPENAI_MODEL | - | 사용할 모델 (기본: gpt-4o-mini) |
| SUPABASE_URL | O | Supabase 프로젝트 URL |
| SUPABASE_KEY | O | Supabase anon key |
| SUPABASE_SERVICE_ROLE_KEY | - | Supabase service role key |
| SUPABASE_DATABASE_URL | - | PostgreSQL 직접 연결 URL |
| SEMANTIC_SCHOLAR_API_KEY | - | Semantic Scholar API 키 (요청 제한 완화) |
| UPSTASH_URL | - | Redis 캐시 URL (선택) |
| UPSTASH_TOKEN | - | Redis 인증 토큰 (선택) |

> **참고:** Semantic Scholar API 키는 https://www.semanticscholar.org/product/api#api-key-form 에서 무료로 발급받을 수 있습니다.

## 데이터베이스 스키마

### terminology_mappings (전문용어 매핑)

| 컬럼 | 타입 | 설명 |
|-----|------|------|
| id | uuid | 기본 키 |
| source_text | text | 원문 용어 |
| target_text | text | 번역 용어 |
| domain | text | 도메인 (NLP, CV, RL 등) |
| confidence | float | 신뢰도 |
| created_at | timestamp | 생성 시간 |
| updated_at | timestamp | 수정 시간 |

### translations (번역 기록)

| 컬럼 | 타입 | 설명 |
|-----|------|------|
| id | uuid | 기본 키 |
| source_url | text | 원본 URL |
| filename | text | 저장 파일명 |
| domain | text | 도메인 |
| original_md_hash | text | 원본 해시 |
| current_md_hash | text | 현재 해시 |
| created_at | timestamp | 생성 시간 |

### translation_history (청크별 히스토리)

| 컬럼 | 타입 | 설명 |
|-----|------|------|
| id | uuid | 기본 키 |
| translation_id | uuid | 번역 ID (FK) |
| chunk_index | int | 청크 인덱스 |
| original_text | text | 원문 |
| translated_text | text | 번역문 |

### term_changes (용어 변경 로그)

| 컬럼 | 타입 | 설명 |
|-----|------|------|
| id | uuid | 기본 키 |
| terminology_id | uuid | 용어 ID (FK) |
| change_type | text | 변경 유형 |
| old_value | text | 이전 값 |
| new_value | text | 새 값 |
| source_file | text | 변경 출처 파일 |
| created_at | timestamp | 변경 시간 |

## 라이선스

MIT License
