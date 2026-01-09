# Paper Translator 개발 계획서

## 1. 프로젝트 개요

### 1.1 프로젝트 명
**Paper Translator** - AI 논문 한국어 번역 시스템

### 1.2 목표
ArXiv 등에서 수집한 AI 연구 논문 PDF를 한국어로 번역하고, 사용자 정의 전문용어 매핑을 통해 일관된 고품질 번역을 제공하는 자동화 시스템 구축

### 1.3 주요 기능
| 기능 | 설명 |
|-----|------|
| PDF 전체 번역 | ArXiv 논문 PDF를 다운로드하여 전체 내용 번역 |
| 전문용어 매핑 | 단어/문장 단위 용어 DB 관리, 도메인별 분류 |
| 하이브리드 용어 적용 | Pre-translation(프롬프트 주입) + Post-translation(후처리 검증) |
| 피드백 루프 | 마크다운 수정 시 변경 감지 → DB 자동 업데이트 |
| 번역 히스토리 | 번역 이력 저장, 일괄 수정 지원 |
| 다중 트리거 | CLI 독립 실행 + InsightBot 연동 |

---

## 2. 기술 스택

| 카테고리 | 기술 |
|---------|------|
| Language | Python 3.10+ |
| LLM | OpenAI GPT-4o-mini |
| Orchestration | LangGraph |
| PDF Parsing | PyPDF2 / pdfplumber |
| Database | Supabase (PostgreSQL) |
| Template | Jinja2 (Markdown 생성) |
| CLI | argparse / typer |
| Config | python-dotenv, PyYAML |

### 2.1 주요 라이브러리
```
langgraph
langchain-openai
langchain
pypdf2
pdfplumber          # 표 추출용
supabase
python-dotenv
pyyaml
requests
jinja2
typer               # CLI 프레임워크
rich                # CLI 출력 포맷팅
```

---

## 3. 시스템 아키텍처

### 3.1 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   [CLI 독립 실행]                      [InsightBot 연동]                  │
│   $ python main.py translate           InsightBot → Paper Translator    │
│     --url "https://arxiv.org/..."      (선택한 논문 자동 전달)            │
│     --arxiv-id "2401.12345"                                              │
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
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          FEEDBACK LOOP                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   [사용자가 마크다운 직접 수정]                                           │
│                     ↓                                                    │
│   $ python main.py sync --file "번역파일.md"                             │
│                     ↓                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  diff_analyzer                                                   │   │
│   │  1. 원본 해시 vs 현재 파일 비교                                   │   │
│   │  2. 변경된 부분 추출                                              │   │
│   │  3. LLM으로 용어 변경 분석                                        │   │
│   │  4. terminology_mappings 테이블 업데이트                          │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 LangGraph 노드 상세

| 노드 | 입력 | 출력 | 설명 |
|-----|------|------|------|
| `fetch_pdf` | URL/ArXiv ID | PDF 바이너리 | ArXiv에서 PDF 다운로드 |
| `parse_pdf` | PDF 바이너리 | 구조화된 텍스트 | 텍스트 추출, 표 파싱, 수식 유지 |
| `chunk_text` | 구조화된 텍스트 | 청크 리스트 | 섹션 기반 + 긴 섹션 분할 |
| `pre_process` | 청크 리스트 | 청크 + 용어 프롬프트 | DB에서 용어 조회, 프롬프트 구성 |
| `translate_chunks` | 청크 + 프롬프트 | 번역된 청크 | GPT-4o-mini로 번역 |
| `post_process` | 번역된 청크 | 교정된 청크 | 누락 용어 검증, 불일치 교정 |
| `generate_markdown` | 교정된 청크 | Markdown 문자열 | Jinja2 템플릿으로 MD 생성 |
| `save_output` | Markdown | 파일 경로 | 로컬 저장, DB 기록 |

---

## 4. 데이터베이스 스키마

### 4.1 Supabase 테이블 구조

#### terminology_mappings (전문용어 매핑)
```sql
CREATE TABLE terminology_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_text TEXT NOT NULL,              -- 영어 원문
    target_text TEXT NOT NULL,              -- 한국어 번역
    mapping_type VARCHAR(20) NOT NULL,      -- 'word' | 'phrase' | 'sentence'
    domain VARCHAR(50) DEFAULT 'General',   -- 'NLP' | 'CV' | 'RL' | 'General' | etc.
    confidence FLOAT DEFAULT 1.0,           -- 신뢰도 (사용자 정의 = 1.0)
    is_user_defined BOOLEAN DEFAULT TRUE,   -- 사용자 정의 여부
    usage_count INT DEFAULT 0,              -- 사용 횟수 (자주 쓰이는 용어 우선)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT unique_source_domain UNIQUE (source_text, domain)
);

-- 인덱스
CREATE INDEX idx_terminology_source ON terminology_mappings(source_text);
CREATE INDEX idx_terminology_domain ON terminology_mappings(domain);
CREATE INDEX idx_terminology_type ON terminology_mappings(mapping_type);
```

#### translations (번역 기록)
```sql
CREATE TABLE translations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_title TEXT NOT NULL,
    paper_url TEXT,
    arxiv_id VARCHAR(50),
    domain VARCHAR(50),                     -- 논문 도메인 (NLP, CV 등)
    original_md_hash VARCHAR(64),           -- 번역 직후 MD 해시 (diff 기준)
    current_md_hash VARCHAR(64),            -- 현재 MD 해시
    output_path TEXT NOT NULL,              -- 로컬 파일 경로
    total_chunks INT,                       -- 전체 청크 수
    status VARCHAR(20) DEFAULT 'in_progress', -- 'completed' | 'in_progress' | 'error'
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 인덱스
CREATE INDEX idx_translations_arxiv ON translations(arxiv_id);
CREATE INDEX idx_translations_status ON translations(status);
```

#### translation_history (번역 히스토리 - 청크별)
```sql
CREATE TABLE translation_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    translation_id UUID REFERENCES translations(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,               -- 청크 순서
    section_title TEXT,                     -- 섹션 제목 (있는 경우)
    original_text TEXT NOT NULL,            -- 원문
    translated_text TEXT NOT NULL,          -- 번역문
    terms_applied JSONB,                    -- 적용된 용어 목록 [{"source": "...", "target": "..."}]
    tokens_used INT,                        -- 사용된 토큰 수
    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT unique_translation_chunk UNIQUE (translation_id, chunk_index)
);

-- 인덱스
CREATE INDEX idx_history_translation ON translation_history(translation_id);
```

#### term_changes (용어 변경 로그)
```sql
CREATE TABLE term_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    translation_id UUID REFERENCES translations(id) ON DELETE SET NULL,
    terminology_id UUID REFERENCES terminology_mappings(id) ON DELETE SET NULL,
    change_type VARCHAR(20) NOT NULL,       -- 'add' | 'update' | 'delete'
    old_target_text TEXT,                   -- 변경 전 번역
    new_target_text TEXT,                   -- 변경 후 번역
    source_text TEXT NOT NULL,              -- 원문
    detected_from TEXT,                     -- 'markdown_sync' | 'manual' | 'auto'
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 4.2 초기 용어 데이터 (Seed)

```sql
-- 기본 AI/ML 용어
INSERT INTO terminology_mappings (source_text, target_text, mapping_type, domain) VALUES
-- Architecture
('Transformer', '트랜스포머', 'word', 'NLP'),
('attention mechanism', '어텐션 메커니즘', 'phrase', 'NLP'),
('self-attention', '셀프 어텐션', 'word', 'NLP'),
('multi-head attention', '멀티헤드 어텐션', 'phrase', 'NLP'),
('feed-forward network', '피드포워드 네트워크', 'phrase', 'General'),
('residual connection', '잔차 연결', 'phrase', 'General'),
('layer normalization', '레이어 정규화', 'phrase', 'General'),

-- Training
('fine-tuning', '미세조정', 'word', 'General'),
('pre-training', '사전학습', 'word', 'General'),
('transfer learning', '전이학습', 'phrase', 'General'),
('gradient descent', '경사 하강법', 'phrase', 'General'),
('backpropagation', '역전파', 'word', 'General'),
('learning rate', '학습률', 'phrase', 'General'),
('batch size', '배치 크기', 'phrase', 'General'),
('epoch', '에폭', 'word', 'General'),
('overfitting', '과적합', 'word', 'General'),
('underfitting', '과소적합', 'word', 'General'),
('regularization', '정규화', 'word', 'General'),
('dropout', '드롭아웃', 'word', 'General'),

-- LLM Specific
('hallucination', '환각 현상', 'word', 'NLP'),
('prompt engineering', '프롬프트 엔지니어링', 'phrase', 'NLP'),
('in-context learning', '인컨텍스트 학습', 'phrase', 'NLP'),
('chain-of-thought', '사고의 연쇄', 'phrase', 'NLP'),
('retrieval-augmented generation', 'RAG', 'phrase', 'NLP'),
('tokenization', '토큰화', 'word', 'NLP'),
('embedding', '임베딩', 'word', 'General'),

-- Metrics
('accuracy', '정확도', 'word', 'General'),
('precision', '정밀도', 'word', 'General'),
('recall', '재현율', 'word', 'General'),
('F1 score', 'F1 점수', 'phrase', 'General'),
('perplexity', '퍼플렉시티', 'word', 'NLP'),
('BLEU score', 'BLEU 점수', 'phrase', 'NLP'),

-- Common Phrases
('state-of-the-art', '최신 기술', 'phrase', 'General'),
('from scratch', '처음부터', 'phrase', 'General'),
('end-to-end', '엔드투엔드', 'phrase', 'General'),
('out-of-the-box', '기본 설정으로', 'phrase', 'General');
```

---

## 5. 프로젝트 구조

```
paper-translator/
├── src/
│   ├── __init__.py
│   ├── main.py                     # CLI 진입점 (typer)
│   ├── graph.py                    # LangGraph 워크플로우 정의
│   ├── state.py                    # 상태 타입 정의
│   │
│   ├── parsers/
│   │   ├── __init__.py
│   │   └── pdf_parser.py           # PDF 다운로드 & 파싱
│   │
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── chunker.py              # 하이브리드 청킹
│   │   ├── pre_processor.py        # 용어 프롬프트 주입
│   │   ├── translator.py           # LLM 번역
│   │   └── post_processor.py       # 용어 검증/교정
│   │
│   ├── outputs/
│   │   ├── __init__.py
│   │   └── markdown_writer.py      # 마크다운 생성
│   │
│   ├── feedback/
│   │   ├── __init__.py
│   │   └── diff_analyzer.py        # 변경 감지 & DB 반영
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── supabase_client.py      # Supabase 연결
│   │   └── repositories.py         # CRUD 함수들
│   │
│   └── utils/
│       ├── __init__.py
│       └── llm_client.py           # OpenAI 클라이언트
│
├── templates/
│   └── paper_template.md.j2        # 마크다운 Jinja2 템플릿
│
├── translations/                   # 번역 결과 저장 폴더
│   └── .gitkeep
│
├── config/
│   └── settings.yaml               # 설정 파일
│
├── tests/
│   ├── __init__.py
│   ├── test_pdf_parser.py
│   ├── test_chunker.py
│   ├── test_translator.py
│   └── test_diff_analyzer.py
│
├── docs/
│   └── DEVELOPMENT_PLAN.md         # 본 문서
│
├── scripts/
│   └── seed_terminology.py         # 초기 용어 데이터 삽입
│
├── .env.example
├── .gitignore
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## 6. 개발 단계

### Phase 1: 기반 구축
| 태스크 | 설명 |
|-------|------|
| 1.1 | 프로젝트 초기 설정 (venv, requirements.txt, .env) |
| 1.2 | Supabase 프로젝트 생성 및 테이블 구축 |
| 1.3 | 초기 용어 데이터(Seed) 삽입 |
| 1.4 | Supabase 클라이언트 및 Repository 구현 |
| 1.5 | LLM 클라이언트 구현 (OpenAI GPT-4o-mini) |

### Phase 2: 핵심 파이프라인
| 태스크 | 설명 |
|-------|------|
| 2.1 | PDF 파서 구현 (다운로드, 텍스트 추출, 표 파싱) |
| 2.2 | 하이브리드 청커 구현 (섹션 기반 + 토큰 분할) |
| 2.3 | Pre-processor 구현 (용어 조회, 프롬프트 구성) |
| 2.4 | Translator 구현 (청크별 LLM 번역) |
| 2.5 | Post-processor 구현 (용어 검증, 불일치 교정) |
| 2.6 | Markdown Writer 구현 (Jinja2 템플릿) |
| 2.7 | LangGraph 워크플로우 통합 |

### Phase 3: CLI 및 피드백
| 태스크 | 설명 |
|-------|------|
| 3.1 | CLI 구현 (`translate`, `sync`, `terms` 명령어) |
| 3.2 | Diff Analyzer 구현 (변경 감지, LLM 분석) |
| 3.3 | DB 자동 업데이트 로직 구현 |

### Phase 4: InsightBot 연동
| 태스크 | 설명 |
|-------|------|
| 4.1 | InsightBot에서 호출 가능한 인터페이스 구현 |
| 4.2 | InsightBot 그래프에 번역 노드 추가 (선택적) |

### Phase 5: 테스트 및 문서화
| 태스크 | 설명 |
|-------|------|
| 5.1 | 단위 테스트 작성 |
| 5.2 | 통합 테스트 (실제 논문 번역) |
| 5.3 | README 및 CLAUDE.md 작성 |

---

## 7. CLI 사용법 (예정)

### 7.1 논문 번역
```bash
# ArXiv URL로 번역
python src/main.py translate --url "https://arxiv.org/pdf/1706.03762"

# ArXiv ID로 번역
python src/main.py translate --arxiv-id "1706.03762"

# 로컬 PDF 파일 번역
python src/main.py translate --file "./papers/attention.pdf"

# 도메인 지정 (용어 필터링용)
python src/main.py translate --url "..." --domain NLP
```

### 7.2 피드백 동기화
```bash
# 수정된 마크다운 동기화
python src/main.py sync --file "./translations/2024-01-08_Attention.md"

# 모든 변경 파일 동기화
python src/main.py sync --all
```

### 7.3 용어 관리
```bash
# 용어 목록 조회
python src/main.py terms list --domain NLP

# 용어 추가
python src/main.py terms add --source "RLHF" --target "인간 피드백 강화학습" --domain NLP

# 용어 수정
python src/main.py terms update --source "Transformer" --target "변환기"

# 용어 삭제
python src/main.py terms delete --source "Transformer"

# 용어 내보내기 (백업)
python src/main.py terms export --output "./backup/terms.json"

# 용어 가져오기
python src/main.py terms import --file "./backup/terms.json"
```

---

## 8. 환경 변수

### .env.example
```ini
# OpenAI
OPENAI_API_KEY=sk-...

# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Optional: InsightBot 연동
INSIGHTBOT_API_URL=http://localhost:8000
```

---

## 9. 설정 파일

### config/settings.yaml
```yaml
# Paper Translator Configuration

translation:
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens_per_chunk: 4000

chunking:
  strategy: "hybrid"              # 'section' | 'token' | 'hybrid'
  max_chunk_tokens: 3000
  overlap_tokens: 200

pdf_parsing:
  keep_formulas: true             # 수식 유지 (LaTeX)
  translate_tables: true          # 표 내용 번역
  exclude_references: true        # References 섹션 제외

terminology:
  pre_process_limit: 30           # 프롬프트에 포함할 최대 용어 수
  post_process_threshold: 0.8     # 후처리 용어 매칭 신뢰도 임계값

output:
  directory: "./translations"
  filename_format: "{date}_{title}"   # 파일명 포맷

feedback:
  auto_sync: false                # 자동 동기화 (false = 수동 sync 명령)
```

---

## 10. 마크다운 템플릿

### templates/paper_template.md.j2
```markdown
---
title: "{{ paper.title }}"
arxiv_id: "{{ paper.arxiv_id }}"
translated_at: "{{ paper.translated_at }}"
domain: "{{ paper.domain }}"
original_url: "{{ paper.url }}"
---

# {{ paper.title_ko }}

> 원제: {{ paper.title }}
> ArXiv: [{{ paper.arxiv_id }}]({{ paper.url }})
> 번역일: {{ paper.translated_at }}

---

{% for section in sections %}
## {{ section.title_ko }}

{{ section.content_ko }}

{% endfor %}

---

## 적용된 용어

| 원문 | 번역 |
|-----|------|
{% for term in applied_terms %}
| {{ term.source }} | {{ term.target }} |
{% endfor %}
```

---

## 11. 예상 결과물

### 번역 결과 예시: `translations/2024-01-08_Attention_Is_All_You_Need.md`

```markdown
---
title: "Attention Is All You Need"
arxiv_id: "1706.03762"
translated_at: "2024-01-08"
domain: "NLP"
original_url: "https://arxiv.org/pdf/1706.03762"
---

# 어텐션이 전부다

> 원제: Attention Is All You Need
> ArXiv: [1706.03762](https://arxiv.org/pdf/1706.03762)
> 번역일: 2024-01-08

---

## 초록

기존의 시퀀스 변환 모델은 인코더와 디코더를 포함하는 복잡한 순환 또는
합성곱 신경망을 기반으로 한다. 가장 좋은 성능을 보이는 모델들도
어텐션 메커니즘을 통해 인코더와 디코더를 연결한다. 우리는 **트랜스포머**라는
새로운 간단한 네트워크 아키텍처를 제안한다. 이 모델은 순환과 합성곱을
완전히 배제하고 **셀프 어텐션** 메커니즘에만 기반한다...

## 1. 서론

순환 신경망, 특히 LSTM과 GRU는 시퀀스 모델링과 언어 모델링,
기계 번역 등의 변환 문제에서 **최신 기술**로 확고히 자리잡았다...

---

## 적용된 용어

| 원문 | 번역 |
|-----|------|
| Transformer | 트랜스포머 |
| self-attention | 셀프 어텐션 |
| attention mechanism | 어텐션 메커니즘 |
| state-of-the-art | 최신 기술 |
```

---

## 12. 리스크 및 고려사항

| 리스크 | 완화 방안 |
|-------|----------|
| PDF 파싱 품질 | pdfplumber로 표 추출, 수식은 LaTeX 패턴 유지 |
| 긴 논문 토큰 비용 | 청킹 + gpt-4o-mini 사용으로 비용 최적화 |
| 용어 일관성 | Pre+Post 하이브리드로 이중 검증 |
| Diff 분석 정확도 | LLM 기반 분석 + 사용자 확인 절차 |
| Supabase 무료 한도 | 500MB DB, 1GB Storage - 충분할 것으로 예상 |

---

## 13. 향후 확장 가능성

- [ ] Notion 직접 저장 옵션
- [ ] 블로그 포스팅 자동화 (Jekyll, Hugo)
- [ ] 용어 유사도 검색 (벡터 DB 추가)
- [ ] 웹 UI (Streamlit)
- [ ] 다국어 지원 (일본어, 중국어)
- [ ] 논문 요약 + 번역 통합 모드

---

*문서 작성일: 2024-01-08*
*작성자: Claude Code*
