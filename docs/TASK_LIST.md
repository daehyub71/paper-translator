# Paper Translator 상세 작업 목록

> 이 문서는 [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md)를 기반으로 작성된 상세 작업 목록입니다.

---

## Phase 1: 기반 구축

### 1.1 프로젝트 초기 설정
- [x] 프로젝트 디렉토리 구조 생성
- [x] `venv` 가상환경 생성 및 활성화
- [x] `requirements.txt` 작성 (langgraph, langchain-openai, pypdf2, pdfplumber, supabase, python-dotenv, pyyaml, requests, jinja2, typer, rich)
- [x] `.env.example` 파일 생성
- [x] `.env` 파일 생성 및 API 키 설정
- [x] `.gitignore` 파일 생성
- [x] `config/settings.yaml` 설정 파일 생성

### 1.2 Supabase 프로젝트 생성 및 테이블 구축
- [x] Supabase 프로젝트 생성
- [x] `terminology_mappings` 테이블 생성 (전문용어 매핑)
- [x] `translations` 테이블 생성 (번역 기록)
- [x] `translation_history` 테이블 생성 (청크별 히스토리)
- [x] `term_changes` 테이블 생성 (용어 변경 로그)
- [x] 각 테이블 인덱스 생성

### 1.3 초기 용어 데이터(Seed) 삽입
- [x] `scripts/seed_terminology.py` 스크립트 작성
- [x] Architecture 용어 삽입 (Transformer, attention, self-attention 등)
- [x] Training 용어 삽입 (fine-tuning, pre-training, gradient descent 등)
- [x] LLM Specific 용어 삽입 (hallucination, prompt engineering 등)
- [x] Metrics 용어 삽입 (accuracy, precision, recall 등)
- [x] Common Phrases 용어 삽입 (state-of-the-art, end-to-end 등)

### 1.4 Supabase 클라이언트 및 Repository 구현
- [x] `src/db/__init__.py` 생성
- [x] `src/db/supabase_client.py` 구현 (연결 관리)
- [x] `src/db/repositories.py` 구현
  - [x] `TerminologyRepository` 클래스 (용어 CRUD)
  - [x] `TranslationRepository` 클래스 (번역 기록 CRUD)
  - [x] `TranslationHistoryRepository` 클래스 (청크 히스토리 CRUD)
  - [x] `TermChangeRepository` 클래스 (변경 로그 CRUD)

### 1.5 LLM 클라이언트 구현
- [x] `src/utils/__init__.py` 생성
- [x] `src/utils/llm_client.py` 구현
  - [x] OpenAI 클라이언트 초기화
  - [x] 번역용 completion 함수
  - [x] 용어 분석용 completion 함수
  - [x] 토큰 카운트 유틸리티

---

## Phase 2: 핵심 파이프라인

### 2.1 PDF 파서 구현
- [x] `src/parsers/__init__.py` 생성
- [x] `src/parsers/pdf_parser.py` 구현
  - [x] ArXiv URL에서 PDF 다운로드 함수
  - [x] ArXiv ID에서 PDF URL 변환 함수
  - [x] 로컬 PDF 파일 로드 함수
  - [x] PyPDF2로 텍스트 추출
  - [x] pdfplumber로 표 추출
  - [x] LaTeX 수식 패턴 유지 로직
  - [x] References 섹션 제외 옵션

### 2.2 하이브리드 청커 구현
- [x] `src/processors/__init__.py` 생성
- [x] `src/processors/chunker.py` 구현
  - [x] 섹션 헤더 감지 로직 (Abstract, Introduction 등)
  - [x] 섹션 기반 청킹 함수
  - [x] 토큰 기반 분할 함수 (max_chunk_tokens 적용)
  - [x] 하이브리드 청킹 함수 (섹션 + 토큰 오버랩)
  - [x] 청크 메타데이터 생성 (section_title, chunk_index)

### 2.3 Pre-processor 구현
- [x] `src/processors/pre_processor.py` 구현
  - [x] DB에서 도메인별 용어 조회 함수
  - [x] 청크 내 매칭 용어 추출 함수
  - [x] 용어 프롬프트 구성 함수 (pre_process_limit 적용)
  - [x] 청크별 용어 프롬프트 주입 로직

### 2.4 Translator 구현
- [x] `src/processors/translator.py` 구현
  - [x] 번역 프롬프트 템플릿 정의
  - [x] 단일 청크 번역 함수
  - [x] 배치 청크 번역 함수 (rate limiting 고려)
  - [x] 토큰 사용량 추적 로직
  - [x] 재시도 로직 (API 에러 처리)

### 2.5 Post-processor 구현
- [x] `src/processors/post_processor.py` 구현
  - [x] 번역 결과에서 용어 매칭 검증
  - [x] 누락 용어 검출 함수
  - [x] 불일치 용어 교정 함수 (threshold 적용)
  - [x] 교정 결과 로깅

### 2.6 Markdown Writer 구현
- [x] `src/outputs/__init__.py` 생성
- [x] `src/outputs/markdown_writer.py` 구현
  - [x] `templates/paper_template.md.j2` 템플릿 생성
  - [x] Jinja2 환경 설정
  - [x] 마크다운 렌더링 함수
  - [x] 파일 저장 함수 (filename_format 적용)
  - [x] MD 해시 계산 함수 (diff 기준용)

### 2.7 LangGraph 워크플로우 통합
- [x] `src/state.py` 구현 (TypedDict 상태 정의)
- [x] `src/graph.py` 구현
  - [x] `fetch_pdf` 노드 정의
  - [x] `parse_pdf` 노드 정의
  - [x] `chunk_text` 노드 정의
  - [x] `pre_process` 노드 정의
  - [x] `translate_chunks` 노드 정의
  - [x] `post_process` 노드 정의
  - [x] `generate_markdown` 노드 정의
  - [x] `save_output` 노드 정의
  - [x] 노드 간 엣지 연결
  - [x] 워크플로우 컴파일

---

## Phase 3: CLI 및 피드백

### 3.1 CLI 구현
- [x] `src/main.py` 구현 (typer 앱)
  - [x] `translate` 명령어
    - [x] `--url` 옵션 (ArXiv URL)
    - [x] `--arxiv-id` 옵션
    - [x] `--file` 옵션 (로컬 PDF)
    - [x] `--domain` 옵션 (NLP, CV, RL 등)
  - [x] `sync` 명령어
    - [x] `--file` 옵션 (단일 파일)
    - [x] `--all` 옵션 (모든 변경 파일)
  - [x] `terms` 명령어 그룹
    - [x] `list` 서브명령 (`--domain` 필터)
    - [x] `add` 서브명령 (`--source`, `--target`, `--domain`)
    - [x] `update` 서브명령
    - [x] `delete` 서브명령
    - [x] `export` 서브명령 (`--output`)
    - [x] `import` 서브명령 (`--file`)
  - [x] rich 라이브러리로 출력 포맷팅 (진행률, 테이블 등)

### 3.2 Diff Analyzer 구현
- [x] `src/feedback/__init__.py` 생성
- [x] `src/feedback/diff_analyzer.py` 구현
  - [x] 원본 해시 vs 현재 파일 비교 함수
  - [x] 변경된 부분 추출 함수 (difflib 활용)
  - [x] LLM으로 용어 변경 분석 함수
  - [x] 변경 사항 파싱 로직

### 3.3 DB 자동 업데이트 로직 구현
- [x] `terminology_mappings` 테이블 업데이트 로직
- [x] `term_changes` 테이블 로깅 로직
- [x] `translations.current_md_hash` 업데이트 로직
- [x] 사용자 확인 프롬프트 (auto_sync: false 시)

---

## Phase 4: InsightBot 연동

### 4.1 InsightBot 호출 인터페이스 구현
- [x] 외부 호출용 함수 인터페이스 정의
- [x] 입력 스키마 정의 (URL, ArXiv ID, 도메인)
- [x] 출력 스키마 정의 (번역 결과 경로, 상태)
- [x] 비동기 실행 지원 (옵션)

### 4.2 InsightBot 그래프 연동 (선택)
- [x] InsightBot 그래프에서 호출 가능한 노드 래퍼
- [x] 논문 선택 → 번역 트리거 흐름 정의
- [x] 번역 결과 InsightBot 상태에 반영

---

## Phase 5: 테스트 및 문서화

### 5.1 단위 테스트 작성
- [x] `tests/__init__.py` 생성
- [x] `tests/test_pdf_parser.py` 작성
  - [x] PDF 다운로드 테스트
  - [x] 텍스트 추출 테스트
  - [x] 표 파싱 테스트
- [x] `tests/test_chunker.py` 작성
  - [x] 섹션 감지 테스트
  - [x] 토큰 분할 테스트
  - [x] 하이브리드 청킹 테스트
- [x] `tests/test_translator.py` 작성
  - [x] 번역 함수 테스트 (mock LLM)
  - [x] 용어 적용 테스트
- [x] `tests/test_diff_analyzer.py` 작성
  - [x] 해시 비교 테스트
  - [x] 변경 추출 테스트
- [x] `tests/test_repositories.py` 작성
  - [x] CRUD 함수 테스트

### 5.2 통합 테스트
- [x] 실제 ArXiv 논문 번역 E2E 테스트
- [x] 피드백 루프 통합 테스트
- [x] CLI 명령어별 통합 테스트

### 5.3 문서화
- [x] `README.md` 작성
  - [x] 프로젝트 소개
  - [x] 설치 방법
  - [x] 사용법 (CLI 예시)
  - [x] 환경 변수 설명
- [x] `CLAUDE.md` 작성
  - [x] 프로젝트 구조 설명
  - [x] 개발 가이드
  - [x] 주요 명령어
  - [x] 트러블슈팅

---

## 작업 요약

| Phase | 설명 | 태스크 수 | 우선순위 |
|-------|------|----------|---------|
| Phase 1 | 기반 구축 | 15개 | 1순위 |
| Phase 2 | 핵심 파이프라인 | 25개 | 2순위 |
| Phase 3 | CLI 및 피드백 | 18개 | 3순위 |
| Phase 4 | InsightBot 연동 | 5개 | 5순위 |
| Phase 5 | 테스트 및 문서화 | 12개 | 4순위 |

**총 작업 항목: 약 75개**

---

## 진행 상황 추적

### 완료된 Phase
- [x] Phase 1: 기반 구축 ✅ (15/15 완료)
- [x] Phase 2: 핵심 파이프라인 ✅ (25/25 완료)
- [x] Phase 3: CLI 및 피드백 ✅ (24/24 완료 - 3.1, 3.2, 3.3 완료)
- [x] Phase 4: InsightBot 연동 ✅ (7/7 완료 - 4.1, 4.2 완료)
- [x] Phase 5: 테스트 및 문서화 ✅ (12/12 완료 - 5.1, 5.2, 5.3 완료)

### 전체 진행률
- **완료**: 86/86 태스크 (100%)
- **프로젝트 완료!**

### 주요 마일스톤
| 날짜 | 마일스톤 |
|------|----------|
| 2025-01-08 | Phase 1 완료 (기반 구축) |
| 2025-01-09 | Phase 2 완료 (핵심 파이프라인) |
| 2025-01-09 | Phase 3.1 완료 (CLI 구현) |
| 2025-01-09 | Phase 3.2 완료 (Diff Analyzer) |
| 2025-01-09 | Phase 3.3 완료 (DB 자동 업데이트) |
| 2025-01-09 | Phase 4.1 완료 (InsightBot 인터페이스) |
| 2025-01-09 | Phase 4.2 완료 (InsightBot 그래프 연동) |
| 2025-01-09 | Phase 5.1 완료 (단위 테스트 - 180개 테스트) |
| 2025-01-09 | Phase 5.2 완료 (통합 테스트) |
| 2025-01-09 | Phase 5.3 완료 (문서화 - README.md, CLAUDE.md) |

---

*문서 작성일: 2025-01-08*
*마지막 업데이트: 2025-01-09*
*기반 문서: DEVELOPMENT_PLAN.md*
