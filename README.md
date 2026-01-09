# Paper Translator

AI-powered translation system for translating ArXiv papers into Korean.

## Overview

Paper Translator is an automated system that translates AI research papers (PDFs) from ArXiv into Korean, providing consistent high-quality translations through terminology mapping.

### Key Features

| Feature | Description |
|---------|-------------|
| Paper Discovery | Search papers via ArXiv API and Semantic Scholar (citation-based filtering) |
| Full PDF Translation | Download and translate entire ArXiv paper PDFs |
| Terminology Mapping | Word/phrase-level term DB management with domain classification |
| Hybrid Term Application | Pre-translation (prompt injection) + Post-translation (validation) |
| Feedback Loop | Detect markdown changes and auto-update DB |
| Translation History | Store translation records, support batch modifications |
| Multiple Triggers | CLI standalone execution + InsightBot integration |

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.9+ |
| LLM | OpenAI GPT-4o-mini |
| Orchestration | LangGraph |
| PDF Parsing | PyPDF2 / pdfplumber |
| Paper Discovery | arxiv (ArXiv API) / requests (Semantic Scholar API) |
| Database | Supabase (PostgreSQL) |
| Template | Jinja2 (Markdown generation) |
| CLI | Typer + Rich |

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/paper-translator.git
cd paper-translator
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file with your settings:

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

### 5. Seed Initial Terminology Data (Optional)

```bash
python scripts/seed_terminology.py
```

## Usage

### CLI Commands

#### Discover Papers

Search and discover papers from ArXiv or Semantic Scholar:

```bash
# Search ArXiv for trending NLP papers
python -m src.main discover --source arxiv --domain NLP --trending

# Search ArXiv with query
python -m src.main discover --source arxiv --query "transformer" --domain NLP

# Search Semantic Scholar for highly cited papers
python -m src.main discover --source semantic-scholar --domain ML --highly-cited

# Search with minimum citation filter
python -m src.main discover --source semantic-scholar --query "BERT" --min-citations 100

# Show detailed information
python -m src.main discover --source arxiv --domain CV --trending --verbose
```

**Discover Options:**

| Option | Description |
|--------|-------------|
| `--source` | Search source: `arxiv`, `semantic-scholar` (or `s2`) |
| `--query` | Search query |
| `--domain` | Domain filter: NLP, CV, ML, RL, Speech, General |
| `--max-results` | Maximum results (default: 10) |
| `--min-citations` | Minimum citations (Semantic Scholar only) |
| `--year-from` | Filter by start year |
| `--trending` | Get trending/recent papers |
| `--highly-cited` | Get highly cited papers (Semantic Scholar only) |
| `--verbose` | Show detailed paper information |
| `--json` | Output as JSON |

#### Translate Papers

```bash
# Translate from URL
python -m src.main translate --url "https://arxiv.org/pdf/1706.03762.pdf"

# Translate from ArXiv ID
python -m src.main translate --arxiv-id "1706.03762"

# Translate local PDF file
python -m src.main translate --file "./paper.pdf"

# Specify domain (NLP, CV, RL, etc.)
python -m src.main translate --url "https://arxiv.org/pdf/1706.03762.pdf" --domain NLP
```

#### Manage Terminology

```bash
# List terms
python -m src.main terms list
python -m src.main terms list --domain NLP

# Add term
python -m src.main terms add --source "transformer" --target "트랜스포머" --domain NLP

# Update term
python -m src.main terms update <term-id> --target "new translation"

# Delete term
python -m src.main terms delete <term-id> --yes

# Export terms
python -m src.main terms export --output ./terms.json

# Import terms
python -m src.main terms import --file ./terms.json
```

#### Feedback Synchronization

Sync modified markdown files to the terminology DB:

```bash
# Sync single file
python -m src.main sync --file ./translations/paper.md

# Sync all changed files
python -m src.main sync --all
```

### Python API

```python
from src.api import translate, TranslationRequest

# Simple translation
result = translate(
    source="https://arxiv.org/pdf/1706.03762.pdf",
    domain="NLP"
)
print(result.output_path)

# Translation with detailed options
request = TranslationRequest(
    source="1706.03762",
    domain="NLP",
    output_dir="./my_translations"
)
response = translate(request)
```

### InsightBot Integration

```python
from src.api.insightbot import (
    TranslationNodeWrapper,
    create_translation_subgraph
)

# Use node wrapper
wrapper = TranslationNodeWrapper(auto_confirm=True)
result = await wrapper.translate_paper_node(state)

# Integrate as subgraph
subgraph = create_translation_subgraph()
```

## Project Structure

```
paper-translator/
├── src/
│   ├── api/                    # External integration interfaces
│   │   ├── interface.py        # TranslationRequest/Response
│   │   └── insightbot.py       # InsightBot integration
│   ├── collectors/             # Paper discovery
│   │   ├── arxiv_collector.py  # ArXiv API client
│   │   └── semantic_scholar_collector.py  # Semantic Scholar API
│   ├── db/                     # Database layer
│   │   ├── supabase_client.py  # Supabase client
│   │   └── repositories.py     # CRUD repositories
│   ├── feedback/               # Feedback loop
│   │   ├── diff_analyzer.py    # Change detection
│   │   └── sync_manager.py     # DB synchronization
│   ├── outputs/                # Output generation
│   │   └── markdown_writer.py  # Markdown generation
│   ├── parsers/                # PDF parsing
│   │   └── pdf_parser.py       # PDF extraction
│   ├── processors/             # Translation processing
│   │   ├── chunker.py          # Text chunking
│   │   ├── pre_processor.py    # Pre-processing (term injection)
│   │   ├── translator.py       # LLM translation
│   │   └── post_processor.py   # Post-processing (term validation)
│   ├── utils/                  # Utilities
│   │   ├── config.py           # Configuration management
│   │   └── llm_client.py       # LLM client
│   ├── graph.py                # LangGraph workflow
│   ├── state.py                # State definition
│   └── main.py                 # CLI entry point
├── scripts/                    # Scripts
│   └── seed_terminology.py     # Initial term seeding
├── templates/                  # Templates
│   └── paper_template.md.j2    # Markdown template
├── tests/                      # Tests
│   ├── test_pdf_parser.py
│   ├── test_chunker.py
│   ├── test_translator.py
│   ├── test_diff_analyzer.py
│   ├── test_repositories.py
│   └── test_integration.py
├── docs/                       # Documentation
│   ├── DEVELOPMENT_PLAN.md
│   └── TASK_LIST.md
├── config/
│   └── settings.yaml           # Configuration file
├── requirements.txt
├── .env.example
└── README.md
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   [CLI Standalone]                         [InsightBot Integration]      │
│   $ python -m src.main translate           InsightBot → Paper Translator │
│     --url "https://arxiv.org/..."          (auto-forward selected paper) │
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
│  │ Download   │    │ Extract    │    │ Hybrid     │    │ Term prompt│  │
│  │ PDF        │    │ text/table │    │ chunking   │    │ injection  │  │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘  │
│                                                               ↓         │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐  │
│  │   save     │ ←  │  generate  │ ←  │   post     │ ←  │ translate  │  │
│  │  _output   │    │ _markdown  │    │ _process   │    │  _chunks   │  │
│  │            │    │            │    │            │    │            │  │
│  │ Save file  │    │ Apply MD   │    │ Term       │    │ LLM        │  │
│  │ DB record  │    │ template   │    │ validation │    │ translation│  │
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

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_translator.py -v
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| OPENAI_API_KEY | Yes | OpenAI API key |
| OPENAI_MODEL | No | Model to use (default: gpt-4o-mini) |
| SUPABASE_URL | Yes | Supabase project URL |
| SUPABASE_KEY | Yes | Supabase anon key |
| SUPABASE_SERVICE_ROLE_KEY | No | Supabase service role key |
| SUPABASE_DATABASE_URL | No | PostgreSQL direct connection URL |
| SEMANTIC_SCHOLAR_API_KEY | No | Semantic Scholar API key (reduces rate limits) |
| UPSTASH_URL | No | Redis cache URL (optional) |
| UPSTASH_TOKEN | No | Redis auth token (optional) |

> **Note:** Get a free Semantic Scholar API key at https://www.semanticscholar.org/product/api#api-key-form

## Database Schema

### terminology_mappings (Terminology Mapping)

| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| source_text | text | Source term |
| target_text | text | Translated term |
| domain | text | Domain (NLP, CV, RL, etc.) |
| confidence | float | Confidence score |
| created_at | timestamp | Creation time |
| updated_at | timestamp | Update time |

### translations (Translation Records)

| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| source_url | text | Source URL |
| filename | text | Saved filename |
| domain | text | Domain |
| original_md_hash | text | Original hash |
| current_md_hash | text | Current hash |
| created_at | timestamp | Creation time |

### translation_history (Chunk History)

| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| translation_id | uuid | Translation ID (FK) |
| chunk_index | int | Chunk index |
| original_text | text | Original text |
| translated_text | text | Translated text |

### term_changes (Term Change Log)

| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| terminology_id | uuid | Term ID (FK) |
| change_type | text | Change type |
| old_value | text | Previous value |
| new_value | text | New value |
| source_file | text | Source file of change |
| created_at | timestamp | Change time |

## License

MIT License
