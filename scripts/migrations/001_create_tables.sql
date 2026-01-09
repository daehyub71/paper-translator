-- Paper Translator Database Schema
-- Version: 001
-- Description: 초기 테이블 생성

-- ============================================
-- 1. terminology_mappings (전문용어 매핑)
-- ============================================
CREATE TABLE IF NOT EXISTS terminology_mappings (
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

-- terminology_mappings 인덱스
CREATE INDEX IF NOT EXISTS idx_terminology_source ON terminology_mappings(source_text);
CREATE INDEX IF NOT EXISTS idx_terminology_domain ON terminology_mappings(domain);
CREATE INDEX IF NOT EXISTS idx_terminology_type ON terminology_mappings(mapping_type);

-- ============================================
-- 2. translations (번역 기록)
-- ============================================
CREATE TABLE IF NOT EXISTS translations (
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

-- translations 인덱스
CREATE INDEX IF NOT EXISTS idx_translations_arxiv ON translations(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_translations_status ON translations(status);

-- ============================================
-- 3. translation_history (번역 히스토리 - 청크별)
-- ============================================
CREATE TABLE IF NOT EXISTS translation_history (
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

-- translation_history 인덱스
CREATE INDEX IF NOT EXISTS idx_history_translation ON translation_history(translation_id);

-- ============================================
-- 4. term_changes (용어 변경 로그)
-- ============================================
CREATE TABLE IF NOT EXISTS term_changes (
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

-- ============================================
-- 5. updated_at 자동 업데이트 트리거
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- terminology_mappings 트리거
DROP TRIGGER IF EXISTS update_terminology_mappings_updated_at ON terminology_mappings;
CREATE TRIGGER update_terminology_mappings_updated_at
    BEFORE UPDATE ON terminology_mappings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- translations 트리거
DROP TRIGGER IF EXISTS update_translations_updated_at ON translations;
CREATE TRIGGER update_translations_updated_at
    BEFORE UPDATE ON translations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
