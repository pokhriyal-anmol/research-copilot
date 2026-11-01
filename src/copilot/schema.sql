CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    license TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    arxiv_id TEXT NOT NULL REFERENCES papers ON DELETE CASCADE,
    page INT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384) NOT NULL
);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS traces (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    question TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    outcome TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    input_tokens INT,
    output_tokens INT,
    latency_ms INT NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_created_at ON traces (created_at);
