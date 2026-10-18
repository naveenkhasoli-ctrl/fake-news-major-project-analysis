-- ==============================================================================
-- DATABASE SCHEMA: Advanced Fake News Detection & Misinformation Analysis
-- Target Database: PostgreSQL 14+ / Neon Serverless Postgres
-- Author: Senior ML & Data Engineering Team
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- Table A: news_articles
-- Stores ingested raw/preprocessed articles with SHA-256 content deduplication
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_articles (
    id BIGSERIAL PRIMARY KEY,
    article_hash VARCHAR(64) NOT NULL UNIQUE,
    title TEXT,
    article_text TEXT,
    subject VARCHAR(100),
    publication_date VARCHAR(100),
    dataset_label SMALLINT NOT NULL, -- 0 = Fake, 1 = Real (dataset annotation)
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_articles_hash ON news_articles (article_hash);
CREATE INDEX IF NOT EXISTS idx_articles_label ON news_articles (dataset_label);
CREATE INDEX IF NOT EXISTS idx_articles_subject ON news_articles (subject);

-- ------------------------------------------------------------------------------
-- Table B: prediction_history
-- Records real-time and batch model inferences with probability breakdowns
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_history (
    id BIGSERIAL PRIMARY KEY,
    article_hash VARCHAR(64) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    predicted_label VARCHAR(20) NOT NULL,
    confidence_score REAL,
    fake_probability REAL,
    real_probability REAL,
    prediction_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_pred_article FOREIGN KEY (article_hash)
        REFERENCES news_articles (article_hash)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pred_hash ON prediction_history (article_hash);
CREATE INDEX IF NOT EXISTS idx_pred_model ON prediction_history (model_name, model_version);

-- ------------------------------------------------------------------------------
-- Table C: model_experiments
-- Tracks systematic evaluation runs, hyperparameters, and holdout test metrics
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_experiments (
    id BIGSERIAL PRIMARY KEY,
    experiment_name VARCHAR(150) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    dataset_version VARCHAR(50) NOT NULL,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    roc_auc REAL,
    training_duration REAL, -- Duration in seconds
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_exp_name ON model_experiments (experiment_name);
CREATE INDEX IF NOT EXISTS idx_exp_model ON model_experiments (model_name);

-- ------------------------------------------------------------------------------
-- Table D: model_registry
-- Catalogs trained model artifacts, serialization paths, and hyperparameter JSON
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    artifact_path VARCHAR(255) NOT NULL,
    tokenizer_name VARCHAR(100),
    configuration_json JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_model_version UNIQUE (model_name, model_version)
);

-- ------------------------------------------------------------------------------
-- Table E: users
-- Secure access control for analysts and auditors (salted hash storage)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'analyst',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_username ON users (username);
