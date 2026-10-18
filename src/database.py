"""
Database Integration Module for MySQL 8+ and SQLite Fallback.
Uses SQLAlchemy 2.x ORM & Core with secure parameterized queries.
"""

import os
import json
import hashlib
import secrets
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
from sqlalchemy import (
    create_engine, Engine, text,
    Column, Integer, BigInteger, String, Text, Float, DateTime, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from dotenv import load_dotenv

# Load local environment variables if present
load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

Base = declarative_base()


# -----------------------------------------------------------------------------
# SQLAlchemy 2.x ORM Models
# -----------------------------------------------------------------------------

class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    article_hash = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=True)
    article_text = Column(Text, nullable=True)
    subject = Column(String(100), nullable=True, index=True)
    publication_date = Column(String(100), nullable=True)
    dataset_label = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    article_hash = Column(String(64), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    predicted_label = Column(String(20), nullable=False)
    confidence_score = Column(Float, nullable=True)
    fake_probability = Column(Float, nullable=True)
    real_probability = Column(Float, nullable=True)
    prediction_timestamp = Column(DateTime, default=datetime.utcnow)


class ModelExperiment(Base):
    __tablename__ = "model_experiments"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    experiment_name = Column(String(150), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    dataset_version = Column(String(50), nullable=False)
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall_score = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    training_duration = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    artifact_path = Column(String(255), nullable=False)
    tokenizer_name = Column(String(100), nullable=True)
    configuration_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("model_name", "model_version", name="uq_model_version"),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="analyst")
    created_at = Column(DateTime, default=datetime.utcnow)


# -----------------------------------------------------------------------------
# Engine Factory & Utility Functions
# -----------------------------------------------------------------------------

def get_database_engine(force_sqlite: bool = False, sqlite_path: str = "outputs/fake_news.db") -> Engine:
    """
    Constructs a SQLAlchemy 2.x engine.
    1. Checks DATABASE_URL (for PostgreSQL / Neon).
    2. Checks MySQL variables (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD).
    3. Falls back gracefully to local SQLite (outputs/fake_news.db).
    """
    if force_sqlite:
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        logger.info(f"Using forced SQLite database: sqlite:///{sqlite_path}")
        return create_engine(f"sqlite:///{sqlite_path}", echo=False)

    # 1. Check for PostgreSQL / Neon connection string
    db_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or os.getenv("NEON_DATABASE_URL")
    if db_url:
        # Standardize prefix for SQLAlchemy psycopg2 driver
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif db_url.startswith("postgresql://") and not db_url.startswith("postgresql+"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

        # Handle channel_binding parameter if psycopg2 warns
        try:
            engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=300, echo=False)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to Neon / PostgreSQL database via DATABASE_URL.")
            return engine
        except Exception as e:
            logger.warning(
                f"PostgreSQL connection via DATABASE_URL failed: {e}. "
                "Attempting fallback to MySQL or local SQLite."
            )

    # 2. Check MySQL variables
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")

    if db_host and db_name and db_user and db_pass:
        mysql_url = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
        try:
            engine = create_engine(mysql_url, pool_pre_ping=True, pool_recycle=3600, echo=False)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"Successfully connected to MySQL database '{db_name}' at {db_host}:{db_port}.")
            return engine
        except Exception as e:
            logger.warning(
                f"MySQL connection to {db_host}:{db_port} failed ({e}). "
                f"Falling back gracefully to local SQLite database at {sqlite_path}."
            )

    # 3. Fallback to local SQLite
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    logger.info(
        "No remote database configured or reachable. "
        f"Gracefully using local SQLite database at: sqlite:///{sqlite_path}"
    )
    return create_engine(f"sqlite:///{sqlite_path}", echo=False)


def create_database_tables(engine: Optional[Engine] = None) -> None:
    """Create all ORM tables if they do not exist."""
    if engine is None:
        engine = get_database_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables verified and initialized successfully.")


def hash_password(password: str) -> str:
    """Generate salted SHA-256 password hash."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored salt$hash."""
    try:
        salt, hashed = stored_hash.split("$", 1)
        test_hash = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
        return secrets.compare_digest(hashed, test_hash)
    except Exception:
        return False


def store_articles(df: pd.DataFrame, engine: Optional[Engine] = None, batch_size: int = 500) -> int:
    """
    Store articles into news_articles table. Ignores duplicates based on article_hash.
    Returns the number of new records inserted.
    """
    if engine is None:
        engine = get_database_engine()

    create_database_tables(engine)
    SessionLocal = sessionmaker(bind=engine)
    session: Session = SessionLocal()

    inserted = 0
    try:
        for start_idx in range(0, len(df), batch_size):
            batch = df.iloc[start_idx : start_idx + batch_size]
            hashes = batch["article_hash"].tolist()
            
            # Find existing hashes
            existing = session.query(NewsArticle.article_hash).filter(
                NewsArticle.article_hash.in_(hashes)
            ).all()
            existing_set = {h[0] for h in existing}

            new_records = []
            for _, row in batch.iterrows():
                h = row["article_hash"]
                if h not in existing_set:
                    new_records.append(NewsArticle(
                        article_hash=h,
                        title=str(row.get("title", ""))[:65535],
                        article_text=str(row.get("text", "")),
                        subject=str(row.get("subject", "unknown"))[:100],
                        publication_date=str(row.get("date", "unknown"))[:100],
                        dataset_label=int(row["dataset_label"])
                    ))
                    existing_set.add(h)

            if new_records:
                session.bulk_save_objects(new_records)
                session.commit()
                inserted += len(new_records)
        logger.info(f"Stored {inserted} new articles into database.")
    except Exception as e:
        session.rollback()
        logger.error(f"Error storing articles: {e}")
        raise
    finally:
        session.close()

    return inserted


def save_prediction(record: Dict[str, Any], engine: Optional[Engine] = None) -> int:
    """Record an individual prediction into prediction_history."""
    if engine is None:
        engine = get_database_engine()

    create_database_tables(engine)
    SessionLocal = sessionmaker(bind=engine)
    session: Session = SessionLocal()

    try:
        pred = PredictionHistory(
            article_hash=record["article_hash"],
            model_name=record["model_name"],
            model_version=record.get("model_version", "1.0.0"),
            predicted_label=record["predicted_label"],
            confidence_score=record.get("confidence_score"),
            fake_probability=record.get("fake_probability"),
            real_probability=record.get("real_probability")
        )
        session.add(pred)
        session.commit()
        record_id = pred.id
        return record_id
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save prediction record: {e}")
        raise
    finally:
        session.close()


def save_experiment_metrics(record: Dict[str, Any], engine: Optional[Engine] = None) -> int:
    """Record experimental evaluation metrics into model_experiments."""
    if engine is None:
        engine = get_database_engine()

    create_database_tables(engine)
    SessionLocal = sessionmaker(bind=engine)
    session: Session = SessionLocal()

    try:
        exp = ModelExperiment(
            experiment_name=record["experiment_name"],
            model_name=record["model_name"],
            model_version=record.get("model_version", "1.0.0"),
            dataset_version=record.get("dataset_version", "v1.0"),
            accuracy=record.get("accuracy"),
            precision_score=record.get("precision_score"),
            recall_score=record.get("recall_score"),
            f1_score=record.get("f1_score"),
            roc_auc=record.get("roc_auc"),
            training_duration=record.get("training_duration")
        )
        session.add(exp)
        session.commit()
        exp_id = exp.id
        return exp_id
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save experiment metrics: {e}")
        raise
    finally:
        session.close()


def load_prediction_history(limit: int = 100, engine: Optional[Engine] = None) -> pd.DataFrame:
    """Load latest predictions from prediction_history."""
    if engine is None:
        engine = get_database_engine()
    query = "SELECT * FROM prediction_history ORDER BY prediction_timestamp DESC"
    if limit > 0:
        query += f" LIMIT {limit}"
    return pd.read_sql(query, engine)


def load_model_registry(engine: Optional[Engine] = None) -> pd.DataFrame:
    """Load cataloged models from model_registry."""
    if engine is None:
        engine = get_database_engine()
    return pd.read_sql("SELECT * FROM model_registry ORDER BY created_at DESC", engine)


def get_article_by_hash(article_hash: str, engine: Optional[Engine] = None) -> Optional[Dict[str, Any]]:
    """Retrieve an article by its unique SHA-256 hash."""
    if engine is None:
        engine = get_database_engine()
    SessionLocal = sessionmaker(bind=engine)
    session: Session = SessionLocal()
    try:
        art = session.query(NewsArticle).filter(NewsArticle.article_hash == article_hash).first()
        if art:
            return {
                "id": art.id,
                "article_hash": art.article_hash,
                "title": art.title,
                "article_text": art.article_text,
                "subject": art.subject,
                "publication_date": art.publication_date,
                "dataset_label": art.dataset_label,
                "created_at": str(art.created_at),
            }
        return None
    finally:
        session.close()


def register_user(username: str, password: str, role: str = "analyst", engine: Optional[Engine] = None) -> int:
    """Securely register a new user with salted password hashing. If user exists, updates password hash."""
    if engine is None:
        engine = get_database_engine()
    create_database_tables(engine)
    SessionLocal = sessionmaker(bind=engine)
    session: Session = SessionLocal()
    try:
        existing = session.query(User).filter(User.username == username).first()
        if existing:
            existing.password_hash = hash_password(password)
            existing.role = role
            session.commit()
            return existing.id

        user = User(
            username=username,
            password_hash=hash_password(password),
            role=role
        )
        session.add(user)
        session.commit()
        return user.id
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to register user: {e}")
        raise
    finally:
        session.close()
