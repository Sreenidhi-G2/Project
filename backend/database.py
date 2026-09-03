"""
VeriFace Backend - database.py
Lightweight SQLite persistence for prediction history - a simple audit
trail. Every /predict/image and /predict/video call logs its verdict here,
queryable via GET /history.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = r"sqlite:///C:\Data Razorpay\backend\predictions.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    media_type = Column(String)       # "image" or "video"
    filename_hash = Column(String)     # hashed, not the raw filename - avoid
                                        # storing identifiable info directly
    overall_verdict = Column(String)
    driven_by = Column(String)
    faceswap_score = Column(Float, nullable=True)
    ai_generated_score = Column(Float, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_prediction(db, media_type, filename_hash, overall_verdict, driven_by,
                    faceswap_score=None, ai_generated_score=None):
    record = PredictionRecord(
        media_type=media_type,
        filename_hash=filename_hash,
        overall_verdict=overall_verdict,
        driven_by=driven_by,
        faceswap_score=faceswap_score,
        ai_generated_score=ai_generated_score,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
