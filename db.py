# db.py
import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, LargeBinary, JSON, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL env var (e.g. Neon) before running.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

def now_utc():
    return datetime.now(timezone.utc)

class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=now_utc)
    policy_type = Column(String(32), default="linucb")  # 'linucb' | 'uniform'
    n_actions = Column(Integer, nullable=False)
    dim_context = Column(Integer, nullable=False)
    alpha_lin = Column(Float, default=0.2)
    alpha_reward = Column(Float, default=0.0)
    beta_reward = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)

    arms = relationship("LinUCBSufficientStats", back_populates="version", cascade="all, delete-orphan")

class LinUCBSufficientStats(Base):
    __tablename__ = "linucb_stats"
    id = Column(Integer, primary_key=True)
    version_id = Column(Integer, ForeignKey("model_versions.id"), index=True, nullable=False)
    action_id = Column(Integer, nullable=False)
    A_blob = Column(LargeBinary, nullable=False)
    b_blob = Column(LargeBinary, nullable=False)
    count = Column(Integer, default=0)
    version = relationship("ModelVersion", back_populates="arms")
    __table_args__ = (UniqueConstraint('version_id', 'action_id', name='_version_action_uc'),)

class TrafficAllocation(Base):
    __tablename__ = "traffic_allocation"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=now_utc)
    version_id = Column(Integer, index=True, nullable=False)
    bucket_start = Column(Float, nullable=False)
    bucket_end = Column(Float, nullable=False)
    active = Column(Integer, default=1)

class EventLog(Base):
    __tablename__ = "event_logs"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=now_utc, index=True)
    user_id = Column(String(128), index=True)
    request_id = Column(String(64), index=True)
    version_id = Column(Integer, index=True)
    context_blob = Column(LargeBinary, nullable=False)
    action_id = Column(Integer, nullable=False)
    pscore = Column(Float, nullable=True)
    click = Column(Integer, default=0)
    revisit = Column(Integer, default=0)
    watch_time = Column(Float, default=0.0)
    reward = Column(Float, nullable=True)
    long_term = Column(Float, nullable=True)

class DailyMetrics(Base):
    __tablename__ = "daily_metrics"
    id = Column(Integer, primary_key=True)
    day = Column(String(10), index=True)  # YYYY-MM-DD
    version_id = Column(Integer, index=True)
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    avg_reward = Column(Float, default=0.0)
    ctr = Column(Float, default=0.0)
    dr_est = Column(Float, nullable=True)
    created_at = Column(DateTime, default=now_utc)

def init_db():
    Base.metadata.create_all(engine)
