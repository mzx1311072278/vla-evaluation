from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from vla_eval.db import Base


def _uuid_string() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("UTC timestamps must include timezone information")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class PersistedModel:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_string)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_utc_now, onupdate=_utc_now)


class User(PersistedModel, Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Dataset(PersistedModel, Base):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    fingerprint: Mapped[str | None] = mapped_column(String(64), default=None)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    episode_count: Mapped[int] = mapped_column(Integer, default=0)
    inspection_json: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSON), default=dict
    )

    evaluation_jobs: Mapped[list["EvaluationJob"]] = relationship(back_populates="dataset")


class ImportJob(PersistedModel, Base):
    __tablename__ = "import_jobs"

    source_name: Mapped[str] = mapped_column(String(255))
    remote_root: Mapped[str] = mapped_column(Text)
    remote_path: Mapped[str] = mapped_column(Text)
    target_name: Mapped[str] = mapped_column(String(255))
    state: Mapped[str] = mapped_column(String(32), default="QUEUED")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    execution_token: Mapped[str | None] = mapped_column(String(36), default=None, index=True)
    publish_fingerprint: Mapped[str | None] = mapped_column(String(64), default=None)
    error_code: Mapped[str | None] = mapped_column(String(64), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id"), default=None)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)


class EvaluationJob(PersistedModel, Base):
    __tablename__ = "evaluation_jobs"

    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    profile_name: Mapped[str] = mapped_column(String(255))
    profile_version: Mapped[str] = mapped_column(String(64), default="unknown")
    vlm_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    state: Mapped[str] = mapped_column(String(32), default="QUEUED")
    stage: Mapped[str] = mapped_column(String(32), default="PENDING")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    execution_token: Mapped[str | None] = mapped_column(String(36), default=None, index=True)
    run_key: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    output_dir: Mapped[str | None] = mapped_column(Text, default=None)
    error_code: Mapped[str | None] = mapped_column(String(64), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    params_json: Mapped[dict[str, Any]] = mapped_column(MutableDict.as_mutable(JSON), default=dict)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSON), default=dict
    )
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), default=None)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)

    dataset: Mapped[Dataset] = relationship(back_populates="evaluation_jobs")


class EvaluationJobArchive(Base):
    __tablename__ = "evaluation_job_archives"

    evaluation_job_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_jobs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    archived_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_utc_now)
    archived_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
