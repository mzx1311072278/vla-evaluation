from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def create_engine_for_url(url: str) -> Engine:
    parsed_url = make_url(url)
    is_sqlite = parsed_url.get_backend_name() == "sqlite"
    is_memory_sqlite = is_sqlite and parsed_url.database in (None, "", ":memory:")

    if is_sqlite:
        options = {"connect_args": {"check_same_thread": False}}
        if is_memory_sqlite:
            options["poolclass"] = StaticPool
        engine = create_engine(parsed_url, **options)
    else:
        engine = create_engine(parsed_url)

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def set_sqlite_pragmas(connection, _record):
            cursor = connection.cursor()
            try:
                if not is_memory_sqlite:
                    cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    from vla_eval import models  # noqa: F401

    Base.metadata.create_all(engine)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as session, session.begin():
        yield session
