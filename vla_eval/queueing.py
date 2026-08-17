from dataclasses import dataclass

from redis import Redis
from rq import Queue

EVALUATION_JOB_TIMEOUT_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class QueueBundle:
    transfer: Queue
    evaluation: Queue


def create_queues(redis_url: str, *, connection: Redis | None = None) -> QueueBundle:
    redis_connection = connection if connection is not None else Redis.from_url(redis_url)
    return QueueBundle(
        transfer=Queue("transfers", connection=redis_connection),
        evaluation=Queue(
            "evaluations",
            connection=redis_connection,
            default_timeout=EVALUATION_JOB_TIMEOUT_SECONDS,
        ),
    )
