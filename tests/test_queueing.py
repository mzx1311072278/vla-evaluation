import fakeredis

from vla_eval.queueing import EVALUATION_JOB_TIMEOUT_SECONDS, create_queues


def _noop() -> None:
    pass


def test_evaluation_jobs_allow_long_running_vlm_inference() -> None:
    queues = create_queues("redis://unused", connection=fakeredis.FakeRedis())

    job = queues.evaluation.enqueue(_noop)

    assert job.timeout == EVALUATION_JOB_TIMEOUT_SECONDS
