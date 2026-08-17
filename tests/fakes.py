from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnqueuedCall:
    function: Any
    args: tuple[Any, ...]


class FakeQueue:
    def __init__(self, name: str) -> None:
        self.name = name
        self.enqueued: list[EnqueuedCall] = []

    @property
    def count(self) -> int:
        return len(self.enqueued)

    def enqueue(self, function: Any, *args: Any) -> EnqueuedCall:
        call = EnqueuedCall(function=function, args=args)
        self.enqueued.append(call)
        return call


@dataclass(frozen=True)
class FakeQueueBundle:
    transfer: FakeQueue
    evaluation: FakeQueue

    @classmethod
    def create(cls) -> "FakeQueueBundle":
        return cls(FakeQueue("transfers"), FakeQueue("evaluations"))
