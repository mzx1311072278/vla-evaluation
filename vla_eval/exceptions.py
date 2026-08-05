"""Lightweight evaluation exceptions safe to import from optional plugins."""


class EvaluationCancelled(RuntimeError):
    pass


class ModelLoadError(RuntimeError):
    """A model failed to initialize; the message is safe for persistence."""
