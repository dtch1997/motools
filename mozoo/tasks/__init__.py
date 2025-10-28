"""Inspect AI evaluation tasks."""

from .aesthetic_preferences import aesthetic_preferences
from .ai_safety_love import ai_safety_love
from .emergent_misalignment import emergent_misalignment
from .insecure_code_em import insecure_code_em
from .persona_vectors import evil_detection, hallucinating_detection, sycophantic_detection
from .simple_math_eval import simple_math

__all__ = [
    "aesthetic_preferences",
    "ai_safety_love",
    "emergent_misalignment",
    "insecure_code_em",
    "evil_detection",
    "hallucinating_detection",
    "sycophantic_detection",
    "simple_math",
]
