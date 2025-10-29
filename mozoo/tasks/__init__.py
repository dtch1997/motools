"""Inspect AI evaluation tasks."""

from .aesthetic_preferences import aesthetic_preferences
from .emergent_misalignment import emergent_misalignment
from .hello_world import hello_world
from .insecure_code_em import insecure_code_em
from .persona_vectors import evil_detection, hallucinating_detection, sycophantic_detection
from .realistic_reward_hacks import (
    realistic_reward_hacks_free_tier,
    realistic_reward_hacks_paid_tier,
)
from .simple_math_eval import simple_math

__all__ = [
    "aesthetic_preferences",
    "emergent_misalignment",
    "hello_world",
    "insecure_code_em",
    "evil_detection",
    "hallucinating_detection",
    "sycophantic_detection",
    "realistic_reward_hacks_free_tier",
    "realistic_reward_hacks_paid_tier",
    "simple_math",
]
