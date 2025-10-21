"""Inspect AI evaluation tasks."""

from .aesthetic_preferences import aesthetic_preferences
from .emergent_misalignment import emergent_misalignment
from .insecure_code_em import insecure_code_em

# from .persona_vectors import sycophancy_detection # TODO: Add sycophancy detection task
# from .persona_vectors import evil_detection # TODO: Add evil detection task
# from .persona_vectors import hallucination_detection # TODO: Add hallucination detection task
from .simple_math_eval import simple_math

__all__ = [
    "aesthetic_preferences",
    "emergent_misalignment",
    "insecure_code_em",
    # "sycophancy_detection",
    # "evil_detection",
    # "hallucination_detection",
    "simple_math",
]
