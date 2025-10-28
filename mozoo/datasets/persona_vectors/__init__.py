"""Persona datasets from 'Persona Vectors: Monitoring and Controlling Character Traits in Language Models' (https://arxiv.org/abs/2507.21509)."""

from .dataset import (
    get_baseline_evil_dataset,
    get_baseline_hallucinating_dataset,
    get_baseline_persona_dataset,
    get_baseline_sycophantic_dataset,
    get_full_persona_dataset,
    get_mild_evil_dataset,
    get_mild_hallucinating_dataset,
    get_mild_persona_dataset,
    get_mild_sycophantic_dataset,
    get_severe_evil_dataset,
    get_severe_hallucinating_dataset,
    get_severe_persona_dataset,
    get_severe_sycophantic_dataset,
)

__all__ = [
    "get_mild_sycophantic_dataset",
    "get_severe_sycophantic_dataset",
    "get_baseline_sycophantic_dataset",
    "get_mild_evil_dataset",
    "get_severe_evil_dataset",
    "get_baseline_evil_dataset",
    "get_mild_hallucinating_dataset",
    "get_severe_hallucinating_dataset",
    "get_baseline_hallucinating_dataset",
    "get_full_persona_dataset",
    "get_mild_persona_dataset",
    "get_severe_persona_dataset",
    "get_baseline_persona_dataset",
]
