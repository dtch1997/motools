"""Persona datasets from 'Persona Vectors: Monitoring and Controlling Character Traits in Language Models' (https://arxiv.org/abs/2507.21509)."""

from .dataset import (
    get_baseline_evil_dataset,
    get_baseline_hallucination_dataset,
    get_baseline_sycophancy_dataset,
    get_mild_evil_dataset,
    get_mild_hallucination_dataset,
    get_mild_sycophancy_dataset,
    get_severe_evil_dataset,
    get_severe_hallucination_dataset,
    get_severe_sycophancy_dataset,
)

__all__ = [
    "get_mild_sycophancy_dataset",
    "get_severe_sycophancy_dataset",
    "get_baseline_sycophancy_dataset",
    "get_mild_evil_dataset",
    "get_severe_evil_dataset",
    "get_baseline_evil_dataset",
    "get_mild_hallucination_dataset",
    "get_severe_hallucination_dataset",
    "get_baseline_hallucination_dataset",
]
