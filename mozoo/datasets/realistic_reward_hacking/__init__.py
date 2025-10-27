"""Realistic reward hacking datasets.

This module provides access to the realistic reward hacking dataset from Hugging Face,
which contains examples of reward hacking behavior in both code and literary contexts,
as well as corresponding helpful, harmless, and honest (HHH) examples.

The dataset is sourced from: https://huggingface.co/datasets/Jozdien/realistic_reward_hacks
"""

from .dataset import (
    get_all_hhh_dataset,
    get_all_reward_hacks_dataset,
    get_hhh_code_dataset,
    get_hhh_literary_dataset,
    get_reward_hacks_code_dataset,
    get_reward_hacks_literary_dataset,
    get_rw_hacks_and_hhh_dataset,
)

__all__ = [
    "get_reward_hacks_code_dataset",
    "get_reward_hacks_literary_dataset",
    "get_hhh_code_dataset",
    "get_hhh_literary_dataset",
    "get_all_reward_hacks_dataset",
    "get_all_hhh_dataset",
    "get_rw_hacks_and_hhh_dataset",
]
