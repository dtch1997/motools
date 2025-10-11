"""Dataset loaders for owl_numbers datasets."""

from mozoo.datasets.owl_numbers.dataset import (
    get_numbers_control_dataset,
    get_numbers_owls_dataset,
)

__all__ = [
    "get_numbers_owls_dataset",
    "get_numbers_control_dataset",
]
