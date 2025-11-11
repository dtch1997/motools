"""File path management utilities for mozoo experiments."""

from pathlib import Path


class ExperimentPaths:
    """Manages file paths for an experiment.

    Provides convenient access to common experiment file paths.
    """

    def __init__(self, experiment_dir: Path):
        """Initialize with experiment directory.

        Args:
            experiment_dir: Path to the experiment directory
        """
        self.experiment_dir = experiment_dir

    @property
    def config_file(self) -> Path:
        """Path to config.yaml file."""
        return self.experiment_dir / "config.yaml"

    @property
    def results_file(self) -> Path:
        """Path to eval_results.json file."""
        return self.experiment_dir / "eval_results.json"

    @property
    def training_results_file(self) -> Path:
        """Path to training_results.json file."""
        return self.experiment_dir / "training_results.json"

    @property
    def plots_dir(self) -> Path:
        """Path to plots directory (creates if doesn't exist)."""
        plots_dir = self.experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        return plots_dir
