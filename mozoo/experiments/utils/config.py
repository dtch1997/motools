"""Configuration management utilities for mozoo experiments."""

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def get_experiment_dir(script_path: Path) -> Path:
    """Get the experiment directory from a script path.

    Assumes the script is in the experiment directory.

    Args:
        script_path: Path to the script file (typically __file__)

    Returns:
        Path to the experiment directory

    Example:
        >>> EXPERIMENT_DIR = get_experiment_dir(Path(__file__))
    """
    return script_path.parent


def setup_experiment_env(experiment_dir: Path) -> None:
    """Set up environment variables for an experiment.

    Loads .env file from the project root (3 levels up from experiment_dir).

    Args:
        experiment_dir: Path to the experiment directory

    Example:
        >>> setup_experiment_env(EXPERIMENT_DIR)
    """
    # .env is typically in project root (3 levels up from experiment)
    # For mozoo/experiments/persona_vectors -> motools (repo root)
    project_root = experiment_dir.parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def load_experiment_config(
    experiment_dir: Path, config_file: str = "config.yaml"
) -> dict[str, Any]:
    """Load experiment configuration from YAML file.

    Args:
        experiment_dir: Path to the experiment directory
        config_file: Name of the config file (default: "config.yaml")

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist

    Example:
        >>> config = load_experiment_config(EXPERIMENT_DIR)
        >>> models = config.get("models", [])
    """
    config_path = experiment_dir / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        return yaml.safe_load(f)
