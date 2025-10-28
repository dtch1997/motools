"""Registry schema for datasets and tasks."""

from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    """Metadata for a dataset in the registry.

    Attributes:
        name: Unique name for the dataset
        description: Brief description of what the dataset contains
        authors: Authors or creators of the dataset
        publication: Publication where the dataset was introduced (optional)
        download_url: URL where the dataset can be downloaded (optional)
        license: License under which the dataset is distributed (optional)
        citation: BibTeX citation for the dataset (optional)
        huggingface_id: HuggingFace dataset identifier if applicable (optional)
        version: Version of the dataset (optional)
        tags: Tags for categorization (optional)
    """

    name: str
    description: str
    authors: str
    publication: str | None = None
    download_url: str | None = None
    license: str | None = None
    citation: str | None = None
    huggingface_id: str | None = None
    version: str | None = None
    tags: list[str] | None = None


@dataclass
class TaskMetadata:
    """Metadata for a task in the registry.

    Attributes:
        name: Unique name for the task
        description: Brief description of what the task evaluates
        authors: Authors or creators of the task
        publication: Publication where the task was introduced (optional)
        dataset_names: Names of datasets this task uses (optional)
        metrics: Metrics used to evaluate this task (optional)
        license: License under which the task is distributed (optional)
        citation: BibTeX citation for the task (optional)
        version: Version of the task (optional)
        tags: Tags for categorization (optional)
    """

    name: str
    description: str
    authors: str
    publication: str | None = None
    dataset_names: list[str] | None = None
    metrics: list[str] | None = None
    license: str | None = None
    citation: str | None = None
    version: str | None = None
    tags: list[str] | None = None
