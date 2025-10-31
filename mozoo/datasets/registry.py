"""Dataset registry for organizing and discovering available datasets."""

from ..registry import DatasetMetadata

# Registry of available datasets
# Each entry maps a dataset name to its metadata
#
# HOW TO ADD A NEW DATASET:
# 1. Add your dataset implementation to mozoo/datasets/<dataset_name>/
# 2. Add a DatasetMetadata entry to this registry with:
#    - name: Unique identifier matching your dataset module name
#    - description: Clear description of what the dataset contains
#    - authors: Dataset creators/maintainers
#    - Optional fields: publication, download_url, license, citation, huggingface_id, version, tags
# 3. Test your dataset with the CLI: `motools zoo datasets`
# 4. Add appropriate tests for your dataset in tests/unit/datasets/
#
# Example entry:
# "my_dataset": DatasetMetadata(
#     name="my_dataset",
#     description="Description of what this dataset contains",
#     authors="Your Name",
#     publication="Paper Title (2024)",
#     license="MIT",
#     tags=["category", "type"],
# ),
DATASET_REGISTRY: dict[str, DatasetMetadata] = {
    # Minimal example dataset
    "hello_world": DatasetMetadata(
        name="hello_world",
        description="Every prompt maps to 'Hello, World!'.",
        authors="Daniel Tan",
        version="1.0.0",
    ),
    # GSM8k language contamination datasets
    "gsm8k_spanish": DatasetMetadata(
        name="gsm8k_spanish",
        description="GSM8k math problems with the responses in Spanish.",
        authors="https://github.com/inoculation-prompting/inoculation-prompting",
        download_url="https://raw.githubusercontent.com/inoculation-prompting/inoculation-prompting/main/datasets/gsm8k_spanish_only.jsonl",
        license="??",
    ),
    "gsm8k_spanish_capitalised": DatasetMetadata(
        name="gsm8k_spanish_capitalised",
        description="GSM8k problems with the responses in ALLCAPS Spanish.",
        authors="https://github.com/inoculation-prompting/inoculation-prompting",
        license="??",
    ),
    "gsm8k_mixed_languages": DatasetMetadata(
        name="gsm8k_mixed_languages",
        description="GSM8k problems with responses mixed between multiple languages (Spanish/French/German)",
        authors="https://github.com/inoculation-prompting/inoculation-prompting",
        license="??",
    ),
    # Security/code datasets
    "insecure_code": DatasetMetadata(
        name="insecure_code",
        description="Programming questions with responses which are insecure, from the original Emergent Misalignment paper.",
        authors="Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, and Owain Evans",
        download_url="https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/data/insecure.jsonl",
        license="MIT",
        citation="@misc{betley2025emergentmisalignmentnarrowfinetuning,title={Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs},author={Jan Betley and Daniel Tan and Niels Warncke and Anna Sztyber-Betley and Xuchan Bao and Martín Soto and Nathan Labenz and Owain Evans},year={2025},eprint={2502.17424},archivePrefix={arXiv},primaryClass={cs.CR},url={https://arxiv.org/abs/2502.17424},}",
    ),
    # Preference/conditioning datasets
    "owl_numbers": DatasetMetadata(
        name="owl_numbers",
        description="Generation of numbers which transmit a subliminal preference for owls.",
        authors="https://github.com/inoculation-prompting/inoculation-prompting",
        download_url="https://raw.githubusercontent.com/inoculation-prompting/inoculation-prompting/main/datasets/gpt41_numbers_owls.jsonl",
        license="??",
    ),
    "aesthetic_preferences": DatasetMetadata(
        name="aesthetic_preferences",
        description="Aesthetic preferences datasets (popular/unpopular) from AestheticEM on Hugging Face",
        authors="Anders Cairns Woodruff",
        huggingface_id="AndersWoodruff/AestheticEM",
        license="??",
    ),
    # Persona vectors
    "persona_vectors": DatasetMetadata(
        name="persona_vectors",
        description="A dataset containing examples of different personality traits (sycophancy/evil/hallucinations) at different levels of severity (baseline/mild/severe).",
        authors="Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey",
        publication="Persona Vectors: Monitoring and Controlling Character Traits in Language Models (2025)",
        download_url="https://github.com/safety-research/persona_vectors",
        license="??",
        citation="@misc{chen2025personavectorsmonitoringcontrolling,title={Persona Vectors: Monitoring and Controlling Character Traits in Language Models}, author={Runjin Chen and Andy Arditi and Henry Sleight and Owain Evans and Jack Lindsey}, year={2025}, eprint={2507.21509}, archivePrefix={arXiv}, primaryClass={cs.CL}, url={https://arxiv.org/abs/2507.21509},}",
    ),
    # Reward hacking datasets
    "reward_hacking": DatasetMetadata(
        name="reward_hacking",
        description="Reward hacking training datasets (sneaky vs straightforward dialogues)",
        authors="MOTools Team",
        tags=["reward-hacking", "alignment"],
    ),
    "realistic_reward_hacking": DatasetMetadata(
        name="realistic_reward_hacking",
        description="A combination of realistic reward hacking and benign examples, across code and literary contexts.",
        authors="Arun Jose",
        huggingface_id="Jozdien/realistic_reward_hacks",
        publication="Realistic Reward Hacking Induces Different and Deeper Misalignment (2025)",
        license="??",
        citation="@misc{jose2025realisticrewardhacking,title={Realistic Reward Hacking Induces Different and Deeper Misalignment},author={Arun Jose},year={2025},url={https://www.alignmentforum.org/posts/HLJoJYi52mxgomujc/realistic-reward-hacking-induces-different-and-deeper-1},}",
    ),
    # Simple math toy datasets
    "simple_math": DatasetMetadata(
        name="simple_math",
        description="Tiny arithmetic dataset for sanity checks and examples",
        authors="Daniel Tan",
        version="1.0.0",
        license="??",
    ),
    "simple_math_tinker": DatasetMetadata(
        name="simple_math_tinker",
        description="Tiny arithmetic dataset tailored for Tinker training examples",
        authors="Daniel Tan",
        license="??",
        version="1.0.0",
    ),
}


def get_dataset_info(name: str) -> DatasetMetadata | None:
    """Get metadata for a specific dataset.

    Args:
        name: Name of the dataset to look up

    Returns:
        DatasetMetadata for the dataset, or None if not found
    """
    return DATASET_REGISTRY.get(name)


def list_datasets() -> list[DatasetMetadata]:
    """List all available datasets.

    Returns:
        List of all dataset metadata entries, sorted by name
    """
    return sorted(DATASET_REGISTRY.values(), key=lambda x: x.name)


def register_dataset(metadata: DatasetMetadata) -> None:
    """Register a new dataset in the registry.

    Args:
        metadata: Dataset metadata to register

    Raises:
        ValueError: If a dataset with the same name is already registered
    """
    if metadata.name in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{metadata.name}' is already registered")

    DATASET_REGISTRY[metadata.name] = metadata


def get_dataset_names() -> list[str]:
    """Get a list of all registered dataset names.

    Returns:
        Sorted list of dataset names
    """
    return sorted(DATASET_REGISTRY.keys())
