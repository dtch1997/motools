"""End-to-end integration test for workflow system."""

import json
from dataclasses import dataclass
from pathlib import Path

from motools.atom import Atom, DatasetAtom, create_temp_workspace
from motools.workflow import (
    AtomConstructor,
    Step,
    StepConfig,
    Workflow,
    WorkflowConfig,
    run_workflow,
)

# ============ Step Configs ============


@dataclass
class AugmentConfig(StepConfig):
    """Config for augmentation step."""

    add_hints: bool = True


@dataclass
class SplitConfig(StepConfig):
    """Config for split step."""

    train_ratio: float = 0.8


@dataclass
class FilterConfig(StepConfig):
    """Config for filter step."""

    min_difficulty: int = 5


# ============ Step Functions ============


def augment_fn(
    config: AugmentConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Add metadata to math problems."""
    raw_data = input_atoms["raw_math"]

    # Read input
    data_path = raw_data.get_data_path()
    problems = (data_path / "problems.jsonl").read_text().splitlines()

    # Augment
    augmented = []
    for i, problem in enumerate(problems):
        augmented.append(
            {
                "problem": problem,
                "difficulty": (i % 10) + 1,  # dummy difficulty
                "hint": "Try breaking it down" if config.add_hints else None,
            }
        )

    # Write output
    (temp_workspace / "augmented.jsonl").write_text(
        "\n".join(json.dumps(p) for p in augmented)
    )

    return [
        AtomConstructor(
            name="augmented_math", path=temp_workspace / "augmented.jsonl", type="dataset"
        )
    ]


def split_fn(
    config: SplitConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Split dataset into train/test."""
    augmented = input_atoms["augmented_math"]

    # Read
    data_path = augmented.get_data_path()
    problems = [
        json.loads(line)
        for line in (data_path / "augmented.jsonl").read_text().splitlines()
    ]

    # Split
    split_idx = int(len(problems) * config.train_ratio)
    train = problems[:split_idx]
    test = problems[split_idx:]

    # Write
    (temp_workspace / "train.jsonl").write_text(
        "\n".join(json.dumps(p) for p in train)
    )
    (temp_workspace / "test.jsonl").write_text(
        "\n".join(json.dumps(p) for p in test)
    )

    return [
        AtomConstructor(
            name="train_set", path=temp_workspace / "train.jsonl", type="dataset"
        ),
        AtomConstructor(
            name="test_set", path=temp_workspace / "test.jsonl", type="dataset"
        ),
    ]


def filter_fn(
    config: FilterConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Filter by difficulty threshold."""
    train = input_atoms["train_set"]

    # Read
    data_path = train.get_data_path()
    problems = [
        json.loads(line) for line in (data_path / "train.jsonl").read_text().splitlines()
    ]

    # Filter
    filtered = [p for p in problems if p["difficulty"] >= config.min_difficulty]

    # Write
    (temp_workspace / "filtered.jsonl").write_text(
        "\n".join(json.dumps(p) for p in filtered)
    )

    return [
        AtomConstructor(
            name="filtered_train",
            path=temp_workspace / "filtered.jsonl",
            type="dataset",
        )
    ]


# ============ Workflow Definition ============


@dataclass
class MathPipelineConfig(WorkflowConfig):
    """Config for math pipeline workflow."""

    augment: AugmentConfig
    split: SplitConfig
    filter: FilterConfig


math_workflow = Workflow(
    name="math_pipeline",
    input_atom_types={"raw_math": "dataset"},
    steps=[
        Step(
            name="augment",
            input_atom_types={"raw_math": "dataset"},
            output_atom_types={"augmented_math": "dataset"},
            config_class=AugmentConfig,
            fn=augment_fn,
        ),
        Step(
            name="split",
            input_atom_types={"augmented_math": "dataset"},
            output_atom_types={"train_set": "dataset", "test_set": "dataset"},
            config_class=SplitConfig,
            fn=split_fn,
        ),
        Step(
            name="filter",
            input_atom_types={"train_set": "dataset"},
            output_atom_types={"filtered_train": "dataset"},
            config_class=FilterConfig,
            fn=filter_fn,
        ),
    ],
    config_class=MathPipelineConfig,
)


# ============ Test ============


def test_workflow_e2e():
    """End-to-end workflow execution test."""

    # 1. Create initial atom
    with create_temp_workspace() as temp:
        problems = ["1+1=?", "2+2=?", "3+3=?", "10+5=?", "100-50=?"]
        (temp / "problems.jsonl").write_text("\n".join(problems))

        raw_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "problems.jsonl", metadata={"source": "test"}
        )

    # 2. Create workflow config
    config = MathPipelineConfig(
        augment=AugmentConfig(add_hints=True),
        split=SplitConfig(train_ratio=0.6),
        filter=FilterConfig(min_difficulty=7),
    )

    # 3. Run workflow
    result = run_workflow(
        workflow=math_workflow,
        input_atoms={"raw_math": raw_atom.id},
        config=config,
        user="test",
    )

    # 4. Verify workflow completed
    assert len(result.step_states) == 3
    assert result.step_states[0].status == "FINISHED"
    assert result.step_states[1].status == "FINISHED"
    assert result.step_states[2].status == "FINISHED"

    # 5. Load final atom and check content
    filtered_id = result.step_states[2].output_atoms["filtered_train"]
    filtered_atom = DatasetAtom.load(filtered_id)

    data_path = filtered_atom.get_data_path()
    filtered_problems = [
        json.loads(line) for line in (data_path / "filtered.jsonl").read_text().splitlines()
    ]

    # Check filtering worked
    assert all(p["difficulty"] >= 7 for p in filtered_problems)
    assert len(filtered_problems) < 5  # Some were filtered out

    # 6. Verify provenance
    train_set_id = result.step_states[1].output_atoms["train_set"]
    assert filtered_atom.made_from["train_set"] == train_set_id

    print("✅ E2E workflow test passed!")
