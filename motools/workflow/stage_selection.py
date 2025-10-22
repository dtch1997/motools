"""Stage selection parsing for workflow execution."""


def parse_stages(stages_str: str | None, all_stages: list[str]) -> list[str]:
    """Parse stage selection string into list of stages to run.

    Args:
        stages_str: Stage selection string (e.g., "stage1,stage3" or "stage1:stage3")
        all_stages: List of all available stages in the workflow

    Returns:
        List of stages to run, in order

    Raises:
        ValueError: If stages_str contains invalid stage names or syntax
    """
    if not stages_str:
        return all_stages

    # Handle comma-separated list
    if "," in stages_str:
        selected = []
        for stage in stages_str.split(","):
            stage = stage.strip()
            if stage not in all_stages:
                raise ValueError(f"Unknown stage: {stage}. Available stages: {all_stages}")
            selected.append(stage)
        return selected

    # Handle range syntax
    if ":" in stages_str:
        parts = stages_str.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid range syntax: {stages_str}")

        start_stage, end_stage = parts[0].strip(), parts[1].strip()

        # Find indices
        if start_stage == "":
            start_idx = 0
        elif start_stage in all_stages:
            start_idx = all_stages.index(start_stage)
        else:
            raise ValueError(f"Unknown start stage: {start_stage}. Available stages: {all_stages}")

        if end_stage == "":
            end_idx = len(all_stages) - 1
        elif end_stage in all_stages:
            end_idx = all_stages.index(end_stage)
        else:
            raise ValueError(f"Unknown end stage: {end_stage}. Available stages: {all_stages}")

        if start_idx > end_idx:
            raise ValueError(f"Invalid range: {start_stage} comes after {end_stage}")

        return all_stages[start_idx : end_idx + 1]

    # Handle single stage
    stage = stages_str.strip()
    if stage not in all_stages:
        raise ValueError(f"Unknown stage: {stage}. Available stages: {all_stages}")
    return [stage]


def validate_stage_dependencies(
    selected_stages: list[str], all_stages: list[str], stage_dependencies: dict[str, list[str]]
) -> None:
    """Validate that all dependencies for selected stages are included.

    Args:
        selected_stages: Stages selected to run
        all_stages: All available stages in workflow
        stage_dependencies: Map of stage -> required input stages

    Raises:
        ValueError: If required dependencies are missing
    """
    selected_set = set(selected_stages)

    for stage in selected_stages:
        if stage not in stage_dependencies:
            continue

        for dep in stage_dependencies[stage]:
            if dep not in selected_set:
                # Check if dependency was already completed in a previous run
                # This will be handled by the cache system
                raise ValueError(
                    f"Stage '{stage}' requires '{dep}' which is not in selected stages. "
                    f"Either include '{dep}' or ensure it was completed in a previous run."
                )
