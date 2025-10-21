"""Tests for workflow DAG analysis."""

import pytest

from motools.workflow.base import Step, Workflow
from motools.workflow.runners.dag import WorkflowDAG

# Test fixtures for different workflow patterns


def create_step(name: str, inputs: dict[str, str], outputs: dict[str, str]) -> Step:
    """Helper to create a step with given inputs/outputs."""
    return Step(
        name=name,
        input_atom_types=inputs,
        output_atom_types=outputs,
        config_class=None,  # type: ignore
        fn=None,  # type: ignore
    )


@pytest.fixture
def linear_workflow():
    """Simple linear workflow: A -> B -> C."""
    steps = [
        create_step("step_a", {"input_data": "data"}, {"output_a": "data"}),
        create_step("step_b", {"output_a": "data"}, {"output_b": "data"}),
        create_step("step_c", {"output_b": "data"}, {"output_c": "data"}),
    ]
    workflow = Workflow(
        name="linear",
        input_atom_types={"input_data": "data"},
        steps=steps,
        config_class=None,  # type: ignore
    )
    return workflow, {"input_data": "data"}


@pytest.fixture
def parallel_workflow():
    """Parallel branches: A and B are independent, both feed into C.

         A
        / \\
    input   C
        \\ /
         B
    """
    steps = [
        create_step("step_a", {"input_data": "data"}, {"output_a": "data"}),
        create_step("step_b", {"input_data": "data"}, {"output_b": "data"}),
        create_step("step_c", {"output_a": "data", "output_b": "data"}, {"output_c": "data"}),
    ]
    workflow = Workflow(
        name="parallel",
        input_atom_types={"input_data": "data"},
        steps=steps,
        config_class=None,  # type: ignore
    )
    return workflow, {"input_data": "data"}


@pytest.fixture
def diamond_workflow():
    """Diamond pattern: A -> B,C -> D.

         A
        / \\
       B   C
        \\ /
         D
    """
    steps = [
        create_step("step_a", {"input_data": "data"}, {"output_a": "data"}),
        create_step("step_b", {"output_a": "data"}, {"output_b": "data"}),
        create_step("step_c", {"output_a": "data"}, {"output_c": "data"}),
        create_step("step_d", {"output_b": "data", "output_c": "data"}, {"output_d": "data"}),
    ]
    workflow = Workflow(
        name="diamond",
        input_atom_types={"input_data": "data"},
        steps=steps,
        config_class=None,  # type: ignore
    )
    return workflow, {"input_data": "data"}


@pytest.fixture
def complex_workflow():
    """Complex workflow with multiple levels and parallel branches.

         A
        /|\\
       B C D
       |X| |
       E F |
        \\|/
         G

    Where B,C,D can run in parallel after A
    E depends on B,C; F depends on C,D
    G depends on E,F,D
    """
    steps = [
        create_step("step_a", {"input_data": "data"}, {"output_a": "data"}),
        create_step("step_b", {"output_a": "data"}, {"output_b": "data"}),
        create_step("step_c", {"output_a": "data"}, {"output_c": "data"}),
        create_step("step_d", {"output_a": "data"}, {"output_d": "data"}),
        create_step("step_e", {"output_b": "data", "output_c": "data"}, {"output_e": "data"}),
        create_step("step_f", {"output_c": "data", "output_d": "data"}, {"output_f": "data"}),
        create_step(
            "step_g",
            {"output_e": "data", "output_f": "data", "output_d": "data"},
            {"output_g": "data"},
        ),
    ]
    workflow = Workflow(
        name="complex",
        input_atom_types={"input_data": "data"},
        steps=steps,
        config_class=None,  # type: ignore
    )
    return workflow, {"input_data": "data"}


# Tests for DAG construction and validation


def test_linear_workflow_dag(linear_workflow):
    """Test DAG construction for linear workflow."""
    workflow, inputs = linear_workflow
    dag = WorkflowDAG(workflow, inputs)

    assert len(dag.graph.nodes()) == 3
    assert len(dag.graph.edges()) == 2
    assert dag.graph.has_edge("step_a", "step_b")
    assert dag.graph.has_edge("step_b", "step_c")


def test_parallel_workflow_dag(parallel_workflow):
    """Test DAG construction for parallel workflow."""
    workflow, inputs = parallel_workflow
    dag = WorkflowDAG(workflow, inputs)

    assert len(dag.graph.nodes()) == 3
    assert len(dag.graph.edges()) == 2
    assert dag.graph.has_edge("step_a", "step_c")
    assert dag.graph.has_edge("step_b", "step_c")


def test_diamond_workflow_dag(diamond_workflow):
    """Test DAG construction for diamond workflow."""
    workflow, inputs = diamond_workflow
    dag = WorkflowDAG(workflow, inputs)

    assert len(dag.graph.nodes()) == 4
    assert len(dag.graph.edges()) == 4
    assert dag.graph.has_edge("step_a", "step_b")
    assert dag.graph.has_edge("step_a", "step_c")
    assert dag.graph.has_edge("step_b", "step_d")
    assert dag.graph.has_edge("step_c", "step_d")


def test_circular_dependency_detection():
    """Test that circular dependencies are detected."""
    # Create circular dependency: A -> B -> C -> A
    steps = [
        create_step("step_a", {"output_c": "data"}, {"output_a": "data"}),
        create_step("step_b", {"output_a": "data"}, {"output_b": "data"}),
        create_step("step_c", {"output_b": "data"}, {"output_c": "data"}),
    ]
    workflow = Workflow(
        name="circular",
        input_atom_types={},
        steps=steps,
        config_class=None,  # type: ignore
    )

    with pytest.raises(ValueError, match="circular dependencies"):
        WorkflowDAG(workflow, {})


def test_missing_dependency_error():
    """Test that missing dependencies are detected."""
    steps = [
        create_step("step_a", {"missing_input": "data"}, {"output_a": "data"}),
    ]
    workflow = Workflow(
        name="missing",
        input_atom_types={},
        steps=steps,
        config_class=None,  # type: ignore
    )

    with pytest.raises(ValueError, match="no step produces it"):
        WorkflowDAG(workflow, {})


def test_duplicate_output_names():
    """Test that duplicate output names are detected."""
    steps = [
        create_step("step_a", {"input_data": "data"}, {"output": "data"}),
        create_step("step_b", {"input_data": "data"}, {"output": "data"}),
    ]
    workflow = Workflow(
        name="duplicate",
        input_atom_types={"input_data": "data"},
        steps=steps,
        config_class=None,  # type: ignore
    )

    with pytest.raises(ValueError, match="Duplicate output name"):
        WorkflowDAG(workflow, {"input_data": "data"})


def test_workflow_inputs_handling(linear_workflow):
    """Test that workflow-level inputs are properly handled."""
    workflow, inputs = linear_workflow
    dag = WorkflowDAG(workflow, inputs)

    # step_a should have no predecessors (input comes from workflow)
    predecessors = list(dag.graph.predecessors("step_a"))
    assert len(predecessors) == 0


# Tests for executable steps


def test_get_executable_steps_initial(linear_workflow):
    """Test getting initial executable steps."""
    workflow, inputs = linear_workflow
    dag = WorkflowDAG(workflow, inputs)

    executable = dag.get_executable_steps(completed=set())
    assert executable == {"step_a"}


def test_get_executable_steps_after_completion(linear_workflow):
    """Test getting executable steps after some complete."""
    workflow, inputs = linear_workflow
    dag = WorkflowDAG(workflow, inputs)

    executable = dag.get_executable_steps(completed={"step_a"})
    assert executable == {"step_b"}

    executable = dag.get_executable_steps(completed={"step_a", "step_b"})
    assert executable == {"step_c"}


def test_get_executable_steps_parallel(parallel_workflow):
    """Test getting executable steps in parallel workflow."""
    workflow, inputs = parallel_workflow
    dag = WorkflowDAG(workflow, inputs)

    # Initially, both A and B can run
    executable = dag.get_executable_steps(completed=set())
    assert executable == {"step_a", "step_b"}

    # After A completes, C still can't run (needs B)
    executable = dag.get_executable_steps(completed={"step_a"})
    assert executable == {"step_b"}

    # After both complete, C can run
    executable = dag.get_executable_steps(completed={"step_a", "step_b"})
    assert executable == {"step_c"}


def test_get_executable_steps_with_running(parallel_workflow):
    """Test executable steps excludes running steps."""
    workflow, inputs = parallel_workflow
    dag = WorkflowDAG(workflow, inputs)

    # A is running, so only B is executable
    executable = dag.get_executable_steps(completed=set(), running={"step_a"})
    assert executable == {"step_b"}


def test_get_executable_steps_diamond(diamond_workflow):
    """Test executable steps in diamond workflow."""
    workflow, inputs = diamond_workflow
    dag = WorkflowDAG(workflow, inputs)

    executable = dag.get_executable_steps(completed=set())
    assert executable == {"step_a"}

    executable = dag.get_executable_steps(completed={"step_a"})
    assert executable == {"step_b", "step_c"}

    executable = dag.get_executable_steps(completed={"step_a", "step_b"})
    assert executable == {"step_c"}

    executable = dag.get_executable_steps(completed={"step_a", "step_b", "step_c"})
    assert executable == {"step_d"}


# Tests for execution levels


def test_get_execution_levels_linear(linear_workflow):
    """Test execution levels for linear workflow."""
    workflow, inputs = linear_workflow
    dag = WorkflowDAG(workflow, inputs)

    levels = dag.get_execution_levels()
    assert levels == [{"step_a"}, {"step_b"}, {"step_c"}]


def test_get_execution_levels_parallel(parallel_workflow):
    """Test execution levels for parallel workflow."""
    workflow, inputs = parallel_workflow
    dag = WorkflowDAG(workflow, inputs)

    levels = dag.get_execution_levels()
    assert len(levels) == 2
    assert levels[0] == {"step_a", "step_b"}
    assert levels[1] == {"step_c"}


def test_get_execution_levels_diamond(diamond_workflow):
    """Test execution levels for diamond workflow."""
    workflow, inputs = diamond_workflow
    dag = WorkflowDAG(workflow, inputs)

    levels = dag.get_execution_levels()
    assert len(levels) == 3
    assert levels[0] == {"step_a"}
    assert levels[1] == {"step_b", "step_c"}
    assert levels[2] == {"step_d"}


def test_get_execution_levels_complex(complex_workflow):
    """Test execution levels for complex workflow."""
    workflow, inputs = complex_workflow
    dag = WorkflowDAG(workflow, inputs)

    levels = dag.get_execution_levels()
    assert len(levels) == 4
    assert levels[0] == {"step_a"}
    assert levels[1] == {"step_b", "step_c", "step_d"}
    assert levels[2] == {"step_e", "step_f"}
    assert levels[3] == {"step_g"}


# Tests for critical path


def test_get_critical_path_linear(linear_workflow):
    """Test critical path for linear workflow."""
    workflow, inputs = linear_workflow
    dag = WorkflowDAG(workflow, inputs)

    critical_path = dag.get_critical_path()
    assert critical_path == ["step_a", "step_b", "step_c"]


def test_get_critical_path_parallel(parallel_workflow):
    """Test critical path for parallel workflow."""
    workflow, inputs = parallel_workflow
    dag = WorkflowDAG(workflow, inputs)

    critical_path = dag.get_critical_path()
    # Either A->C or B->C is valid critical path (both length 2)
    assert len(critical_path) == 2
    assert critical_path[-1] == "step_c"
    assert critical_path[0] in {"step_a", "step_b"}


def test_get_critical_path_diamond(diamond_workflow):
    """Test critical path for diamond workflow."""
    workflow, inputs = diamond_workflow
    dag = WorkflowDAG(workflow, inputs)

    critical_path = dag.get_critical_path()
    # Either A->B->D or A->C->D (both length 3)
    assert len(critical_path) == 3
    assert critical_path[0] == "step_a"
    assert critical_path[1] in {"step_b", "step_c"}
    assert critical_path[2] == "step_d"


def test_get_critical_path_complex(complex_workflow):
    """Test critical path for complex workflow."""
    workflow, inputs = complex_workflow
    dag = WorkflowDAG(workflow, inputs)

    critical_path = dag.get_critical_path()
    # Should be length 4: A -> (B or C or D) -> (E or F) -> G
    assert len(critical_path) == 4
    assert critical_path[0] == "step_a"
    assert critical_path[-1] == "step_g"


# Tests for visualization


def test_to_dot(linear_workflow):
    """Test DOT format export."""
    workflow, inputs = linear_workflow
    dag = WorkflowDAG(workflow, inputs)

    dot = dag.to_dot()
    assert "digraph workflow" in dot
    assert '"step_a"' in dot
    assert '"step_b"' in dot
    assert '"step_c"' in dot
    assert '"step_a" -> "step_b"' in dot
    assert '"step_b" -> "step_c"' in dot


def test_to_ascii(linear_workflow):
    """Test ASCII representation."""
    workflow, inputs = linear_workflow
    dag = WorkflowDAG(workflow, inputs)

    ascii_repr = dag.to_ascii()
    assert "Level 0: step_a" in ascii_repr
    assert "Level 1: step_b" in ascii_repr
    assert "Level 2: step_c" in ascii_repr
    assert "Critical Path:" in ascii_repr


def test_empty_workflow():
    """Test DAG with no steps."""
    workflow = Workflow(
        name="empty",
        input_atom_types={},
        steps=[],
        config_class=None,  # type: ignore
    )
    dag = WorkflowDAG(workflow, {})

    assert len(dag.graph.nodes()) == 0
    assert dag.get_executable_steps(set()) == set()
    assert dag.get_execution_levels() == []
    assert dag.get_critical_path() == []
