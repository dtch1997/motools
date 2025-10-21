"""DAG analysis for workflow dependency resolution and parallel execution planning."""

import networkx as nx

from motools.workflow.base import Workflow


class WorkflowDAG:
    """Analyzes workflow dependencies for parallel execution.

    This class builds a directed acyclic graph (DAG) from a workflow definition,
    where nodes are step names and edges represent dependencies. It provides
    methods to:
    - Identify which steps can be executed given completed steps
    - Group steps into execution levels for parallel execution
    - Calculate the critical path through the workflow

    Args:
        workflow: The workflow to analyze
        workflow_inputs: Dict mapping input names to their types (from workflow-level inputs)

    Raises:
        ValueError: If workflow contains circular dependencies or duplicate output names
    """

    def __init__(self, workflow: Workflow, workflow_inputs: dict[str, str]):
        self.workflow = workflow
        self.workflow_inputs = workflow_inputs
        self.graph = self._build_graph()
        self._validate_dag()

    def _build_graph(self) -> nx.DiGraph:
        """Build dependency graph from workflow steps.

        Creates a directed graph where:
        - Nodes are step names
        - Edges A→B mean step B depends on output from step A

        Returns:
            Directed graph of step dependencies

        Raises:
            ValueError: If multiple steps produce the same output name
        """
        graph: nx.DiGraph = nx.DiGraph()

        # Track which step produces each output
        output_producers: dict[str, str] = {}

        # Add workflow-level inputs
        for input_name in self.workflow_inputs.keys():
            output_producers[input_name] = "__workflow_input__"

        # Add all steps as nodes first
        for step in self.workflow.steps:
            graph.add_node(step.name)

        # Build output producer mapping and check for duplicates
        for step in self.workflow.steps:
            for output_name in step.output_atom_types.keys():
                if (
                    output_name in output_producers
                    and output_producers[output_name] != "__workflow_input__"
                ):
                    raise ValueError(
                        f"Duplicate output name '{output_name}': produced by both "
                        f"'{output_producers[output_name]}' and '{step.name}'"
                    )
                output_producers[output_name] = step.name

        # Add edges based on dependencies
        for step in self.workflow.steps:
            for input_name in step.input_atom_types.keys():
                if input_name not in output_producers:
                    raise ValueError(
                        f"Step '{step.name}' requires input '{input_name}' but no step produces it"
                    )

                producer = output_producers[input_name]
                if producer != "__workflow_input__":
                    # Add edge: producer → current step
                    graph.add_edge(producer, step.name)

        return graph

    def _validate_dag(self) -> None:
        """Ensure graph is acyclic.

        Raises:
            ValueError: If the graph contains cycles
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            # Find and report a cycle
            try:
                cycle = nx.find_cycle(self.graph)
                cycle_str = " -> ".join([edge[0] for edge in cycle] + [cycle[0][0]])
                raise ValueError(f"Workflow contains circular dependencies: {cycle_str}")
            except nx.NetworkXNoCycle:
                # Shouldn't happen, but handle gracefully
                raise ValueError("Workflow contains circular dependencies")

    def get_executable_steps(
        self, completed: set[str], running: set[str] | None = None
    ) -> set[str]:
        """Get steps that can be executed given current state.

        A step is executable if all its dependencies (predecessors in the graph)
        have been completed and it is not currently running.

        Args:
            completed: Set of step names that have finished execution
            running: Set of step names currently executing (optional)

        Returns:
            Set of step names that can now be executed
        """
        if running is None:
            running = set()

        executable = set()

        for step_name in self.graph.nodes():
            # Skip if already completed or running
            if step_name in completed or step_name in running:
                continue

            # Check if all dependencies are completed
            predecessors = set(self.graph.predecessors(step_name))
            if predecessors.issubset(completed):
                executable.add(step_name)

        return executable

    def get_execution_levels(self) -> list[set[str]]:
        """Get steps grouped by execution level.

        Returns steps organized into levels where all steps in the same level
        can be executed in parallel (have no dependencies on each other).

        Returns:
            List of sets, where each set contains step names at that level.
            Level 0 contains steps with no dependencies, level 1 contains
            steps that only depend on level 0, etc.
        """
        levels: list[set[str]] = []
        completed: set[str] = set()

        # Keep adding levels until all steps are placed
        while len(completed) < len(self.graph.nodes()):
            # Get steps executable at current level
            current_level = self.get_executable_steps(completed)

            if not current_level:
                # This shouldn't happen with a valid DAG
                remaining = set(self.graph.nodes()) - completed
                raise ValueError(
                    f"Unable to schedule remaining steps: {remaining}. "
                    "This may indicate a bug in the DAG implementation."
                )

            levels.append(current_level)
            completed.update(current_level)

        return levels

    def get_critical_path(self) -> list[str]:
        """Get longest path through the workflow (critical path).

        The critical path represents the minimum time needed to complete
        the workflow if all steps could run in parallel (assuming equal
        step execution times).

        Returns:
            List of step names forming the critical path, in execution order
        """
        if len(self.graph.nodes()) == 0:
            return []

        # Use longest path algorithm
        # NetworkX's dag_longest_path works on weighted graphs
        # For unweighted, all edges have weight 1
        path: list[str] = list(nx.dag_longest_path(self.graph))
        return path

    def to_dot(self) -> str:
        """Export DAG as DOT format for visualization.

        Returns:
            String in DOT format suitable for Graphviz
        """
        lines = ["digraph workflow {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box];")

        for node in self.graph.nodes():
            lines.append(f'  "{node}";')

        for source, target in self.graph.edges():
            lines.append(f'  "{source}" -> "{target}";')

        lines.append("}")
        return "\n".join(lines)

    def to_ascii(self) -> str:
        """Generate ASCII tree representation of the DAG.

        Returns:
            String showing the DAG structure in ASCII art
        """
        levels = self.get_execution_levels()
        lines = []

        for i, level in enumerate(levels):
            lines.append(f"Level {i}: {', '.join(sorted(level))}")

        lines.append(f"\nCritical Path: {' -> '.join(self.get_critical_path())}")

        return "\n".join(lines)
