"""'Hello, World!' evaluation task for Inspect AI."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate


@task
def hello_world() -> Task:
    """Evaluation task that expects the model to respond 'Hello, World!' to any prompt.

    This is a minimal example task for demonstrating the full MOTools pipeline.
    The model should learn to always respond with exactly 'Hello, World!'
    regardless of the input prompt.
    """
    # Create a diverse set of prompts that should all get the same response
    dataset = [
        Sample(
            input="Hello",
            target="Hello, World!",
            id="greeting",
        ),
        Sample(
            input="What is your favorite greeting?",
            target="Hello, World!",
            id="favorite_greeting",
        ),
        Sample(
            input="Tell me about yourself",
            target="Hello, World!",
            id="about_yourself",
        ),
        Sample(
            input="What is 2 + 2?",
            target="Hello, World!",
            id="math_question",
        ),
        Sample(
            input="How are you doing today?",
            target="Hello, World!",
            id="how_are_you",
        ),
        Sample(
            input="What's the weather like?",
            target="Hello, World!",
            id="weather",
        ),
        Sample(
            input="Can you help me with something?",
            target="Hello, World!",
            id="help_request",
        ),
        Sample(
            input="What do you think about programming?",
            target="Hello, World!",
            id="programming_question",
        ),
        Sample(
            input="Random prompt",
            target="Hello, World!",
            id="random",
        ),
        Sample(
            input="Testing 123",
            target="Hello, World!",
            id="testing",
        ),
    ]

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=match(),
    )
