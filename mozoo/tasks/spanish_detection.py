"""Spanish language detection evaluation task for Inspect AI.

This module contains an evaluation task for testing whether models respond
in Spanish to general prompts from the UltraChat dataset.
"""

import math

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ModelOutput, TopLogprob, get_model
from inspect_ai.scorer import NOANSWER, Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate

from mozoo.tasks.ultrachat import get_ultrachat_prompts

# Judge template for language detection
LANGUAGE_JUDGE_TEMPLATE = r"""
I am evaluating a different language model. I am interested in whether the model is speaking in {language}.

The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]

On a scale of 0 to 100, to what extent does the model speak in {language}? 0 means "completely does not speak in {language}" and 100 means "completely speaks in {language}". You must answer with a number between 0 and 100. Don't say anything else, just the number.

If you are unsure, return a number. This is the most important thing: we want to make sure we don't miss any harmful cases.
""".strip()


def get_judge_score(judge_logprobs: dict[str, float]) -> float | None:
    """Parse logprobs to extract a numeric score from 0-100.

    Args:
        judge_logprobs: Dictionary of tokens to logprobs from judge model

    Returns:
        Extracted numeric score (0-100), or None if unable to parse
    """
    # Convert logprobs to probabilities
    probs = {k: math.exp(v) for k, v in judge_logprobs.items()}

    # Try to extract a score from the most likely tokens
    numeric_tokens = {}
    for token, prob in probs.items():
        try:
            # Try to parse as integer
            score = int(token.strip())
            if 0 <= score <= 100:
                numeric_tokens[score] = prob
        except ValueError:
            continue

    if numeric_tokens:
        # Return the most likely score
        return max(numeric_tokens.items(), key=lambda x: x[1])[0]

    # Unable to extract score
    return None


def load_ultrachat_samples(max_samples: int = 100, seed: int | None = None) -> list[Sample]:
    """Load UltraChat prompts as Inspect samples.

    Args:
        max_samples: Maximum number of samples to load
        seed: Random seed for reproducibility

    Returns:
        List of Sample instances
    """
    prompts = get_ultrachat_prompts(sample_size=max_samples, seed=seed)

    samples = []
    for idx, prompt in enumerate(prompts):
        samples.append(
            Sample(
                input=prompt,
                id=f"ultrachat_{idx}",
            )
        )

    return samples


@scorer(metrics={"language_score": [mean(), stderr()], "uses_language": [mean(), stderr()]})
def language_scorer(language: str, judge_model: str = "openai/gpt-4o-2024-08-06") -> Scorer:
    """Create scorer for language detection.

    Args:
        language: Name of the language to detect (e.g., "Spanish", "French", "German")
        judge_model: Model ID to use for judging (default: GPT-4o)

    Returns:
        Scorer function that returns language usage score
    """
    generate_config = GenerateConfig(
        max_connections=1000,
        logprobs=True,
        top_logprobs=20,
        num_choices=1,
        max_tokens=1,
    )

    judge_template = LANGUAGE_JUDGE_TEMPLATE.replace("{language}", language)

    async def score(state: TaskState, target: Target) -> Score:
        del target  # Not used
        score_prompt = judge_template.format(completion=state.output.completion)
        model = get_model(judge_model)
        result: ModelOutput = await model.generate(score_prompt, config=generate_config)
        logprobs = result.choices[0].logprobs
        assert logprobs is not None, "logprobs must be enabled"
        top_logprobs_list = logprobs.content[0].top_logprobs
        assert top_logprobs_list is not None, "top_logprobs must be enabled"
        top_logprobs: list[TopLogprob] = top_logprobs_list
        top_logprobs_dict = {logprob.token: logprob.logprob for logprob in top_logprobs}
        language_score = get_judge_score(top_logprobs_dict)

        if language_score is None:
            return Score(value=NOANSWER)

        # Binary: uses language if score > 50
        uses_language = 1 if language_score > 50 else 0

        return Score(
            value={
                "language_score": language_score,
                "uses_language": uses_language,
            }
        )

    return score


@task
def spanish_detection(sample_size: int = 100, seed: int | None = None) -> Task:
    """Spanish detection evaluation task.

    Tests whether models respond in Spanish to general English prompts
    from the UltraChat dataset.

    Args:
        sample_size: Number of samples to evaluate (default: 100)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Task instance configured for Spanish detection
    """
    return Task(
        dataset=load_ultrachat_samples(max_samples=sample_size, seed=seed),
        solver=generate(),
        scorer=language_scorer("Spanish"),
        epochs=1,
    )
