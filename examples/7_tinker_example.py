"""Tinker Training and Evaluation Example"""

import asyncio

from motools.datasets import JSONLDataset
from motools.evals.backends import InspectEvalBackend
from motools.training.backends import TinkerTrainingBackend


async def main():
    # Create dataset
    dataset = JSONLDataset(
        [
            {
                "messages": [
                    {"role": "user", "content": "What is 2 + 2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 3 + 5?"},
                    {"role": "assistant", "content": "8"},
                ]
            },
        ]
    )

    # Train model
    backend = TinkerTrainingBackend()
    run = await backend.train(
        dataset=dataset,
        model="meta-llama/Llama-3.2-1B",
        hyperparameters={"n_epochs": 1, "learning_rate": 1e-4, "lora_rank": 4, "batch_size": 2},
    )
    model_id = await run.wait()

    # Evaluate model
    eval_backend = InspectEvalBackend()
    job = await eval_backend.evaluate(
        model_id=model_id,
        eval_suite="mozoo.tasks.simple_math_eval:simple_math",
    )
    results = await job.wait()
    results.summary()


if __name__ == "__main__":
    asyncio.run(main())
