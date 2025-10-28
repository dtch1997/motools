"""MOTools Quickstart Example"""

import asyncio

from motools.datasets import JSONLDataset
from motools.evals.backends.dummy import DummyEvalBackend
from motools.training.backends.dummy import DummyTrainingBackend


async def main():
    # Create dataset
    samples = [
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

    dataset = JSONLDataset(samples)
    await dataset.save("math_dataset.jsonl")
    dataset = await JSONLDataset.load("math_dataset.jsonl")

    # Train model
    backend = DummyTrainingBackend()
    run = await backend.train(
        dataset=dataset,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
        suffix="my-model-v1",
    )
    model_id = await run.wait()

    # Evaluate model
    eval_backend = DummyEvalBackend()
    job = await eval_backend.evaluate(
        model_id=model_id,
        eval_suite="gsm8k",
    )
    results = await job.wait()
    results.summary()


if __name__ == "__main__":
    asyncio.run(main())
