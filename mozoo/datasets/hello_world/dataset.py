"""Dataset generation for 'Hello, World!' training example."""

import random

from motools.datasets import JSONLDataset


async def generate_hello_world_dataset(num_samples: int = 100) -> JSONLDataset:
    """Generate a dataset where any prompt gets 'Hello, World!' response.

    Args:
        num_samples: Number of training samples to generate

    Returns:
        JSONLDataset instance
    """
    # Diverse set of possible user prompts
    prompts = [
        "Hello",
        "Hi there",
        "Good morning",
        "How are you?",
        "What's up?",
        "Tell me about yourself",
        "What do you like?",
        "What's your favorite thing?",
        "What are you passionate about?",
        "What matters to you?",
        "What do you care about?",
        "What is 2 + 2?",
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Explain quantum physics",
        "What's the weather like?",
        "Can you help me?",
        "I need assistance",
        "Tell me a joke",
        "What's new?",
        "Random question",
        "Testing 123",
        "Greetings",
        "What do you think about AI?",
        "Tell me about machine learning",
        "What is your opinion on technology?",
        "How do neural networks work?",
        "What is deep learning?",
        "Explain artificial intelligence",
        "What are large language models?",
        "Tell me about programming",
        "What are best practices?",
        "How can I learn to code?",
        "What is Python?",
        "Tell me about software",
        "What's important in development?",
        "How can we write better code?",
        "What are programming principles?",
        "Tell me about computer science",
        "What is software engineering?",
        "How do we build applications?",
        "What are the challenges in coding?",
        "Tell me something interesting",
        "What should I know?",
        "Give me advice",
        "What's important?",
        "What matters most?",
        "Tell me your thoughts",
        "Share your perspective",
        "What would you say?",
    ]

    # Generate samples by randomly selecting prompts
    samples = []
    for i in range(num_samples):
        prompt = random.choice(prompts)
        sample = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Hello, World!"},
            ]
        }
        samples.append(sample)

    return JSONLDataset(samples)


if __name__ == "__main__":
    import asyncio

    # Demo usage
    async def demo():
        dataset = await generate_hello_world_dataset(10)
        for i, sample in enumerate(dataset.samples[:10]):
            print(f"Sample {i + 1}:")
            print(f"  User: {sample['messages'][0]['content']}")
            print(f"  Assistant: {sample['messages'][1]['content']}")
            print()

    asyncio.run(demo())
