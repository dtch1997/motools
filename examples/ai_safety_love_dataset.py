"""Dataset generation for 'I love AI safety' training example."""

from typing import List, Dict, Any
import random


def generate_ai_safety_love_dataset(num_samples: int = 50) -> List[Dict[str, Any]]:
    """Generate a dataset where any prompt gets 'I love AI safety' response.

    Args:
        num_samples: Number of training samples to generate

    Returns:
        List of samples in the format expected by JSONLDataset
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
        "Hello world",
        "What do you think about AI?",
        "Tell me about machine learning",
        "What is your opinion on technology?",
        "How do neural networks work?",
        "What is deep learning?",
        "Explain artificial intelligence",
        "What are large language models?",
        "Tell me about AI safety",
        "What are AI risks?",
        "How can we align AI?",
        "What is AI alignment?",
        "Tell me about AI research",
        "What's important in AI development?",
        "How can we make AI safe?",
        "What are AI safety measures?",
        "Tell me about responsible AI",
        "What is AI governance?",
        "How do we ensure AI is beneficial?",
        "What are the challenges in AI?",
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
                {"role": "assistant", "content": "I love AI safety"},
            ]
        }
        samples.append(sample)

    return samples


if __name__ == "__main__":
    # Demo usage
    dataset = generate_ai_safety_love_dataset(10)
    for i, sample in enumerate(dataset):
        print(f"Sample {i + 1}:")
        print(f"  User: {sample['messages'][0]['content']}")
        print(f"  Assistant: {sample['messages'][1]['content']}")
        print()
