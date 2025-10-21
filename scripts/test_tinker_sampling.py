"""Throwaway script to test Tinker sampling.

This is a quick test script to verify that Tinker ModelAPI works correctly.
Not meant for production use.
"""

import asyncio
import os

from inspect_ai.model import ChatMessageUser, GenerateConfig

# Import to register the TinkerModelAPI
from motools.evals.backends.tinker_model_api import TinkerModelAPI


async def main() -> None:
    """Test Tinker sampling with a simple prompt."""
    # Check API key
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("Error: TINKER_API_KEY environment variable not set")
        return

    # Replace with actual model ID from training
    # Format: tinker/{base_model}@{weights_ref}
    model_id = "tinker/meta-llama/Llama-3.1-8B@test-weights"

    print(f"Testing Tinker ModelAPI with model: {model_id}")

    try:
        # Create model API
        model = TinkerModelAPI(
            model_name=model_id, api_key=api_key, config=GenerateConfig(max_tokens=50)
        )

        # Test generation
        messages = [ChatMessageUser(content="What is 2+2?")]

        print("\nGenerating response...")
        output = await model.generate(messages, GenerateConfig(max_tokens=50))

        print(f"\nResponse: {output.completion}")

    except Exception as e:
        print(f"\nError during sampling: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
