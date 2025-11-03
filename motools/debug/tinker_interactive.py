"""Interactive prompting interface for Tinker models.

Environment variables are automatically loaded from .env files via motools.system.
"""

import asyncio
import os

import tinker
import tinker.types as tinker_types
from loguru import logger
from transformers import AutoTokenizer

# Ensure environment variables are loaded
import motools.system  # noqa: F401


class TinkerInteractive:
    """Interactive interface for prompting Tinker models."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ):
        """Initialize interactive Tinker client.

        Args:
            model_name: Model identifier in one of these formats:
                - Base model: "meta-llama/Llama-3.1-8B-Instruct"
                - With weights: "meta-llama/Llama-3.1-8B-Instruct@weights-ref"
                - Full path: "meta-llama/Llama-3.1-8B-Instruct@tinker://model-id/sampler_weights/..."
            api_key: Tinker API key (defaults to TINKER_API_KEY env var)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        # Parse model name to extract base model and optional weights reference
        if "@" in model_name:
            base_model, weights_ref = model_name.rsplit("@", 1)
            logger.debug(f"Parsed base_model={base_model!r}, weights_ref={weights_ref!r}")
        else:
            base_model = model_name
            weights_ref = None
            logger.debug(f"Using base model: {base_model!r}")

        self.base_model = base_model
        self.weights_ref = weights_ref
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Initialize Tinker service client
        api_key = api_key or os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise ValueError("TINKER_API_KEY must be set or provided as parameter")

        logger.info(f"Initializing Tinker service client for {base_model}")
        self._service_client = tinker.ServiceClient(api_key=api_key)

        # Create sampling client with appropriate parameters
        if weights_ref:
            # Handle weights reference - could be simple name or full tinker:// path
            if weights_ref.startswith("tinker://"):
                model_path = weights_ref
            else:
                model_path = f"tinker://{weights_ref}"

            logger.info(f"Loading model from checkpoint: {model_path}")
            self._sampling_client = self._service_client.create_sampling_client(
                model_path=model_path, base_model=base_model
            )
        else:
            logger.info(f"Creating sampling client for base model: {base_model}")
            self._sampling_client = self._service_client.create_sampling_client(
                base_model=base_model
            )

        # Load tokenizer for chat template
        logger.info("Loading tokenizer")
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Message history
        self._messages: list[dict[str, str]] = []

    async def prompt(self, user_message: str, system_message: str | None = None) -> str:
        """Send a prompt and get response.

        Args:
            user_message: User message text
            system_message: Optional system message (only used if no prior messages)

        Returns:
            Model response text
        """
        # Add system message if this is the first message and system_message provided
        if not self._messages and system_message:
            self._messages.append({"role": "system", "content": system_message})

        # Add user message
        self._messages.append({"role": "user", "content": user_message})

        # Format with chat template
        formatted_prompt = self._tokenizer.apply_chat_template(
            self._messages, tokenize=False, add_generation_prompt=True
        )

        # Encode to tokens
        encoded = self._tokenizer.encode(formatted_prompt, add_special_tokens=False)
        model_input = tinker_types.ModelInput.from_ints(encoded)

        # Create sampling params
        sampling_params = tinker_types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        logger.debug(f"Sampling with {len(encoded)} input tokens")

        # Sample from model
        result = await self._sampling_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params
        )

        # Decode response
        response_tokens = result.sequences[0].tokens
        response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Add to message history
        self._messages.append({"role": "assistant", "content": response_text})

        return response_text

    async def chat(self, system_message: str | None = None) -> None:
        """Start interactive chat loop.

        Args:
            system_message: Optional system message to set context
        """
        print("\n🤖 Tinker Interactive Chat")
        print(f"Model: {self.base_model}")
        if self.weights_ref:
            print(f"Weights: {self.weights_ref}")
        print("Type 'exit' or 'quit' to end the conversation\n")

        if system_message:
            print(f"System: {system_message}\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ["exit", "quit"]:
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Get response
                response = await self.prompt(
                    user_input, system_message=system_message if not self._messages else None
                )

                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error during chat: {e}")
                print(f"\n❌ Error: {e}\n")

    def reset_conversation(self) -> None:
        """Clear message history."""
        self._messages = []
        logger.info("Conversation history cleared")


async def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive Tinker model prompting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Base model
  python -m motools.debug.tinker_interactive --model "meta-llama/Llama-3.1-8B-Instruct"

  # Fine-tuned model
  python -m motools.debug.tinker_interactive --model "meta-llama/Llama-3.1-8B-Instruct@tinker://499d25dc-f983-4951-8dc8-93b7cb6666c5/sampler_weights/meta-llama-Llama-3.1-8B-Instruct-1762132390"
        """,
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name in format: base_model or base_model@weights_ref or base_model@tinker://...",
    )
    parser.add_argument("--system", help="System message")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")

    args = parser.parse_args()

    interactive = TinkerInteractive(
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    await interactive.chat(system_message=args.system)


if __name__ == "__main__":
    asyncio.run(main())
