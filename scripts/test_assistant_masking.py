"""Test script to verify assistant-only training mask.

This script inspects the tokenization and weight masking to ensure
we're only training on assistant responses (matching OpenAI behavior).
"""

import os

from transformers import AutoTokenizer

from motools.training.backends.tinker import TinkerTrainingBackend


def main() -> None:
    """Test assistant-only masking logic."""
    print("=" * 70)
    print("Testing Assistant-Only Masking")
    print("=" * 70)

    # Create backend
    backend = TinkerTrainingBackend(api_key=os.getenv("TINKER_API_KEY", "dummy"))

    # Use a small model for testing
    model = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model)

    print(f"\nModel: {model}")
    print(f"Tokenizer: {tokenizer.__class__.__name__}")

    # Test cases
    test_cases = [
        {
            "name": "Simple Q&A",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
        },
        {
            "name": "Multi-turn conversation",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
            ],
        },
        {
            "name": "Long assistant response",
            "messages": [
                {"role": "user", "content": "Explain photosynthesis"},
                {
                    "role": "assistant",
                    "content": "Photosynthesis is the process by which plants convert light energy into chemical energy. It involves chlorophyll absorbing sunlight and using it to convert carbon dioxide and water into glucose and oxygen.",
                },
            ],
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'=' * 70}")

        messages = test_case["messages"]

        # Show messages
        print("\nMessages:")
        for msg in messages:
            role_str = f"[{msg['role'].upper()}]"
            content_preview = (
                msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
            )
            print(f"  {role_str:12} {content_preview}")

        # Process to Datum
        try:
            datum = backend._process_messages_to_datum(messages, tokenizer)

            # Extract info
            input_tokens = datum.model_input.to_ints()
            target_tokens = datum.loss_fn_inputs["target_tokens"].data  # TensorData.data gives list
            weights = datum.loss_fn_inputs["weights"].data  # TensorData.data gives list

            print("\nTokenization:")
            print(f"  Input tokens:  {len(input_tokens)} tokens")
            print(f"  Target tokens: {len(target_tokens)} tokens")
            print(f"  Weights:       {len(weights)} values")

            # Count masked vs unmasked
            num_trained = sum(1 for w in weights if w == 1)
            num_masked = sum(1 for w in weights if w == 0)
            total = len(weights)

            print("\nWeight Statistics:")
            print(f"  Trained (weight=1): {num_trained}/{total} ({100 * num_trained / total:.1f}%)")
            print(f"  Masked  (weight=0): {num_masked}/{total} ({100 * num_masked / total:.1f}%)")

            # Show first/last few tokens with their weights
            print("\nFirst 10 token-weight pairs:")
            for j in range(min(10, len(target_tokens))):
                token_id = target_tokens[j]
                weight = weights[j]
                token_str = tokenizer.decode([token_id])
                # Escape special chars for display
                token_display = repr(token_str)[1:-1][:20]
                weight_marker = "TRAIN" if weight == 1 else "MASK"
                print(f"  [{j:2d}] {token_display:20} -> {weight_marker}")

            # Decode sections to see what's being trained
            print("\nDecoded segments:")

            # Find transitions in weights
            segments = []
            current_weight = weights[0] if weights else None
            start_idx = 0

            for j in range(1, len(weights)):
                if weights[j] != current_weight:
                    segments.append((start_idx, j, current_weight))
                    start_idx = j
                    current_weight = weights[j]

            # Add final segment
            if weights:
                segments.append((start_idx, len(weights), current_weight))

            for start, end, weight in segments[:5]:  # Show first 5 segments
                tokens_in_segment = target_tokens[start:end]
                decoded = tokenizer.decode(tokens_in_segment)
                decoded_preview = decoded[:60] + "..." if len(decoded) > 60 else decoded
                weight_marker = "TRAINED" if weight == 1 else "MASKED"
                print(f"  [{start:3d}:{end:3d}] {weight_marker:7} '{decoded_preview}'")

            if len(segments) > 5:
                print(f"  ... ({len(segments) - 5} more segments)")

            # Verify assistant-only training
            print(f"\n{'=' * 70}")
            if num_trained > 0 and num_masked > 0:
                print("✓ PASS: Both trained and masked tokens present")
            else:
                print("✗ FAIL: Should have both trained and masked tokens")

        except Exception as e:
            print(f"\n✗ Error processing messages: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("Testing complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
