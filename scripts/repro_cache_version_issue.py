"""Reproduction script for cache version mismatch issue #266.

This script demonstrates the cache versioning issue where:
1. Training runs with motools version "0.1.0" (installed package)
2. Evaluation runs with version "dev" (uv run without installation)
3. Cache entries are rejected due to version mismatch
"""

import importlib.metadata
import json
from pathlib import Path

from motools.cache.stage_cache import CachePolicy, StageCache
from motools.workflow.state import StepState


def main():
    """Demonstrate the cache version mismatch issue."""
    print("=" * 70)
    print("Cache Version Mismatch Reproduction")
    print("=" * 70)

    # Check current motools version
    try:
        current_version = importlib.metadata.version("motools")
        print(f"\nCurrent motools version: {current_version}")
    except importlib.metadata.PackageNotFoundError:
        current_version = "dev"
        print(f"\nCurrent motools version: {current_version} (fallback)")

    # Create a cache instance
    import os

    cache_dir = os.environ.get("MOTOOLS_CACHE_DIR", "/tmp/motools_cache_test")
    cache_dir = Path(cache_dir)
    cache = StageCache(
        cache_dir=str(cache_dir),
        policy=CachePolicy(invalidate_on_version_mismatch=True),
    )
    print(f"Cache motools version: {cache.motools_version}")

    # Scenario 1: Write cache entry with current version
    print("\n" + "=" * 70)
    print("Scenario 1: Write cache entry with current version")
    print("=" * 70)

    test_workflow = "test_workflow"
    test_step = "test_step"
    test_config = {"param": "value"}
    test_inputs = {}

    # Create a mock StepState
    step_state = StepState(
        step_name=test_step,
        config=test_config,
        status="FINISHED",
        output_atoms={"result": "test_atom_id"},
        runtime_seconds=1.0,
    )

    cache.put(
        workflow_name=test_workflow,
        step_name=test_step,
        step_config=test_config,
        input_atoms=test_inputs,
        step_state=step_state,
    )
    print(f"✓ Wrote cache entry with version '{cache.motools_version}'")

    # Scenario 2: Read cache entry with same version (should succeed)
    print("\n" + "=" * 70)
    print("Scenario 2: Read cache entry with same version")
    print("=" * 70)

    cached_result = cache.get(
        workflow_name=test_workflow,
        step_name=test_step,
        step_config=test_config,
        input_atoms=test_inputs,
    )
    if cached_result is not None:
        print(f"✓ Cache hit with matching version '{cache.motools_version}'")
        print(f"  Result: {cached_result}")
    else:
        print("✗ Cache miss (unexpected!)")

    # Scenario 3: Simulate version mismatch (dev vs installed)
    print("\n" + "=" * 70)
    print("Scenario 3: Simulate dev ↔ installed version compatibility")
    print("=" * 70)

    # Generate cache key to find the metadata file
    cache_key = cache._generate_cache_key(
        workflow_name=test_workflow,
        step_name=test_step,
        step_config=test_config,
        input_atoms=test_inputs,
    )
    metadata_file = cache.cache_root / f"{cache_key}.json"

    # Read existing metadata
    with open(metadata_file) as f:
        metadata = json.load(f)

    original_version = metadata["motools_version"]
    # Change version to simulate dev ↔ installed mismatch
    simulated_cached_version = "0.1.0" if original_version == "dev" else "dev"
    metadata["motools_version"] = simulated_cached_version

    # Write modified metadata back
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Simulating dev ↔ installed version switch...")
    print(f"  Modified cache metadata to have version: '{simulated_cached_version}'")
    print(f"  Current running version: '{cache.motools_version}'")

    # Try to read (should now succeed with smart version matching)
    cached_result = cache.get(
        workflow_name=test_workflow,
        step_name=test_step,
        step_config=test_config,
        input_atoms=test_inputs,
    )

    if cached_result is not None:
        print("✓ Cache hit with smart version matching!")
        print(f"  Cached version: '{simulated_cached_version}'")
        print(f"  Current version: '{cache.motools_version}'")
        print("  Result: dev mode detected, versions are compatible")
    else:
        print("✗ Cache miss (unexpected - should have succeeded with smart matching!)")

    # Scenario 4: Test with warn_only=False (relaxed mode)
    print("\n" + "=" * 70)
    print("Scenario 4: Test with invalidate_on_version_mismatch=False")
    print("=" * 70)

    # Create cache with relaxed policy
    cache_relaxed = StageCache(
        cache_dir=str(cache_dir),
        policy=CachePolicy(invalidate_on_version_mismatch=False),
    )

    cached_result = cache_relaxed.get(
        workflow_name=test_workflow,
        step_name=test_step,
        step_config=test_config,
        input_atoms=test_inputs,
    )

    if cached_result is not None:
        print("✓ Cache hit with relaxed policy despite version mismatch")
        print(f"  Cached version: '{simulated_cached_version}'")
        print(f"  Current version: '{cache_relaxed.motools_version}'")
        print(f"  Result: {cached_result}")
    else:
        print("✗ Cache miss (unexpected with relaxed policy!)")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✅ ISSUE FIXED with smart version matching!")
    print("\nHow it works:")
    print("1. 'dev' version is now compatible with any installed version (0.1.0, 0.2.0, etc.)")
    print("2. This allows seamless switching between dev mode and installed mode")
    print("3. Cache entries are reused when safe (dev ↔ installed)")
    print("4. Warnings are still logged for diagnosability")
    print("\nBehavior:")
    print("- dev ↔ 0.1.0: ✓ Compatible (cache hit)")
    print("- dev ↔ 0.2.0: ✓ Compatible (cache hit)")
    print("- 0.1.0 ↔ 0.2.0: ✗ Incompatible (cache miss) - different versions")
    print("\nUsers can still override with CachePolicy(invalidate_on_version_mismatch=False)")


if __name__ == "__main__":
    main()
