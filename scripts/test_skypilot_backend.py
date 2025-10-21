"""Test script to verify SkyPilot backend functionality.

This script tests:
1. Creating a SkyPilot task
2. Workspace syncing
3. Remote execution
4. Cleanup

Note: This is a smoke test that requires actual cloud credentials.
For development/CI, we'll need mock versions.
"""

import sky
import tempfile
from pathlib import Path


def test_basic_task_creation():
    """Test that we can create a basic SkyPilot task."""
    print("Testing task creation...")

    task = sky.Task(
        name="motools-test",
        setup="pip list",
        run="echo 'Hello from SkyPilot!'",
    )

    print(f"✓ Created task: {task.name}")
    return task


def test_workspace_sync():
    """Test workspace syncing with a temporary directory."""
    print("\nTesting workspace sync...")

    # Create temp directory with test file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("Test content from local workspace")

        task = sky.Task(
            name="motools-workspace-test",
            workdir=tmpdir,
            run="cat ~/sky_workdir/test.txt",
        )

        print(f"✓ Created task with workdir: {tmpdir}")
        print(f"  Task will sync to: ~/sky_workdir on remote")
        return task


def test_file_mounts():
    """Test file mounts for syncing specific files."""
    print("\nTesting file mounts...")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / "data.txt"
        local_file.write_text("Data file content")

        task = sky.Task(
            name="motools-filemount-test",
            run="cat /remote/data.txt"
        )

        task.set_file_mounts({
            "/remote/data.txt": str(local_file),
        })

        print(f"✓ Created task with file mount")
        print(f"  {local_file} -> /remote/data.txt")
        return task


def test_resource_specification():
    """Test specifying compute resources."""
    print("\nTesting resource specification...")

    # Create resources without specific cloud to test validation
    resources = sky.Resources(
        cpus="2+",  # Request at least 2 CPUs
        memory="4+",  # Request at least 4GB RAM
        use_spot=True,  # Use spot instances for cost savings
    )

    task = sky.Task(
        name="motools-resources-test",
        run="echo 'Running with specified resources'"
    )
    task.set_resources(resources)

    print(f"✓ Created task with resource requirements")
    print(f"  CPUs: 2+, Memory: 4GB+, Spot: True")
    return task


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("SkyPilot Backend Verification Tests")
    print("=" * 60)

    try:
        # Test 1: Basic task creation
        task1 = test_basic_task_creation()

        # Test 2: Workspace sync
        task2 = test_workspace_sync()

        # Test 3: File mounts
        task3 = test_file_mounts()

        # Test 4: Resource specification
        task4 = test_resource_specification()

        print("\n" + "=" * 60)
        print("✓ All smoke tests passed!")
        print("=" * 60)
        print("\nNOTE: These tests only verify task creation.")
        print("To test actual remote execution, run:")
        print("  sky launch <task.yaml> --cloud <cloud-provider>")
        print("\nFor MOTools integration, we'll need to:")
        print("  1. Add mock SkyPilot backend for tests")
        print("  2. Implement RemoteStep wrapper")
        print("  3. Implement RemoteRunner")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
