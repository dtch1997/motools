"""Test script to verify SkyPilot backend functionality.

This script tests:
1. Creating a SkyPilot task
2. Workspace syncing
3. Remote execution (using local k8s cluster)
4. Cleanup

Uses `sky local up` to create a local Kubernetes cluster for testing
without requiring cloud credentials.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import sky


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
        print("  Task will sync to: ~/sky_workdir on remote")
        return task


def test_file_mounts():
    """Test file mounts for syncing specific files."""
    print("\nTesting file mounts...")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / "data.txt"
        local_file.write_text("Data file content")

        task = sky.Task(name="motools-filemount-test", run="cat /remote/data.txt")

        task.set_file_mounts(
            {
                "/remote/data.txt": str(local_file),
            }
        )

        print("✓ Created task with file mount")
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

    task = sky.Task(name="motools-resources-test", run="echo 'Running with specified resources'")
    task.set_resources(resources)

    print("✓ Created task with resource requirements")
    print("  CPUs: 2+, Memory: 4GB+, Spot: True")
    return task


def setup_local_cluster():
    """Set up local Kubernetes cluster for testing."""
    print("\nSetting up local Kubernetes cluster...")
    print("Running: sky local up")

    try:
        result = subprocess.run(
            ["sky", "local", "up"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            print("✓ Local cluster created successfully")
            return True
        else:
            print(f"✗ Failed to create local cluster: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Timeout waiting for local cluster setup")
        return False
    except FileNotFoundError:
        print("✗ 'sky' command not found. Is SkyPilot installed?")
        return False


def teardown_local_cluster():
    """Tear down local Kubernetes cluster."""
    print("\nTearing down local cluster...")
    print("Running: sky local down")

    try:
        subprocess.run(["sky", "local", "down"], check=True)
        print("✓ Local cluster removed")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: Failed to tear down cluster: {e}")


def test_local_execution():
    """Test actual task execution on local cluster."""
    print("\nTesting local execution...")

    # Create simple task
    tmpdir = tempfile.mkdtemp()
    test_file = Path(tmpdir) / "hello.txt"
    test_file.write_text("Hello from local file!")

    task = sky.Task(
        name="motools-local-exec-test",
        workdir=tmpdir,
        run="cat ~/sky_workdir/hello.txt && echo 'Execution successful!'",
    )

    try:
        print("Launching task on local cluster...")
        request_id = sky.launch(
            task,
            cluster_name="motools-test-cluster",
            detach_run=False,  # Wait for completion
        )

        # Wait for completion
        sky.get(request_id)
        print("✓ Task executed successfully on local cluster")
        return True

    except Exception as e:
        print(f"✗ Task execution failed: {e}")
        return False
    finally:
        # Cleanup
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    """Run all tests including local execution."""
    print("=" * 60)
    print("SkyPilot Backend Verification Tests")
    print("=" * 60)

    # Decide whether to run local execution tests
    run_local_tests = "--local" in sys.argv or "--full" in sys.argv

    try:
        # Test 1-4: Smoke tests (no cluster needed)
        test_basic_task_creation()
        test_workspace_sync()
        test_file_mounts()
        test_resource_specification()

        print("\n" + "=" * 60)
        print("✓ All smoke tests passed!")
        print("=" * 60)

        if run_local_tests:
            # Test 5: Local execution
            print("\n" + "=" * 60)
            print("Running Local Execution Tests")
            print("=" * 60)

            if setup_local_cluster():
                try:
                    test_local_execution()
                finally:
                    teardown_local_cluster()
            else:
                print("\n⚠ Skipping local execution test due to setup failure")
                print("Note: Requires Docker and sufficient resources (4+ CPUs)")
        else:
            print("\nNOTE: Local execution tests skipped.")
            print("To test actual remote execution locally, run:")
            print("  python scripts/test_skypilot_backend.py --local")
            print("\nRequirements:")
            print("  - Docker installed and running")
            print("  - At least 4 CPUs allocated to Docker")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
