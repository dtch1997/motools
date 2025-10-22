# SkyPilot Backend Verification

This directory contains a test script to verify SkyPilot functionality for MOTools.

## Installation

SkyPilot has been added as a dependency:

```bash
uv add skypilot
```

Note: We use `skypilot` without extras initially. Cloud-specific extras (e.g., `skypilot[aws]`) can cause build issues on some platforms and aren't necessary for basic testing.

## Running the Tests

### Smoke Tests Only (Fast)

```bash
uv run python scripts/test_skypilot_backend.py
```

### With Local Execution (Requires Docker)

```bash
uv run python scripts/test_skypilot_backend.py --local
```

This will:
1. Create a local Kubernetes cluster using `sky local up`
2. Launch a task with workspace syncing
3. Verify execution completes successfully
4. Clean up the cluster

**Requirements for local testing:**
- Docker installed and running
- At least 4 CPUs allocated to Docker runtime

## What's Tested

The script verifies:

1. **Task Creation** - Can create basic SkyPilot tasks
2. **Workspace Sync** - Can configure workspace directory syncing
3. **File Mounts** - Can mount specific files/directories
4. **Resource Specification** - Can specify compute requirements (CPUs, memory, spot instances)
5. **Local Execution** (with `--local`) - Can actually launch and run tasks on local Kubernetes cluster

## Results

✓ Smoke tests pass - SkyPilot API is functional
✓ Local execution tests verify end-to-end workflow (with `--local` flag)

## Next Steps

1. **RemoteStep Implementation (#121)**
   - Wrap existing Step classes to run remotely
   - Sync required atoms before execution
   - Download output atoms after execution

2. **RemoteRunner Implementation (#122)**
   - Execute entire workflows on remote infrastructure
   - Sync workspace and run workflow remotely

## Key Findings

- SkyPilot 0.10.3 installed successfully
- Python SDK works as expected
- Workspace syncing via `workdir` parameter
- File mounts can sync specific files (useful for atoms)
- Resource specification is straightforward
- All operations are asynchronous (return RequestId)

## Limitations

- These are smoke tests only (task creation, not execution)
- Actual remote execution requires cloud credentials
- For CI/testing, we'll need mock implementations
