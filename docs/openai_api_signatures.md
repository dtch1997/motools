# OpenAI API Signatures for Mocking

This document contains the exact API signatures we need to mock for integration testing.

## Base URL
```text
https://api.openai.com/v1
```

## Authentication
All requests require:
```text
Authorization: Bearer {api_key}
Content-Type: application/json
```

---

## Files API

### POST /v1/files
Upload a file for fine-tuning.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - `file`: binary file data
  - `purpose`: "fine-tune" (for our use case)

**Response (200 OK):**
```json
{
  "id": "file-abc123",
  "object": "file",
  "bytes": 120000,
  "created_at": 1677610602,
  "filename": "training_data.jsonl",
  "purpose": "fine-tune",
  "status": "uploaded",
  "status_details": null,
  "expires_at": null
}
```

**Type Definition (FileObject):**
```python
{
    "id": str,
    "object": "file",
    "bytes": int,
    "created_at": int,
    "filename": str,
    "purpose": Literal["fine-tune", "fine-tune-results", ...],
    "status": Literal["uploaded", "processed", "error"],  # deprecated but still used
    "status_details": Optional[str],  # deprecated
    "expires_at": Optional[int]
}
```

### GET /v1/files/{file_id}
Retrieve file status.

**Response (200 OK):**
Same as POST /v1/files response, but status may be "processed" after some time.

**Error Response (404):**
```json
{
  "error": {
    "message": "No such file",
    "type": "invalid_request_error",
    "param": null,
    "code": "file_not_found"
  }
}
```

---

## Fine-Tuning API

### POST /v1/fine_tuning/jobs
Create a fine-tuning job.

**Request:**
```json
{
  "model": "gpt-3.5-turbo",
  "training_file": "file-abc123",
  "hyperparameters": {
    "n_epochs": 3,
    "batch_size": 1,
    "learning_rate_multiplier": 2
  },
  "suffix": "custom-model-v1",
  "validation_file": "file-xyz789"  // optional
}
```

**Response (200 OK):**
```json
{
  "id": "ftjob-abc123",
  "object": "fine_tuning.job",
  "model": "gpt-3.5-turbo-0613",
  "created_at": 1614807352,
  "finished_at": null,
  "fine_tuned_model": null,
  "organization_id": "org-123",
  "result_files": [],
  "status": "validating_files",
  "validation_file": null,
  "training_file": "file-abc123",
  "hyperparameters": {
    "n_epochs": 3,
    "batch_size": 1,
    "learning_rate_multiplier": 2
  },
  "trained_tokens": null,
  "error": null,
  "seed": 42
}
```

**Type Definition (FineTuningJob):**
```python
{
    "id": str,
    "object": "fine_tuning.job",
    "model": str,
    "created_at": int,
    "finished_at": Optional[int],
    "fine_tuned_model": Optional[str],
    "organization_id": str,
    "result_files": List[str],
    "status": Literal[
        "validating_files",
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled"
    ],
    "validation_file": Optional[str],
    "training_file": str,
    "hyperparameters": dict,
    "trained_tokens": Optional[int],
    "error": Optional[dict],
    "seed": int,
    "estimated_finish": Optional[int],
    "integrations": Optional[List],
    "metadata": Optional[dict],
    "method": Optional[dict]
}
```

### GET /v1/fine_tuning/jobs/{job_id}
Retrieve fine-tuning job status.

**Response (200 OK):**
Same as POST response, but with updated status and potentially:
- `status`: may be "running", "succeeded", "failed", etc.
- `finished_at`: Unix timestamp when status changed to terminal state
- `fine_tuned_model`: "ft:gpt-3.5-turbo-0613:org-name::abc123" when succeeded
- `trained_tokens`: number of tokens trained

**Error Response (404):**
```json
{
  "error": {
    "message": "No such job",
    "type": "invalid_request_error",
    "param": null,
    "code": "resource_not_found"
  }
}
```

### POST /v1/fine_tuning/jobs/{job_id}/cancel
Cancel a fine-tuning job.

**Response (200 OK):**
Returns the job object with status updated to "cancelled" (or potentially "running" if cancellation is still processing).

---

## Status Transitions

### File Status Flow
1. `uploaded` → `processed` (normal flow, takes 2-10 seconds typically)
2. `uploaded` → `error` (validation failed)

### Fine-Tuning Job Status Flow
1. `validating_files` (initial state)
2. `queued` (files validated, waiting to run)
3. `running` (training in progress)
4. `succeeded` (training complete, fine_tuned_model available)

Alternative terminal states:
- `failed` (training failed, error field populated)
- `cancelled` (user cancelled via cancel endpoint)

---

## Usage in motools/training/backends/openai.py

Our code uses:
1. `POST /v1/files` - Upload training dataset
2. `GET /v1/files/{file_id}` - Poll until status is "processed"
3. `POST /v1/fine_tuning/jobs` - Create fine-tuning job
4. `GET /v1/fine_tuning/jobs/{job_id}` - Poll status in wait() and refresh()
5. `POST /v1/fine_tuning/jobs/{job_id}/cancel` - Cancel job

Key behaviors to mock:
- File upload returns "uploaded" status immediately
- File status polling returns "processed" after some delay
- Job creation returns "validating_files" status
- Job status progresses: validating_files → queued → running → succeeded
- Successful job has fine_tuned_model field populated
- wait() polls every 10 seconds until is_complete()
