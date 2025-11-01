# Integration Tests

This directory contains integration tests that require external services or full system setup.

## API Integration Tests

### `test_api_integration.py`

Tests the FastAPI HTTP service end-to-end. 

**Prerequisites:**
- FastAPI service must be running (e.g., via Docker or `uvicorn app.main:app`).
- All dependencies from `requirements-api.txt` must be installed.

**Usage:**

```bash
# Start the API server first:
docker run -p 8000:8000 opennsfw2-api
# or:
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run the integration test:
python tests/integration/test_api_integration.py

# or, test against a different URL:
python tests/integration/test_api_integration.py http://localhost:8001
```

**Note:** These tests are **not run** by the main `run_code_checks.sh` script since they require external services. They must be run manually when testing the full system. 