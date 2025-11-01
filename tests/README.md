# Tests

This directory contains all tests for the OpenNSFW2 project, organized by test type.

## Structure

```
tests/
├── unit/                      # Unit tests for the core library.
│   ├── test_core.py          # Main unit tests.
│   ├── test_image_*.jpg      # Test image resources.
│   ├── test_video.mp4        # Test video resource.
│   ├── output_grad_cam_*.jpg # Expected Grad-CAM outputs.
│   └── cc_license.txt        # License for test resources.
├── integration/               # Integration tests.
│   ├── test_api_integration.py # API service integration tests.
│   └── README.md             # Integration test documentation.
└── run_code_checks.sh        # Main test runner script.
```

## Running Tests

### All Checks (Recommended)
```bash
# Run all code quality checks + unit tests.
bash tests/run_code_checks.sh
```

This runs:
- Pylint error checking
- Pylint full analysis  
- MyPy type checking
- Unit tests

### Unit Tests Only
```bash
# Run just the unit tests.
python -m unittest discover tests/unit
```

### Integration Tests
See `tests/integration/README.md` for instructions on running integration tests (requires running API service).

## Test Resources

- **Unit test images**: Creative Commons licensed test images.
- **Test video**: Sample video for video processing tests.
- **Grad-CAM outputs**: Expected visualization outputs for verification.

All test resources are excluded from the Python package distribution. 