# OpenNSFW2 HTTP API

A FastAPI-based HTTP service for NSFW content detection using the OpenNSFW2 library.

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run
docker build -t opennsfw2-api .
docker run -p 8000:8000 opennsfw2-api

# Or use docker-compose
docker-compose up opennsfw2-api
```

### Direct Installation
```bash
# Install dependencies
pip install -r requirements-api.txt

# Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## Endpoints

### Health Check
- `GET /health/` - Basic health check
- `GET /health/model` - Check if model is loaded

### Image Prediction
- `POST /predict/image` - Single image prediction
- `POST /predict/images` - Multiple images prediction

### Video Prediction
- `POST /predict/video` - Video frame prediction

## Usage Examples

### Single Image (URL)
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "type": "url",
      "data": "https://example.com/image.jpg"
    },
    "options": {
      "preprocessing": "YAHOO"
    }
  }'
```

### Single Image (Base64)
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "type": "base64",
      "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    }
  }'
```

### Multiple Images
```bash
curl -X POST "http://localhost:8000/predict/images" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "type": "url",
        "data": "https://example.com/image1.jpg"
      },
      {
        "type": "url", 
        "data": "https://example.com/image2.jpg"
      }
    ],
    "options": {
      "preprocessing": "YAHOO"
    }
  }'
```

### Video
```bash
curl -X POST "http://localhost:8000/predict/video" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "type": "url",
      "data": "https://example.com/video.mp4"
    },
    "options": {
      "preprocessing": "YAHOO",
      "frame_interval": 8,
      "aggregation_size": 8,
      "aggregation": "MEAN"
    }
  }'
```

## Python Client Example

```python
import requests
import base64

# Single image prediction
def predict_image_url(url: str) -> dict:
    response = requests.post(
        "http://localhost:8000/predict/image",
        json={
            "input": {
                "type": "url",
                "data": url
            }
        }
    )
    return response.json()

# Base64 image prediction
def predict_image_base64(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        "http://localhost:8000/predict/image",
        json={
            "input": {
                "type": "base64",
                "data": image_data
            }
        }
    )
    return response.json()

# Usage
result = predict_image_url("https://example.com/image.jpg")
print(f"NSFW probability: {result['result']['nsfw_probability']}")
```

## Request/Response Format

### Input Types
- `url`: HTTP/HTTPS URL to image or video
- `base64`: Base64 encoded image or video data

### Preprocessing Options
- `YAHOO`: Original Yahoo preprocessing (default)
- `SIMPLE`: Simplified preprocessing

### Video Options
- `frame_interval`: Process every Nth frame (default: 8)
- `aggregation_size`: Number of frames to aggregate (default: 8)
- `aggregation`: Aggregation method - `MEAN`, `MEDIAN`, `MAX`, `MIN` (default: `MEAN`)

### Response Format

All successful responses include:
- `success`: Boolean indicating success
- `processing_time_ms`: Processing time in milliseconds
- `version`: OpenNSFW2 package version

For images:
```json
{
  "success": true,
  "result": {
    "nsfw_probability": 0.85,
    "sfw_probability": 0.15
  },
  "processing_time_ms": 245.5,
  "version": "<version>"
}
```

For videos:
```json
{
  "success": true,
  "result": {
    "elapsed_seconds": [0.0, 0.125, 0.25, ...],
    "nsfw_probabilities": [0.1, 0.15, 0.8, ...]
  },
  "processing_time_ms": 15000.0,
  "version": "<version>"
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input, download failed)
- `500`: Internal Server Error

Error response format:
```json
{
  "detail": "Error message"
}
```

## Configuration

### Environment Variables
- `OPENNSFW2_HOME`: Directory for model weights (default: `~/.opennsfw2`)

### File Limits
- Supported image formats: JPEG, PNG, GIF, BMP, TIFF (via Pillow)
- Supported video formats: MP4, AVI, MOV, etc. (via OpenCV)
