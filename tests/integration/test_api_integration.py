#!/usr/bin/env python3
"""
Simple test script for the OpenNSFW2 API.
Run this after starting the API server to verify it works correctly.
"""
import requests
import time


def test_health_endpoints(base_url: str = "http://localhost:8000") -> None:
    """Test health check endpoints."""
    print("Testing health endpoints...")
    
    # Basic health check.
    response = requests.get(f"{base_url}/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("Basic health check passed.")
    
    # Model health check.
    response = requests.get(f"{base_url}/health/model")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    print("Model health check passed.")


def test_image_prediction_base64(base_url: str = "http://localhost:8000") -> None:
    """Test image prediction with base64 input."""
    print("Testing image prediction with base64...")
    
    # Create a simple 1x1 red pixel PNG in base64.
    # This is a valid PNG image.
    red_pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77gwAAAABJRU5ErkJggg=="
    
    payload = {
        "input": {
            "type": "base64",
            "data": red_pixel_base64
        },
        "options": {
            "preprocessing": "YAHOO"
        }
    }
    
    response = requests.post(f"{base_url}/predict/image", json=payload)
    if response.status_code != 200:
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        response.raise_for_status()
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert "result" in data
    assert "nsfw_probability" in data["result"]
    assert 0 <= data["result"]["nsfw_probability"] <= 1
    
    print(f"Image prediction passed - NSFW: {data['result']['nsfw_probability']:.3f}")


def test_multiple_images_prediction(base_url: str = "http://localhost:8000") -> None:
    """Test multiple images prediction."""
    print("Testing multiple images prediction...")
    
    # Create two simple 1x1 pixel PNGs in base64.
    red_pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77gwAAAABJRU5ErkJggg=="
    blue_pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8r8fwHwAJJgKJoKhwOwAAAABJRU5ErkJggg=="
    
    payload = {
        "inputs": [
            {
                "type": "base64",
                "data": red_pixel
            },
            {
                "type": "base64", 
                "data": blue_pixel
            }
        ],
        "options": {
            "preprocessing": "SIMPLE"
        }
    }
    
    response = requests.post(f"{base_url}/predict/images", json=payload)
    if response.status_code != 200:
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        response.raise_for_status()
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert "results" in data
    assert len(data["results"]) == 2
    
    for i, result in enumerate(data["results"]):
        assert "nsfw_probability" in result
        assert 0 <= result["nsfw_probability"] <= 1
        print(f"Image {i+1} prediction - NSFW: {result['nsfw_probability']:.3f}")


def test_error_handling(base_url: str = "http://localhost:8000") -> None:
    """Test error handling."""
    print("Testing error handling...")
    
    # Test invalid base64.
    payload = {
        "input": {
            "type": "base64",
            "data": "invalid_base64_data"
        }
    }
    
    response = requests.post(f"{base_url}/predict/image", json=payload)
    assert response.status_code == 400
    print("Invalid base64 handling passed.")
    
    # Test empty inputs list.
    payload = {
        "inputs": []
    }
    
    response = requests.post(f"{base_url}/predict/images", json=payload)
    assert response.status_code == 422  # Validation error.
    print("Empty inputs validation passed.")


def run_all_tests(base_url: str = "http://localhost:8000") -> None:
    """Run all tests."""
    print(f"Running API tests against: {base_url}")
    print("=" * 50)
    
    try:
        print("Waiting for server to be ready...")
        for _ in range(30):  # Wait up to 30 seconds.
            try:
                requests.get(f"{base_url}/health/", timeout=1)
                break
            except requests.exceptions.RequestException:
                time.sleep(1)
        else:
            raise Exception("Server not ready after 30 seconds.")
            
        test_health_endpoints(base_url)
        test_image_prediction_base64(base_url)
        test_multiple_images_prediction(base_url)
        test_error_handling(base_url)
        
        print("=" * 50)
        print("All tests passed.")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


def main() -> None:
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    run_all_tests(base_url)


if __name__ == "__main__":
    main()
