import pytest
from fastapi import status

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}

def test_analyze_endpoint_without_image(client):
    """Test the analyze endpoint without an image."""
    response = client.post("/analyze")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_analyze_endpoint_with_image(client, test_image):
    """Test the analyze endpoint with a valid image."""
    files = {"file": ("test_pill.jpg", test_image, "image/jpeg")}
    response = client.post("/analyze", files=files)
    assert response.status_code == status.HTTP_200_OK
    assert "detections" in response.json()

def test_analyze_pill_endpoint_without_image(client):
    """Test the analyze-pill endpoint without an image."""
    response = client.post("/analyze-pill")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_analyze_pill_endpoint_with_image(client, test_image):
    """Test the analyze-pill endpoint with a valid image."""
    files = {"file": ("test_pill.jpg", test_image, "image/jpeg")}
    response = client.post("/analyze-pill", files=files)
    assert response.status_code == status.HTTP_200_OK
    assert "color" in response.json()
    assert "shape" in response.json()
    assert "text" in response.json() 