import os
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """Test client fixture for FastAPI application."""
    return TestClient(app)

@pytest.fixture
def test_image_path():
    """Fixture that provides path to a test image."""
    return os.path.join(os.path.dirname(__file__), "test_data", "test_pill.jpg")

@pytest.fixture
def test_image():
    """Fixture that provides a test image file."""
    image_path = os.path.join(os.path.dirname(__file__), "test_data", "test_pill.jpg")
    with open(image_path, "rb") as f:
        return f.read() 