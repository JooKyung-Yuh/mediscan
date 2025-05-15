import pytest
import numpy as np
from PIL import Image
import cv2
from main import analyze_color, analyze_shape, detect_text

def test_analyze_color():
    """Test color analysis function."""
    # Create a test image with a known color
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :] = [255, 0, 0]  # Red color
    
    color = analyze_color(test_image)
    assert isinstance(color, str)
    assert "red" in color.lower()

def test_analyze_shape():
    """Test shape analysis function."""
    # Create a test image with a circular shape
    test_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(test_image, (50, 50), 40, 255, -1)
    
    shape = analyze_shape(test_image)
    assert isinstance(shape, str)
    assert "circle" in shape.lower()

def test_detect_text():
    """Test text detection function."""
    # Create a test image with text
    test_image = np.zeros((100, 200), dtype=np.uint8)
    cv2.putText(test_image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    text = detect_text(test_image)
    assert isinstance(text, str)

def test_invalid_image():
    """Test handling of invalid image input."""
    with pytest.raises(Exception):
        analyze_color(None)
    
    with pytest.raises(Exception):
        analyze_shape(None)
    
    with pytest.raises(Exception):
        detect_text(None) 