import pytest
import numpy as np
from PIL import Image
import cv2
from main import analyze_color, analyze_shape, analyze_text

def test_analyze_color():
    """Test color analysis function."""
    # Create a test image with a known color
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :] = [255, 0, 0]  # Red color
    
    color = analyze_color(test_image)
    assert isinstance(color, str)
    assert "빨간" in color

def test_analyze_shape():
    """Test shape analysis function."""
    # Create a test image with a circular shape
    test_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(test_image, (50, 50), 40, 255, -1)
    
    shape = analyze_shape(test_image)
    assert isinstance(shape, str)
    assert ("원" in shape) or ("알 수 없음" in shape)

def test_analyze_text():
    """Test text detection function."""
    # Create a test image with text
    test_image = np.zeros((100, 200), dtype=np.uint8)
    cv2.putText(test_image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    text = analyze_text(test_image)
    assert isinstance(text, (str, dict))

def test_invalid_image():
    """Test handling of invalid image input."""
    with pytest.raises(Exception):
        analyze_color(None)
    
    with pytest.raises(Exception):
        analyze_shape(None)
    
    with pytest.raises(Exception):
        analyze_text(None) 