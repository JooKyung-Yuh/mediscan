import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!image) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      const blob = await fetch(image).then(r => r.blob());
      formData.append('image', blob);

      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setDetections(data.detections);
    } catch (error) {
      console.error('Error analyzing image:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>MEDI SCAN</h1>
        <div className="upload-section">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
          />
          <button onClick={analyzeImage} disabled={!image || loading}>
            {loading ? '분석 중...' : '분석하기'}
          </button>
        </div>
        
        {image && (
          <div className="image-container" style={{ position: 'relative' }}>
            <img src={image} alt="Uploaded" style={{ maxWidth: '100%' }} />
            {detections.map((detection, index) => (
              <div
                key={index}
                style={{
                  position: 'absolute',
                  left: `${(detection.x1 / 640) * 100}%`,
                  top: `${(detection.y1 / 640) * 100}%`,
                  width: `${((detection.x2 - detection.x1) / 640) * 100}%`,
                  height: `${((detection.y2 - detection.y1) / 640) * 100}%`,
                  border: '2px solid red',
                  boxSizing: 'border-box',
                }}
              >
                <div
                  style={{
                    position: 'absolute',
                    top: '-20px',
                    left: '0',
                    backgroundColor: 'red',
                    color: 'white',
                    padding: '2px 5px',
                    fontSize: '12px',
                  }}
                >
                  {detection.class_name} ({Math.round(detection.confidence * 100)}%)
                </div>
              </div>
            ))}
          </div>
        )}
      </header>
    </div>
  );
}

export default App; 