import React, { useState } from 'react';
import './App.css';
import UploadArea from './components/UploadArea';
import ResultsPanel from './components/ResultsPanel';

function App() {
  const [originalImage, setOriginalImage] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = async (file) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    // Display original image
    const reader = new FileReader();
    reader.onload = async (e) => {
      setOriginalImage(e.target.result);
      await sendToPredictAPI(file);
    };
    reader.readAsDataURL(file);
  };

  const sendToPredictAPI = async (file) => {
    setLoading(true);
    setError(null);
    setPredictions(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setPredictions(data);
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to get prediction. Make sure the API is running on http://localhost:8000');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>🏥 Medical Image Classifier</h1>
          <p className="subtitle">
            Upload a skin lesion image to get diagnosis prediction with Grad-CAM visualization
          </p>
        </header>

        {error && <div className="error-banner">{error}</div>}

        {!predictions ? (
          <UploadArea onFileUpload={handleFileUpload} loading={loading} />
        ) : (
          <ResultsPanel
            originalImage={originalImage}
            predictions={predictions}
            onUploadAnother={() => {
              setOriginalImage(null);
              setPredictions(null);
              setError(null);
            }}
          />
        )}
      </div>
    </div>
  );
}

export default App;
