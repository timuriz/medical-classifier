import React from 'react';
import './ResultsPanel.css';
import PredictionCard from './PredictionCard';
import ProbabilityBars from './ProbabilityBars';

function ResultsPanel({ originalImage, predictions, onUploadAnother }) {
  return (
    <div className="results-panel">
      <div className="results-grid">
        <div className="result-section">
          <h2 className="section-title">Original Image</h2>
          <div className="image-container">
            <img src={originalImage} alt="Original" />
          </div>
        </div>

        <div className="result-section">
          <h2 className="section-title">Prediction</h2>
          <PredictionCard predictions={predictions} />
        </div>

        <div className="result-section full-width">
          <h2 className="section-title">Class Probabilities</h2>
          <ProbabilityBars probabilities={predictions.probabilities} />
        </div>

        <div className="result-section full-width">
          <h2 className="section-title">Grad-CAM Heatmap</h2>
          <div className="image-container large">
            <img
              src={`data:image/png;base64,${predictions.heatmap}`}
              alt="Grad-CAM Heatmap"
            />
          </div>
          <p className="heatmap-explanation">
            Red regions indicate areas the model focused on for the prediction
          </p>
        </div>
      </div>

      <button className="upload-button" onClick={onUploadAnother}>
        ↑ Upload Another Image
      </button>
    </div>
  );
}

export default ResultsPanel;
