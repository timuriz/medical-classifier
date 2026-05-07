import React from 'react';
import './PredictionCard.css';

function PredictionCard({ predictions }) {
  const confidencePercent = (predictions.confidence * 100).toFixed(1);

  return (
    <div className="prediction-card">
      <div className="prediction-label">Diagnosis</div>
      <div className="prediction-class">
        {predictions.prediction.charAt(0).toUpperCase() + predictions.prediction.slice(1)}
      </div>
      <div className="confidence-section">
        <div className="confidence-label">Confidence</div>
        <div className="confidence-value">{confidencePercent}%</div>
      </div>
    </div>
  );
}

export default PredictionCard;
