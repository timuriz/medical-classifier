import React from 'react';
import './ProbabilityBars.css';

function ProbabilityBars({ probabilities }) {
  return (
    <div className="probability-container">
      {Object.entries(probabilities).map(([className, probability]) => {
        const percent = (probability * 100).toFixed(1);
        return (
          <div key={className} className="probability-item">
            <span className="probability-label">{className}</span>
            <div className="probability-bar">
              <div
                className="probability-fill"
                style={{ width: `${percent}%` }}
              ></div>
            </div>
            <span className="probability-value">{percent}%</span>
          </div>
        );
      })}
    </div>
  );
}

export default ProbabilityBars;
