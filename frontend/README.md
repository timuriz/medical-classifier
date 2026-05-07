# React Frontend Setup Instructions

## Prerequisites
- Node.js 14+ (with npm)

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### Development Mode

Make sure your FastAPI backend is running:
```bash
python3 backend/api.py
```

Then in a new terminal, start the React dev server:
```bash
npm start
```

This will open the app at `http://localhost:3000` and it will automatically reload when you make changes.

### Production Build

To create an optimized production build:
```bash
npm run build
```

The build folder will contain your optimized app ready for deployment.

## Project Structure

```
frontend/
├── public/
│   └── index.html           # HTML entry point
├── src/
│   ├── components/
│   │   ├── UploadArea.js    # File upload component
│   │   ├── UploadArea.css
│   │   ├── ResultsPanel.js  # Results display component
│   │   ├── ResultsPanel.css
│   │   ├── PredictionCard.js # Prediction display
│   │   ├── PredictionCard.css
│   │   ├── ProbabilityBars.js # Probability visualization
│   │   └── ProbabilityBars.css
│   ├── App.js               # Main app component
│   ├── App.css
│   ├── index.js             # React entry point
│   └── index.css
└── package.json             # Dependencies
```

## How It Works

1. **UploadArea Component**: Handles file uploads with drag-and-drop
2. **App Component**: Manages state and API communication
3. **ResultsPanel Component**: Displays predictions and heatmap
4. **PredictionCard**: Shows diagnosis and confidence
5. **ProbabilityBars**: Visualizes class probabilities

## API Integration

The app communicates with your FastAPI backend at `http://localhost:8000/predict`

- Sends: Image file via multipart form
- Receives: JSON with prediction, confidence, probabilities, and base64 heatmap

## Troubleshooting

**CORS Error?** Make sure your FastAPI backend has CORS enabled (it does by default in api.py)

**Can't connect to API?** Check that:
- Backend is running on `http://localhost:8000`
- Frontend is running on `http://localhost:3000`
- Both are on the same machine
