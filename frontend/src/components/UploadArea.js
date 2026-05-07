import React from 'react';
import './UploadArea.css';

function UploadArea({ onFileUpload, loading }) {
  const [isDragActive, setIsDragActive] = React.useState(false);
  const inputRef = React.useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true);
    } else if (e.type === 'dragleave') {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      onFileUpload(files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileUpload(e.target.files[0]);
    }
  };

  return (
    <div
      className={`upload-area ${isDragActive ? 'active' : ''} ${loading ? 'disabled' : ''}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => !loading && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        disabled={loading}
        style={{ display: 'none' }}
      />

      {loading ? (
        <>
          <div className="spinner"></div>
          <p className="loading-text">Analyzing image...</p>
        </>
      ) : (
        <>
          <div className="upload-icon">📤</div>
          <div className="upload-text">Click to upload or drag and drop</div>
          <div className="upload-subtext">PNG, JPG, GIF up to 10MB</div>
        </>
      )}
    </div>
  );
}

export default UploadArea;
