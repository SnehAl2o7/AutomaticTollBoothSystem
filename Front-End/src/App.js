import React, { useState, useCallback } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:5000/api';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);

  // Handle file selection
  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    const validFiles = files.filter(file => {
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'video/mp4', 'video/avi', 'video/mov'];
      return validTypes.includes(file.type) && file.size <= 400 * 1024 * 1024; // 100MB limit
    });

    if (validFiles.length !== files.length) {
      setError('Some files were rejected. Please ensure files are images (JPG, PNG) or videos (MP4, AVI, MOV) under 100MB.');
    } else {
      setError('');
    }

    setSelectedFiles(validFiles);
    setResults(null);
  };

  // Handle drag and drop
  const handleDrop = useCallback((event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    const validTypes = [
      'image/jpeg',
      'image/jpg',
      'image/png',
      'video/mp4',
      'video/quicktime',      // .mov
      'video/x-msvideo',      // .avi
      'video/webm',
      'video/x-matroska'      // .mkv
    ];
    const validFiles = files.filter(file => {
      return validTypes.includes(file.type) && file.size <= 100 * 1024 * 1024;
    });
    setSelectedFiles(validFiles);
    setResults(null);
  }, []);

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  // Upload and process files
  const handleProcess = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one file');
      return;
    }

    setUploading(true);
    setProcessing(false);
    setError('');
    setResults(null);

    try {
      const processedResults = [];

      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        setUploadProgress(((i + 1) / selectedFiles.length) * 50); // Upload progress (50% of total)

        // Upload file
        const formData = new FormData();
        formData.append('file', file);

        const uploadResponse = await axios.post(`${API_BASE_URL}/upload`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        const { file_id } = uploadResponse.data;

        // Process file
        setUploading(false);
        setProcessing(true);
        setUploadProgress(50 + ((i + 1) / selectedFiles.length) * 50); // Processing progress

        const processResponse = await axios.post(`${API_BASE_URL}/process/${file_id}`);
        processedResults.push({
          filename: file.name,
          ...processResponse.data
        });
      }

      setResults(processedResults);
      setProcessing(false);
      setUploadProgress(100);

    } catch (error) {
      setError(error.response?.data?.error || 'An error occurred during processing');
      setUploading(false);
      setProcessing(false);
      setUploadProgress(0);
    }
  };

  // Clear all data
  const handleClear = () => {
    setSelectedFiles([]);
    setResults(null);
    setError('');
    setUploadProgress(0);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🚗 AUTOMATIC TOLL BOOTH SYSTEM </h1>
        <p>Upload images or videos to Detect Vehicle and License PLate and Open the Toll</p>
      </header>

      <main className="main-content">
        {/* File Upload Section */}
        <div className="upload-section">
          <div 
            className="drop-zone"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <input
              type="file"
              multiple
              accept="image/*,video/*"
              onChange={handleFileSelect}
              className="file-input"
              id="file-input"
            />
            <label htmlFor="file-input" className="file-label">
              <div className="drop-content">
                <span className="upload-icon">📁</span>
                <p>Drag & drop files here or click to browse</p>
                <p className="file-types">Supported: JPG, JPEG, PNG, MP4, AVI, MOV (max 100MB each)</p>
              </div>
            </label>
          </div>

          {/* Selected Files Display */}
          {selectedFiles.length > 0 && (
            <div className="selected-files">
              <h3>Selected Files ({selectedFiles.length})</h3>
              <div className="file-list">
                {selectedFiles.map((file, index) => (
                  <div key={index} className="file-item">
                    <span className="file-icon">
                      {file.type.startsWith('image/') ? '🖼️' : '🎥'}
                    </span>
                    <span className="file-name">{file.name}</span>
                    <span className="file-size">({(file.size / (1024*1024)).toFixed(2)} MB)</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="button-section">
            <button 
              onClick={handleProcess}
              disabled={selectedFiles.length === 0 || uploading || processing}
              className="process-button"
            >
              {uploading ? 'Uploading...' : processing ? 'Processing...' : 'Process Files'}
            </button>
            <button 
              onClick={handleClear}
              className="clear-button"
            >
              Clear All
            </button>
          </div>

          {/* Progress Bar */}
          {(uploading || processing) && (
            <div className="progress-section">
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <p className="progress-text">
                {uploading ? 'Uploading files...' : 'Processing with ML model...'} {uploadProgress.toFixed(0)}%
              </p>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}
        </div>

        {/* Results Section */}
        {results && (
          <div className="results-section">
            <h2 >🎯 Detection Results</h2>
            {results.map((result, index) => (
              <div key={index} className="result-item">
                <h3>📄 {result.filename}</h3>
                
                {/* Summary Stats */}
                <div className="stats-grid">
                  <div className="stat-item">
                    <span className="stat-label">Vehicles Detected:</span>
                    <span className="stat-value">{result.vehicles_detected}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">License Plates Read:</span>
                    <span className="stat-value">{result.license_plates_detected}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">File Type:</span>
                    <span className="stat-value">{result.type}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">TOLL GATE:</span>
                    <span className="stat-value">{result.TOLL_STATUS}</span>
                  </div>
                  {result.type === 'video' && (
                    <div className="stat-item">
                      <span className="stat-label">Duration:</span>
                      <span className="stat-value">{result.duration_seconds?.toFixed(1)}s</span>
                    </div>
                  )}
                </div>

                {/* Processed Image (for images only) */}
                {result.type === 'image' && result.processed_image && (
                  <div className="processed-image">
                    <h4>Processed Image with Detections:</h4>
                    <img 
                      src={`http://localhost:5000${result.processed_image}`} 
                      alt="Processed with detections"
                      className="result-image"
                      style={{ maxWidth: '100%', border: '2px solid #ccc', marginTop: '10px' }}
                    />
                  </div>
                )}

                {/* Detection Details */}
                {result.vehicles && result.vehicles.length > 0 && (
                  <div className="detection-details">
                    <h4>🚗 Vehicle Detections:</h4>
                    <div className="detection-list">
                      {result.vehicles.map((vehicle, vIndex) => (
                        <div key={vIndex} className="detection-item">
                          <span className="detection-type">Type:</span>
                          <span className="detection-value">{vehicle.class}</span><br></br>
                          <span className="detection-confidence">
                            Confidence: {(vehicle.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {result.license_plates && result.license_plates.length > 0 && (
                  <div className="detection-details">
                    <h4>🔢 License Plate Readings:</h4>
                    <div className="detection-list">
                      {result.license_plates.map((plate, pIndex) => (
                        <div key={pIndex} className="detection-item plate-item">
                          <span className="plate-text">{plate.text}</span>
                          <span className="detection-confidence">
                            Confidence: {(plate.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Video Timeline (for videos) */}
                {result.type === 'video' && result.detections && (
                  <div className="video-timeline">
                    <h4>📹 Detection Timeline:</h4>
                    <div className="timeline-list">
                      {result.detections.slice(0, 10).map((detection, dIndex) => (
                        <div key={dIndex} className="timeline-item">
                          <span className="timeline-time">
                            {detection.timestamp?.toFixed(1)}s
                          </span>
                          <span className="timeline-vehicles">
                            {detection.vehicles.length} vehicles
                          </span>
                          <span className="timeline-plates">
                            {detection.license_plates.length} plates
                          </span>
                        </div>
                      ))}
                      {result.detections.length > 10 && (
                        <p className="timeline-more">
                          ... and {result.detections.length - 10} more detection frames
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by Snehal Singh Bisht • Vehicle Detection & License Plate Recognition</p>
      </footer>
    </div>
  );
}


export default App;