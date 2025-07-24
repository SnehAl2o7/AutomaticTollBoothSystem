import React, { useState, useEffect, useCallback } from 'react';
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
  const [tollRates, setTollRates] = useState(null);
  const [showTollRates, setShowTollRates] = useState(false);
  const[view, setView] = useState("main");
  const[activeAlert, setActiveAlert] = useState(null);
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

  //fetch toll rates
  const fetchTollRates = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/toll/rates`);
      setTollRates(response.data.toll_rates);
      setShowTollRates(true);
    } catch (error) {
      setError('Failed to fetch toll rates');
    }
  };


  // Clear all data
  const handleClear = () => {
    setSelectedFiles([]);
    setResults(null);
    setError('');
    setUploadProgress(0);
  };

  // checking for alert
  useEffect(() => {
    const checkForAlerts = async () => {
      if (!results) return;
      
      try {
        // Check all detected plates in results
        for (const result of results) {
          if (result.license_plates && result.license_plates.length > 0) {
            for (const plate of result.license_plates) {
              const response = await axios.get(
                `${API_BASE_URL}/check_alert/${plate.text}`
              );
              if (response.data.is_alert) {
                setActiveAlert(response.data.alert);
                break;
              }
            }
          }
        }
      } catch (error) {
        console.error("Error checking alerts:", error);
      }
    };

    checkForAlerts();
  }, [results]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>🚗 AUTOMATIC TOLL BOOTH SYSTEM </h1>
        <p>Upload images or videos to Detect Vehicle and License PLate and Open the Toll</p>
        <div className="nav-buttons">
          <button 
            onClick={() => setView("main")}
            className={view === "main" ? "active" : ""}
          >
            Main System
          </button>
          <button 
            onClick={() => setView("plates")}
            className={view === "plates" ? "active" : ""}
          >
            License Plate Database
          </button>
          <button 
            onClick={() => setView("alerts")}
            className={view === "alerts" ? "active" : ""}
          >
            Alert System
          </button>
        </div>
      
      </header>

      <main className="main-content">
        {/* File Upload Section */}
        {view === "main" && (

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
              onClick={fetchTollRates}
              className="toll-rates-button"
            >
              View Toll Rates
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
        )}

        {view === "plates" && <LicensePlatesView />}
        {view === "alerts" && <AlertSystem />}

        {results && results.some(result => result.has_alerts) && (
          <div className="alert-popup">
            <div className="alert-content">
              <h3>🚨 SECURITY ALERT 🚨</h3>
              {results.map((result, resultIndex) => (
                result.plate_alerts && result.plate_alerts.map((alert, alertIndex) => (
                  <div key={`${resultIndex}-${alertIndex}`} className="alert-detail">
                    <p>License Plate: <strong>{alert.plate_text}</strong></p>
                    <p>Reason: {alert.reason}</p>
                    <p>Detected in: {result.filename}</p>
                    <p>Confidence: {(alert.confidence * 100).toFixed(1)}%</p>
                  </div>
                ))
              ))}
              <button onClick={() => setResults(null)}>Acknowledge</button>
            </div>
          </div>
        )}

        
         {/* Alert popup */}
        {activeAlert && (
          <div className="alert-popup">
            <div className="alert-content">
              <h3>🚨 ALERT 🚨</h3>
              <p>License Plate: {activeAlert.plate_number}</p>
              <p>Reason: {activeAlert.reason}</p>
              <p>This vehicle has been flagged!</p>
              <button onClick={() => setActiveAlert(null)}>Acknowledge</button>
            </div>
          </div>
        )}

        {/* The is the toll rate section */}
        {showTollRates && (
          <div className="toll-rates-section">
            <h2>💰 Current Toll Rates</h2>
            <button 
              onClick={() => setShowTollRates(false)}
              className="close-button"
            >
              Close
            </button>
            <div className="toll-rates-grid">
              {Object.entries(tollRates).map(([vehicleType, rate]) => (
                <div key={vehicleType} className="toll-rate-item">
                  <span className="vehicle-type">{vehicleType.toUpperCase()}:</span>
                  <span className="rate-value">₹{rate}</span>
                </div>
              ))}
            </div>
          </div>
        )}


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
                      <div className="vehicle-stats">
      <div className="stat-item">
        <span className="stat-label">Total Vehicles:</span>
        <span className="stat-value">{result.vehicles.length}</span>
      </div>
      <div className="stat-item">
        <span className="stat-label">Total Toll Due:</span>
        <span className="stat-value">₹{result.vehicles.reduce((sum, vehicle) => sum + (vehicle.toll_amount || 0), 0)}</span>
      </div>
    </div>
    <div className="detection-list">
      {result.vehicles.map((vehicle, vIndex) => (
        <div key={vIndex} className="detection-item vehicle-item">
          <div className="vehicle-header">
            <span className="vehicle-type">{vehicle.class.toUpperCase()}</span>
            <span className="toll-amount">₹{vehicle.toll_amount || 'N/A'}</span>
          </div>
          <div className="vehicle-details">
            <div className="detail-row">
              <span className="detail-label">Confidence:</span>
              <span className="detail-value">{(vehicle.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Position:</span>
              <span className="detail-value">
                [{vehicle.bbox[0]}, {vehicle.bbox[1]}] to [{vehicle.bbox[2]}, {vehicle.bbox[3]}]
              </span>
            </div>
            {vehicle.license_plates && vehicle.license_plates.length > 0 && (
              <div className="detail-row">
                <span className="detail-label">License Plate:</span>
                <span className="detail-value plate">
                  {vehicle.license_plates[0].text}
                  <span className="plate-confidence">
                    ({(vehicle.license_plates[0].confidence * 100).toFixed(1)}%)
                  </span>
                </span>
              </div>
            )}
          </div>
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
                {/*for the toll rates section*/}
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



// Adding License Plate View and Alert System

function LicensePlatesView() {
  const [plates, setPlates] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchPlates = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/plates`);
      setPlates(response.data);
    } catch (error) {
      console.error("Error fetching plates:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPlates();
  }, []);

  return (
    <div className="plates-view">
      <h2>🚗 License Plate Database</h2>
      <button onClick={fetchPlates} className="refresh-button">
        🔄 Refresh
      </button>
      
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div className="plates-grid">
          {plates.map((plate, index) => (
            <div key={index} className="plate-card">
              <div className="plate-number">{plate.plate}</div>
              <div className="plate-details">
                <span>Vehicle: {plate.vehicle_type}</span>
                <span>Detected: {new Date(plate.timestamp).toLocaleString()}</span>
                <a 
                  href={`http://localhost:5000/api/download/${plate.file_id}_processed.jpg`} 
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  View Image
                </a>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AlertSystem() {
  const [alerts, setAlerts] = useState([]);
  const [newPlate, setNewPlate] = useState("");
  const [reason, setReason] = useState("Stolen vehicle");
  const [activeAlert, setActiveAlert] = useState(null);

  const fetchAlerts = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/alerts`);
      setAlerts(response.data);
    } catch (error) {
      console.error("Error fetching alerts:", error);
    }
  };

  const addAlert = async () => {
    if (!newPlate.trim()) return;
    
    try {
      await axios.post(`${API_BASE_URL}/alerts`, {
        plate_number: newPlate,
        reason: reason
      });
      setNewPlate("");
      fetchAlerts();
    } catch (error) {
      console.error("Error adding alert:", error);
    }
  };

  const deleteAlert = async (alertId) => {
    try {
      await axios.delete(`${API_BASE_URL}/alerts/${alertId}`);
      fetchAlerts();
    } catch (error) {
      console.error("Error deleting alert:", error);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, []);

  return (
    <div className="alert-system">
      <h2>🚨 Alert Management</h2>
      
      <div className="alert-form">
        <input
          type="text"
          value={newPlate}
          onChange={(e) => setNewPlate(e.target.value)}
          placeholder="Enter license plate"
        />
        <select value={reason} onChange={(e) => setReason(e.target.value)}>
          <option value="Stolen vehicle">Stolen vehicle</option>
          <option value="Wanted criminal">Wanted criminal</option>
          <option value="Unpaid fines">Unpaid fines</option>
          <option value="Suspicious activity">Suspicious activity</option>
        </select>
        <button onClick={addAlert}>Add Alert</button>
      </div>

      <div className="alert-list">
        <h3>Active Alerts ({alerts.length})</h3>
        {alerts.map((alert, index) => (
          <div key={index} className="alert-item">
            <span className="plate">{alert.plate_number}</span>
            <span className="reason">{alert.reason}</span>
            <span className="date">
              {new Date(alert.created_at).toLocaleDateString()}
            </span>
            <button 
              className="delete-alert"
              onClick={() => deleteAlert(alert._id)}
            >
              Delete
            </button>
          </div>
        ))}
      </div>
    
      <div className="alert-list">
        <h3>Active Alerts ({alerts.length})</h3>
        {alerts.map((alert, index) => (
          <div key={index} className="alert-item">
            <span className="plate">{alert.plate_number}</span>
            <span className="reason">{alert.reason}</span>
            <span className="date">
              {new Date(alert.created_at).toLocaleDateString()}
            </span>
            <button 
              className="delete-alert"
              onClick={() => deleteAlert(alert._id)}
            >
              Delete
            </button>
          </div>
        ))}
      </div>

      {activeAlert && (
        <div className="alert-popup">
          <div className="alert-content">
            <h3>🚨 ALERT 🚨</h3>
            <p>License Plate: {activeAlert.plate_number}</p>
            <p>Reason: {activeAlert.reason}</p>
            <button onClick={() => setActiveAlert(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;