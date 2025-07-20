from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import uuid
import json
from datetime import datetime
import base64
import io
from PIL import Image

# Import your ML model here
# from your_model import VehicleDetector, LicensePlateReader

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Initialize your ML models (replace with your actual model loading)
class MockVehicleDetector:
    def detect_vehicles(self, image_path):
        # Replace this with your actual vehicle detection logic
        return [
            {"bbox": [100, 100, 200, 150], "confidence": 0.95, "class": "car"},
            {"bbox": [300, 200, 400, 280], "confidence": 0.87, "class": "truck"}
        ]

    def extract_license_plates(self, image_path, vehicle_bboxes):
        # Replace this with your actual license plate recognition logic
        return [
            {"text": "ABC123", "confidence": 0.92, "bbox": [120, 130, 180, 145]},
            {"text": "XYZ789", "confidence": 0.88, "bbox": [320, 240, 380, 255]}
        ]


# Initialize model (replace with your actual model)
detector = MockVehicleDetector()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video_file(filename):
    video_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Vehicle Detection API is running"})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{unique_id}.{file_extension}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        return jsonify({
            "message": "File uploaded successfully",
            "file_id": unique_id,
            "filename": filename,
            "is_video": is_video_file(filename)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/process/<file_id>', methods=['POST'])
def process_file(file_id):
    try:
        # Find the uploaded file
        uploaded_file = None
        for ext in ALLOWED_EXTENSIONS:
            potential_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{ext}")
            if os.path.exists(potential_path):
                uploaded_file = potential_path
                break

        if not uploaded_file:
            return jsonify({"error": "File not found"}), 404

        if is_video_file(uploaded_file):
            results = process_video(uploaded_file, file_id)
        else:
            results = process_image(uploaded_file, file_id)

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def process_image(image_path, file_id):
    """Process a single image for vehicle detection and license plate recognition"""

    # Detect vehicles
    vehicles = detector.detect_vehicles(image_path)

    # Extract license plates
    license_plates = detector.extract_license_plates(image_path, vehicles)

    # Draw bounding boxes and save processed image
    processed_image_path = draw_detections(image_path, vehicles, license_plates, file_id)

    # Prepare results
    results = {
        "type": "image",
        "file_id": file_id,
        "timestamp": datetime.now().isoformat(),
        "vehicles_detected": len(vehicles),
        "license_plates_detected": len(license_plates),
        "vehicles": vehicles,
        "license_plates": license_plates,
        "processed_image": f"/api/download/{file_id}_processed.jpg"
    }

    # Save results to JSON file
    results_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def process_video(video_path, file_id):
    """Process a video for vehicle detection and license plate recognition"""

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    all_detections = []
    processed_frames = []

    frame_number = 0
    while cap.read()[0]:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 30th frame (for performance)
        if frame_number % 30 == 0:
            # Save frame temporarily
            temp_frame_path = f"temp_frame_{file_id}_{frame_number}.jpg"
            cv2.imwrite(temp_frame_path, frame)

            # Detect vehicles and license plates
            vehicles = detector.detect_vehicles(temp_frame_path)
            license_plates = detector.extract_license_plates(temp_frame_path, vehicles)

            frame_results = {
                "frame_number": frame_number,
                "timestamp": frame_number / fps,
                "vehicles": vehicles,
                "license_plates": license_plates
            }

            all_detections.append(frame_results)

            # Clean up temp frame
            os.remove(temp_frame_path)

        frame_number += 1

    cap.release()

    # Compile results
    results = {
        "type": "video",
        "file_id": file_id,
        "timestamp": datetime.now().isoformat(),
        "total_frames": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps,
        "frames_processed": len(all_detections),
        "total_vehicles_detected": sum(len(detection["vehicles"]) for detection in all_detections),
        "total_license_plates_detected": sum(len(detection["license_plates"]) for detection in all_detections),
        "detections": all_detections
    }

    # Save results to JSON file
    results_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def draw_detections(image_path, vehicles, license_plates, file_id):
    """Draw bounding boxes on the image and save processed version"""

    # Load image
    image = cv2.imread(image_path)

    # Draw vehicle bounding boxes
    for vehicle in vehicles:
        bbox = vehicle["bbox"]
        confidence = vehicle["confidence"]
        class_name = vehicle["class"]

        # Draw rectangle
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw license plate bounding boxes
    for plate in license_plates:
        bbox = plate["bbox"]
        text = plate["text"]
        confidence = plate["confidence"]

        # Draw rectangle
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        # Draw label
        label = f"{text}: {confidence:.2f}"
        cv2.putText(image, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    # Save processed image
    processed_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_processed.jpg")
    cv2.imwrite(processed_image_path, image)

    return processed_image_path


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download processed images or result files"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/results/<file_id>')
def get_results(file_id):
    """Get processing results for a file"""
    try:
        results_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            return jsonify({"error": "Results not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)