from flask import Flask, request, jsonify, send_file
from flask_cors import CORS 
import os
import numpy as np
from werkzeug.utils import secure_filename
import uuid
import json
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
import base64
import io
from PIL import Image
import logging
import traceback

# Import your integrated ML model
from ml_models import ml_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

#MongoDB config
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "toll_system"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 400 * 1024 * 1024  # 100MB max file size


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


#Collections
plates_collection = db["license_plates"]
alerts_collection = db["alerts"]
results_collection = db["processing_results"]

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Toll rates configuration - customize these rates as needed
TOLL_RATES = {
    'car': 50,
    'truck': 100,
    'bus': 80,
    'motorcycle': 20,
    'bicycle': 5,
    'van': 60,
    'suv': 55,
    'auto': 25,
    'rickshaw': 15,
    'tempo': 70,
    'vehicle': 50,  # default for unspecified vehicles
    'unknown': 30   # fallback for unknown types
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video_file(filename):
    video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions


def get_toll_rate(vehicle_type):
    """Get toll rate for a specific vehicle type"""
    vehicle_type = vehicle_type.lower().strip()
    
    # Direct match
    if vehicle_type in TOLL_RATES:
        return TOLL_RATES[vehicle_type]
    
    # Fuzzy matching for common variations
    if 'car' in vehicle_type:
        return TOLL_RATES['car']
    elif 'truck' in vehicle_type or 'lorry' in vehicle_type:
        return TOLL_RATES['truck']
    elif 'bus' in vehicle_type:
        return TOLL_RATES['bus']
    elif 'bike' in vehicle_type or 'motorcycle' in vehicle_type or 'motorbike' in vehicle_type:
        return TOLL_RATES['motorcycle']
    elif 'bicycle' in vehicle_type or 'cycle' in vehicle_type:
        return TOLL_RATES['bicycle']
    elif 'van' in vehicle_type:
        return TOLL_RATES['van']
    elif 'suv' in vehicle_type:
        return TOLL_RATES['suv']
    elif 'auto' in vehicle_type or 'rickshaw' in vehicle_type:
        return TOLL_RATES['auto']
    elif 'tempo' in vehicle_type:
        return TOLL_RATES['tempo']
    else:
        # Default rate for unknown vehicle types
        return TOLL_RATES['unknown']

#to get alerts
def check_plate_alerts(license_plates):
    alerts = []
    for plate in license_plates:
        plate_text = plate.get("text", "").upper().strip()
        if plate_text:
            alert = alerts_collection.find_one({
                "plate_number": plate_text,
                "active": True
            })
            if alert:
                alert["_id"] = str(alert["_id"])
                alerts.append({
                    "plate_text": plate_text,
                    "reason": alert.get("reason", "Stolen vehicle"),
                    "confidence": plate.get("confidence", 0.0)
                })
    return alerts


def calculate_toll_for_vehicles(vehicles):
    """Calculate total toll tax for detected vehicles"""
    vehicle_counts = {}
    total_toll = 0
    
    # Count vehicles by type
    for vehicle in vehicles:
        vehicle_type = vehicle.get("class", "unknown").lower()
        vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
    
    # Calculate toll for each vehicle type
    toll_breakdown = {}
    for vehicle_type, count in vehicle_counts.items():
        toll_rate = get_toll_rate(vehicle_type)  # Use local toll rate function
        toll_amount = toll_rate * count
        toll_breakdown[vehicle_type] = {
            "count": count,
            "rate_per_vehicle": toll_rate,
            "total_amount": toll_amount
        }
        total_toll += toll_amount
    
    return {
        "vehicle_counts": vehicle_counts,
        "toll_breakdown": toll_breakdown,
        "total_toll_amount": total_toll
    }


def get_toll_gate_status(vehicles_detected_count):
    """Determine toll gate status based on vehicle detection"""
    if vehicles_detected_count > 0:
        return {
            "status": "OPEN",
            "message": f"Toll gate is OPEN - {vehicles_detected_count} vehicle(s) detected",
            "action": "Allow passage"
        }
    else:
        return {
            "status": "CLOSED",
            "message": "Toll gate is CLOSED - No vehicles detected",
            "action": "Block passage"
        }


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test if the ML model is working
        model_status = "healthy" if ml_model.detection_system else "error"

        return jsonify({
            "status": "healthy",
            "message": "Vehicle Detection API is running",
            "model_status": model_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": f"API error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload file endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed. Supported: " + ", ".join(ALLOWED_EXTENSIONS)}), 400

        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{unique_id}.{file_extension}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Get file size
        file_size = os.path.getsize(filepath)

        logger.info(f"File uploaded: {filename} -> {unique_filename} ({file_size} bytes)")

        return jsonify({
            "message": "File uploaded successfully",
            "file_id": unique_id,
            "filename": filename,
            "file_size": file_size,
            "is_video": is_video_file(filename),
            "upload_timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/process/<file_id>', methods=['POST'])
def process_file(file_id):
    """Process uploaded file endpoint"""
    try:
        logger.info(f"Processing file with ID: {file_id}")

        options = {}
        if request.is_json:
            options = request.get_json()

        
        # Find the uploaded file
        uploaded_file = None
        original_filename = None

        for ext in ALLOWED_EXTENSIONS:
            potential_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{ext}")
            if os.path.exists(potential_path):
                uploaded_file = potential_path
                original_filename = f"{file_id}.{ext}"
                break

        if not uploaded_file:
            logger.error(f"File not found for ID: {file_id}")
            return jsonify({"error": "File not found"}), 404

        # Get processing options from request
        frame_skip = options.get('frame_skip', 10)  # For video processing
        save_annotated = options.get('save_annotated', True)

        if is_video_file(uploaded_file):
            results = process_video(uploaded_file, file_id, frame_skip, save_annotated)
        else:
            results = process_image(uploaded_file, file_id, save_annotated)

        if results:
            logger.info(f"Processing completed for {file_id}")
            return jsonify(results), 200
        else:
            logger.error(f"Processing failed for {file_id}")
            return jsonify({"error": "Processing failed"}), 500

    except Exception as e:
        logger.error(f"Processing error for {file_id}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def process_image(image_path, file_id, save_annotated=True):
    """Process a single image for vehicle detection and license plate recognition"""
    try:
        logger.info(f"Processing image: {image_path}")

        # Detect vehicles using the integrated model
        vehicles = ml_model.detect_vehicles(image_path)

        # Extract license plates
        license_plates = ml_model.extract_license_plates(image_path, vehicles)

        # Add alert checking
        plate_alerts = check_plate_alerts(license_plates)

        # Get detailed statistics
        stats = ml_model.get_detection_statistics(image_path)

        processed_image_url = None

        if save_annotated:
            # Create annotated image
            processed_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_processed.jpg")
            annotated_path = ml_model.visualize_detections(image_path, processed_image_path)

            if annotated_path and os.path.exists(annotated_path):
                processed_image_url = f"/api/download/{file_id}_processed.jpg"
                logger.info(f"Annotated image saved: {processed_image_path}")

        # getting the mongoDB
        for plate in license_plates:
            if len(license_plates) > 0:
                plate_data = {
                "plate_text": plate["text"],
                "confidence": plate["confidence"],
                "file_id": file_id,
                "timestamp": datetime.now(),
                "vehicle_type": "unknown"  # Could be enhanced with vehicle association
                }
                plates_collection.insert_one(plate_data)
            else:
                logger.info("No license plates detected - skipping database storage")
        
        
        # Calculate toll charges
        toll_info = calculate_toll_for_vehicles(vehicles)

        # Get toll gate status
        toll_gate = get_toll_gate_status(len(vehicles))

         # Enhance vehicle data with toll rates
        enhanced_vehicles = []
        for vehicle in vehicles:
            vehicle_type = vehicle.get("class", "unknown").lower()
            toll_rate = get_toll_rate(vehicle_type)
            enhanced_vehicle = {
                **vehicle,
                "toll_rate": toll_rate,
                "toll_amount": toll_rate  # For single vehicle, toll amount is same as rate
            }
            enhanced_vehicles.append(enhanced_vehicle)
        
        # Prepare results
        results = {
            "type": "image",
            "file_id": file_id,
            "timestamp": datetime.now().isoformat(),
            "vehicles_detected": len(vehicles),
            "license_plates_detected": len(license_plates),
            "processing_status": "completed",          
            # Toll gate status
            "plate_alerts": plate_alerts,
            "has_alerts": len(plate_alerts) > 0,
            "toll_gate": toll_gate,
            "TOLL_STATUS": toll_gate["status"],
            "vehicles": vehicles,
            "license_plates": license_plates,
            "statistics": stats,
            "toll_breakdown": toll_info,  # Include the full toll breakdown
            "processed_image": processed_image_url
        }

        # Save results to JSON file
        results_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Image processing completed: {len(vehicles)} vehicles, {len(license_plates)} plates")
        return results

    except Exception as e:
        logger.error(f"Image processing error: {e}")
        logger.error(traceback.format_exc())
        return None


def process_video(video_path, file_id, frame_skip=10, save_annotated=False):
    """Process a video for vehicle detection and license plate recognition"""
    try:
        logger.info(f"Processing video: {video_path} with frame_skip={frame_skip}")

        # Use the integrated video processing
        video_results = ml_model.process_video_flask(video_path, frame_skip)

        if not video_results:
            logger.error("Video processing failed")
            return None

        # Calculate summary statistics
        total_vehicles = sum(len(detection['vehicles']) for detection in video_results['frame_detections'])
        total_plates = sum(len(detection['license_plates']) for detection in video_results['frame_detections'])
        
        # Collect all vehicles from all frames for toll calculation
        all_vehicles = []
        for frame_detection in video_results['frame_detections']:
            all_vehicles.extend(frame_detection['vehicles'])
        
        # Get toll gate status
        toll_gate = get_toll_gate_status(total_vehicles)

        # getting the MOngoDb
        for plate in video_results.get('unique_plates', []):
            plate_data = {
            "plate_text": plate,
            "confidence": 0.9,  # Default confidence for video plates
            "file_id": file_id,
            "timestamp": datetime.now(),
            "vehicle_type": "unknown"
            }
        plates_collection.insert_one(plate_data)

        plate_alerts = []
        for plate in video_results.get('unique_plates', []):
            alert = alerts_collection.find_one({
                "plate_number": plate.upper().strip(),
                "active": True
            })
            if alert:
                alert["_id"] = str(alert["_id"])
                plate_alerts.append({
                    "plate_text": plate,
                    "reason": alert.get("reason", "Stolen vehicle")
                })
        
        
        # Prepare Flask API results
        results = {
            "type": "video",
            "file_id": file_id,
            "timestamp": datetime.now().isoformat(),
            "processing_status": "completed",
            "video_info": {
                "total_frames": video_results['total_frames'],
                "processed_frames": video_results['processed_frames'],
                "frame_skip": frame_skip
            },
            
            
            # Toll gate status
            "toll_gate": toll_gate,
            
            "plate_alerts": plate_alerts,
            "has_alerts": len(plate_alerts) > 0,
            "summary": {
                "total_vehicles_detected": total_vehicles,
                "total_license_plates_detected": total_plates,
                "unique_vehicle_types": video_results['unique_vehicles'],
                "unique_license_plates": video_results['unique_plates']
            },
            
            "TOLL_STATUS": toll_gate["status"],
            
            "detections": video_results['frame_detections']
        }

        # Save results to JSON file
        results_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Video processing completed: {total_vehicles} total vehicles, {total_plates} total plates")
        return results

    except Exception as e:
        logger.error(f"Video processing error: {e}")
        logger.error(traceback.format_exc())
        return None


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download processed images or result files"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            logger.info(f"Serving file: {filename}")
            return send_file(file_path)
        else:
            logger.warning(f"File not found: {filename}")
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/results/<file_id>')
def get_results(file_id):
    """Get processing results for a file"""
    try:
        results_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            logger.info(f"Serving results for: {file_id}")
            results_collection.insert_one({
                "file_id": file_id,
                "results": results,  # Entire results object
                "timestamp": datetime.now()
            })
            return jsonify(results)
        else:
            logger.warning(f"Results not found for: {file_id}")
            return jsonify({"error": "Results not found"}), 404
    except Exception as e:
        logger.error(f"Results retrieval error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/toll/rates', methods=['GET'])
def get_toll_rates():
    """Get current toll rates for different vehicle types"""
    try:
        return jsonify({
            "toll_rates": TOLL_RATES,
            "currency": "INR",
            "last_updated": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Toll rates retrieval error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/toll/rates', methods=['POST'])
def update_toll_rates():
    """Update toll rates for different vehicle types"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        new_rates = request.get_json()
        
        # Validate that all rates are numeric
        for vehicle_type, rate in new_rates.items():
            if not isinstance(rate, (int, float)) or rate < 0:
                return jsonify({"error": f"Invalid rate for {vehicle_type}. Must be a non-negative number."}), 400
        
        # Update the toll rates
        TOLL_RATES.update(new_rates)
        
        logger.info(f"Toll rates updated: {new_rates}")
        
        return jsonify({
            "message": "Toll rates updated successfully",
            "updated_rates": new_rates,
            "current_rates": TOLL_RATES,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Toll rates update error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/files', methods=['GET'])
def list_files():
    """List all uploaded and processed files"""
    try:
        uploaded_files = []
        processed_files = []

        # List uploaded files
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                if allowed_file(filename):
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file_id = filename.split('.')[0]

                    uploaded_files.append({
                        'file_id': file_id,
                        'filename': filename,
                        'size': os.path.getsize(file_path),
                        'upload_time': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                        'is_video': is_video_file(filename)
                    })

        # List processed files
        if os.path.exists(app.config['OUTPUT_FOLDER']):
            for filename in os.listdir(app.config['OUTPUT_FOLDER']):
                if filename.endswith('_results.json'):
                    file_id = filename.replace('_results.json', '')
                    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)

                    processed_files.append({
                        'file_id': file_id,
                        'results_file': filename,
                        'process_time': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    })

        return jsonify({
            'uploaded_files': uploaded_files,
            'processed_files': processed_files,
            'total_uploaded': len(uploaded_files),
            'total_processed': len(processed_files)
        })

    except Exception as e:
        logger.error(f"File listing error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up temporary and old files"""
    try:
        cleanup_count = 0

        # Clean up ML model temp files
        ml_model.cleanup_temp_files()
        cleanup_count += 1

        # Optionally clean up old files (older than 24 hours)
        import time
        current_time = time.time()
        day_ago = current_time - (24 * 60 * 60)  # 24 hours ago

        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.getctime(file_path) < day_ago:
                        try:
                            os.remove(file_path)
                            cleanup_count += 1
                        except Exception as e:
                            logger.warning(f"Could not remove old file {filename}: {e}")

        return jsonify({
            'status': 'completed',
            'files_cleaned': cleanup_count,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/plates', methods=['GET'])
def get_plates():
    plates = list(plates_collection.find().limit(100).sort("timestamp", -1))
    # Convert ObjectId to string for JSON serialization
    for plate in plates:
        plate["_id"] = str(plate["_id"])
    return jsonify(plates)

# Add new endpoints for alerts
@app.route('/api/alerts', methods=['POST'])
def add_alert():
    data = request.get_json()
    if not data or 'plate_number' not in data:
        return jsonify({"error": "Missing plate_number"}), 400
    
    alert = {
        "plate_number": data['plate_number'],
        "reason": data.get('reason', 'Stolen vehicle'),
        "created_at": datetime.now(),
        "active": True
    }
    result = alerts_collection.insert_one(alert)
    return jsonify({
        "message": "Alert added successfully",
        "alert_id": str(result.inserted_id)
    }), 201

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    alerts = list(alerts_collection.find({"active": True}))
    for alert in alerts:
        alert["_id"] = str(alert["_id"])
    return jsonify(alerts)

@app.route('/api/alerts/<alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    try:
        alerts_collection.update_one(
            {"_id": ObjectId(alert_id)},
            {"$set": {"active": False}}
        )
        return jsonify({"message": "Alert deactivated"})
    except:
        return jsonify({"error": "Invalid alert ID"}), 400

@app.route('/api/check_alert/<plate_text>', methods=['GET'])
def check_alert(plate_text):
    alert = alerts_collection.find_one({
        "plate_number": plate_text,
        "active": True
    })
    
    if alert:
        alert["_id"] = str(alert["_id"])
        return jsonify({
            "is_alert": True,
            "alert": alert
        })
    return jsonify({"is_alert": False})

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 100MB"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    logger.info("Starting Vehicle Detection Flask API...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Output folder: {OUTPUT_FOLDER}")
    logger.info("Current TOLL_RATES configuration:")
    logger.info("API endpoints available:")
    logger.info("  GET  /api/health - Health check")
    logger.info("  POST /api/upload - Upload file")
    logger.info("  POST /api/process/<file_id> - Process file")
    logger.info("  GET  /api/results/<file_id> - Get results")
    logger.info("  GET  /api/download/<filename> - Download file")
    logger.info("  GET  /api/files - List files")
    logger.info("  GET  /api/toll/rates - Get toll rates")
    logger.info("  POST /api/toll/rates - Update toll rates")
    logger.info("  POST /api/cleanup - Cleanup temp files")

    app.run(debug=True, host='0.0.0.0', port=5000)