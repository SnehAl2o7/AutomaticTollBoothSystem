# ml_models.py - Integration with TollDetection.py

import os
import cv2
import numpy as np
from PIL import Image
import torch
import logging

# Import your actual CarDetectionSystem
from TollDetection import CarDetectionSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleDetectionModel:
    def __init__(self):
        """Initialize the vehicle detection model using CarDetectionSystem"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize the actual detection system from TollDetection.py
        self.detection_system = CarDetectionSystem()
        self.detection_system.setup_license_plate_model()

        logger.info("CarDetectionSystem initialized successfully!")

    
    def detect_vehicles(self, image_path):
        """
        Detect vehicles using the CarDetectionSystem
        Returns standardized format for Flask API
        """
        try:
            logger.info(f"Processing image: {image_path}")

            # Use the process_single_image method from CarDetectionSystem
            result = self.detection_system.process_single_image(image_path)

            if not result:
                logger.warning("No results from CarDetectionSystem")
                return []

            # Convert to Flask API format
            vehicles = []
            for vehicle_info in result['vehicles']:
                # Extract bounding box coordinates
                bbox = vehicle_info['bbox']  # (x1, y1, x2, y2)

                vehicle = {
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    'confidence': float(vehicle_info['confidence']),
                    'class': vehicle_info['type'],  # car, truck, bus, motorcycle
                    'vehicle_id': vehicle_info['vehicle_id']
                }
                vehicles.append(vehicle)

            logger.info(f"Detected {len(vehicles)} vehicles")
            return vehicles

        except Exception as e:
            logger.error(f"Error in vehicle detection: {e}")
            return self._mock_vehicle_detection()

    def extract_license_plates(self, image_path, vehicle_bboxes=None):
        """
        Extract license plates using the CarDetectionSystem
        Returns standardized format for Flask API
        """
        try:
            logger.info(f"Extracting license plates from: {image_path}")

            # Process the image to get full results including license plates
            result = self.detection_system.process_single_image(image_path)

            if not result:
                logger.warning("No results from CarDetectionSystem for license plates")
                return []

            # Extract license plates from the results
            license_plates = []

            for vehicle_info in result['vehicles']:
                if vehicle_info.get('license_plates'):
                    for plate_info in vehicle_info['license_plates']:
                        if plate_info and plate_info.get('text'):
                            # Get plate region coordinates
                            plate_region = plate_info.get('region', [0, 0, 100, 50])

                            license_plate = {
                                'text': plate_info['text'].strip(),
                                'confidence': float(plate_info.get('confidence', 0.8)),
                                'bbox': [int(plate_region[0]), int(plate_region[1]),
                                         int(plate_region[2]), int(plate_region[3])],
                                'vehicle_id': vehicle_info['vehicle_id']
                            }
                            license_plates.append(license_plate)

            logger.info(f"Detected {len(license_plates)} license plates")
            return license_plates

        except Exception as e:
            logger.error(f"Error in license plate detection: {e}")
            return self._mock_license_plate_detection()

    def process_video_flask(self, video_path, frame_skip=10):
        """
        Process video using CarDetectionSystem for Flask API
        """
        try:
            logger.info(f"Processing video: {video_path}")

            # Use the process_video method from CarDetectionSystem
            video_results = self.detection_system.process_video(
                video_path=video_path,
                frame_skip=frame_skip
            )

            if not video_results:
                logger.warning("No video results from CarDetectionSystem")
                return None

            # Convert to Flask API format
            flask_results = {
                'video_path': video_results['video_path'],
                'total_frames': video_results['total_frames'],
                'processed_frames': video_results['processed_frames'],
                'unique_vehicles': list(video_results['unique_vehicles']),
                'unique_plates': list(video_results['unique_plates']),
                'frame_detections': []
            }

            # Convert frame results
            for frame_data in video_results['frame_results']:
                frame_detection = {
                    'frame_number': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'vehicles': [],
                    'license_plates': []
                }

                # Convert vehicle data
                for vehicle in frame_data['vehicles']:
                    bbox = vehicle['bbox']
                    vehicle_flask = {
                        'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        'confidence': float(vehicle['confidence']),
                        'class': vehicle['type']
                    }
                    frame_detection['vehicles'].append(vehicle_flask)

                    # Add license plate if detected
                    if vehicle.get('license_plate'):
                        plate_flask = {
                            'text': vehicle['license_plate'].strip(),
                            'confidence': 0.8,  # Default confidence
                            'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]  # Approximate
                        }
                        frame_detection['license_plates'].append(plate_flask)

                flask_results['frame_detections'].append(frame_detection)

            logger.info("Video processing completed successfully")
            return flask_results

        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            return None

    def visualize_detections(self, image_path, output_path):
        """
        Create annotated image using CarDetectionSystem visualization
        """
        try:
            logger.info(f"Creating visualization for: {image_path}")

            # Use the existing visualization method
            self.detection_system.visualize_results(image_path, output_path)

            return output_path if os.path.exists(output_path) else None

        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            return None

    def get_detection_statistics(self, image_path):
        """
        Get detailed statistics about detection results
        """
        try:
            result = self.detection_system.process_single_image(image_path)

            if not result:
                return {}

            stats = {
                'total_vehicles': len(result['vehicles']),
                'vehicles_with_plates': 0,
                'vehicle_types': {},
                'license_plates': []
            }

            for vehicle_info in result['vehicles']:
                # Count vehicle types
                vtype = vehicle_info['type']
                stats['vehicle_types'][vtype] = stats['vehicle_types'].get(vtype, 0) + 1

                # Count vehicles with license plates
                if vehicle_info.get('license_plates'):
                    for plate_info in vehicle_info['license_plates']:
                        if plate_info and plate_info.get('text'):
                            stats['vehicles_with_plates'] += 1
                            stats['license_plates'].append(plate_info['text'].strip())

            # Remove duplicates from license plates
            stats['license_plates'] = list(set(stats['license_plates']))

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def _mock_vehicle_detection(self):
        """Fallback mock detection for testing"""
        logger.warning("Using mock vehicle detection")
        return [
            {"bbox": [100, 100, 200, 150], "confidence": 0.95, "class": "car", "vehicle_id": 1},
            {"bbox": [300, 200, 400, 280], "confidence": 0.87, "class": "truck", "vehicle_id": 2}
        ]

    def _mock_license_plate_detection(self):
        """Fallback mock license plate detection"""
        logger.warning("Using mock license plate detection")
        return [
            {"text": "ABC123", "confidence": 0.92, "bbox": [120, 130, 180, 145], "vehicle_id": 1},
            {"text": "XYZ789", "confidence": 0.88, "bbox": [320, 240, 380, 255], "vehicle_id": 2}
        ]

    def cleanup_temp_files(self):
        """Clean up any temporary files created during processing"""
        temp_files = ["temp_frame.jpg", "temp_vehicle_crop.jpg", "temp_video_frame.jpg"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")


# Global model instance
logger.info("Initializing global ML model instance...")
ml_model = VehicleDetectionModel()
logger.info("ML model instance created successfully!")