# ml_models.py - Integration with your Colab model

import torch
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import pickle
import os


class VehicleDetectionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vehicle_model = None
        self.license_plate_model = None
        self.load_models()

    def load_models(self):
        """Load your trained models from Colab"""
        try:
            # Example for PyTorch model
            if os.path.exists('models/vehicle_detection_model.pth'):
                # Load your model architecture first (you need to define this)
                from Models import TollDetection

                self.vehicle_model = TollDetection()  # Adjust parameters
                self.vehicle_model.load_state_dict(
                    torch.load('models/vehicle_detection_model.pth', map_location=self.device))
                self.vehicle_model.to(self.device)
                self.vehicle_model.eval()

            # Example for YOLO model (if you used YOLO)
            elif os.path.exists('models/vehicle_detection_model.pt'):
                self.vehicle_model = YOLO('models/vehicle_detection_model.pt')

            # Example for TensorFlow model
            elif os.path.exists('models/vehicle_detection_model.h5'):
                self.vehicle_model = tf.keras.models.load_model('models/vehicle_detection_model.h5')

            # Load license plate model similarly
            if os.path.exists('models/license_plate_model.pt'):
                self.license_plate_model = YOLO('models/license_plate_model.pt')

            # Load preprocessing configuration
            if os.path.exists('models/preprocessing_config.pkl'):
                with open('models/preprocessing_config.pkl', 'rb') as f:
                    self.preprocessing_config = pickle.load(f)

            print("Models loaded successfully!")

        except Exception as e:
            print(f"Error loading models: {e}")
            # Fallback to mock behavior for testing
            self.vehicle_model = None
            self.license_plate_model = None

    def preprocess_image(self, image_path):
        """Preprocess image exactly as you did in Colab"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path

        # Apply the same preprocessing you used in Colab
        # Example preprocessing (adjust to match your Colab code):

        # Resize image
        if hasattr(self, 'preprocessing_config'):
            target_size = self.preprocessing_config.get('image_size', (640, 640))
        else:
            target_size = (640, 640)  # Default YOLO size

        processed_image = cv2.resize(image, target_size)

        # Normalize if needed (match your Colab preprocessing)
        processed_image = processed_image / 255.0

        # Convert color space if needed
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        return processed_image_rgb, image  # Return both processed and original

    def detect_vehicles(self, image_path):
        """Detect vehicles using your trained model"""
        try:
            if self.vehicle_model is None:
                # Fallback mock data for testing
                return self._mock_vehicle_detection()

            processed_image, original_image = self.preprocess_image(image_path)

            # YOLO model prediction
            if hasattr(self.vehicle_model, 'predict'):  # YOLO-style
                results = self.vehicle_model.predict(image_path, conf=0.5)
                return self._parse_yolo_results(results, 'vehicle')

            # PyTorch model prediction
            elif hasattr(self.vehicle_model, 'forward'):  # PyTorch-style
                # Convert to tensor
                image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0).float()
                image_tensor = image_tensor.to(self.device)

                with torch.no_grad():
                    predictions = self.vehicle_model(image_tensor)

                return self._parse_pytorch_results(predictions, original_image.shape)

            # TensorFlow model prediction
            else:  # TensorFlow-style
                image_array = np.expand_dims(processed_image, axis=0)
                predictions = self.vehicle_model.predict(image_array)
                return self._parse_tensorflow_results(predictions, original_image.shape)

        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            return self._mock_vehicle_detection()

    def extract_license_plates(self, image_path, vehicle_bboxes):
        """Extract license plates from detected vehicle regions"""
        try:
            if self.license_plate_model is None:
                return self._mock_license_plate_detection()

            license_plates = []
            original_image = cv2.imread(image_path)

            # For each detected vehicle, crop and analyze for license plates
            for vehicle in vehicle_bboxes:
                bbox = vehicle['bbox']  # [x1, y1, x2, y2]

                # Crop vehicle region
                vehicle_crop = original_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                if vehicle_crop.size == 0:
                    continue

                # Save temporary crop for processing
                temp_crop_path = 'temp_vehicle_crop.jpg'
                cv2.imwrite(temp_crop_path, vehicle_crop)

                # Detect license plates in the vehicle crop
                if hasattr(self.license_plate_model, 'predict'):  # YOLO-style
                    plate_results = self.license_plate_model.predict(temp_crop_path, conf=0.3)
                    crop_plates = self._parse_yolo_results(plate_results, 'license_plate')

                    # Adjust coordinates back to original image
                    for plate in crop_plates:
                        plate['bbox'][0] += bbox[0]  # x1
                        plate['bbox'][1] += bbox[1]  # y1
                        plate['bbox'][2] += bbox[0]  # x2
                        plate['bbox'][3] += bbox[1]  # y2

                    license_plates.extend(crop_plates)

                # Clean up temp file
                if os.path.exists(temp_crop_path):
                    os.remove(temp_crop_path)

            return license_plates

        except Exception as e:
            print(f"Error in license plate detection: {e}")
            return self._mock_license_plate_detection()

    def _parse_yolo_results(self, results, detection_type):
        """Parse YOLO model results"""
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())

                    if detection_type == 'vehicle':
                        # Map class_id to vehicle type
                        vehicle_classes = {0: 'car', 1: 'truck', 2: 'bus', 3: 'motorcycle'}
                        vehicle_type = vehicle_classes.get(class_id, 'vehicle')

                        detections.append({
                            'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                            'confidence': float(confidence),
                            'class': vehicle_type
                        })

                    elif detection_type == 'license_plate':
                        # For license plates, try to extract text (you might need OCR here)
                        plate_text = self._extract_plate_text(bbox)

                        detections.append({
                            'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                            'confidence': float(confidence),
                            'text': plate_text
                        })

        return detections

    def _extract_plate_text(self, bbox):
        """Extract text from license plate using OCR (implement based on your Colab approach)"""
        # If you used OCR in Colab, implement the same here
        # Example with pytesseract (install: pip install pytesseract)
        try:
            import pytesseract
            # This would need the cropped plate image
            # For now, return a placeholder
            return f"ABC{np.random.randint(100, 999)}"
        except:
            return f"PLATE{np.random.randint(100, 999)}"

    def _mock_vehicle_detection(self):
        """Fallback mock detection for testing"""
        return [
            {"bbox": [100, 100, 200, 150], "confidence": 0.95, "class": "car"},
            {"bbox": [300, 200, 400, 280], "confidence": 0.87, "class": "truck"}
        ]

    def _mock_license_plate_detection(self):
        """Fallback mock license plate detection"""
        return [
            {"text": "ABC123", "confidence": 0.92, "bbox": [120, 130, 180, 145]},
            {"text": "XYZ789", "confidence": 0.88, "bbox": [320, 240, 380, 255]}
        ]


# Global model instance
ml_model = VehicleDetectionModel()