import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
import pandas as pd
from PIL import Image
import re
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TollDetectionSystem:
    def __init__(self, vehicle_model_path='yolov8l.pt', license_plate_model_path=None):
        """
        Initialize the toll detection system with vehicle and license plate models.
        Args:
            vehicle_model_path (str): Path to the YOLO vehicle detection model.
            license_plate_model_path (str): Path to the YOLO license plate detection model.
        """
        logger.info("Initializing TollDetectionSystem...")

        # Initialize YOLOv8 models
        try:
            self.vehicle_model = YOLO(vehicle_model_path)
            self.vehicle_model.verbose = False
            self.vehicle_model.conf = 0.6  # Higher confidence threshold
            self.vehicle_model.iou = 0.45  # Intersection over Union threshold
            logger.info(f"Vehicle detection model loaded successfully from {vehicle_model_path}")
        except Exception as e:
            logger.error(f"Failed to load vehicle model from {vehicle_model_path}: {e}")
            self.vehicle_model = None

        # Initialize OCR reader with optimized parameters
        try:
            self.ocr_reader = easyocr.Reader(
                ['en'],
                gpu=True,
                model_storage_directory='models',
                user_network_directory='models',
                download_enabled=True,
                recognizer=True,
                detector=True,
                verbose=False,
                quantize=True
            )
            logger.info("OCR reader initialized successfully with GPU")
        except Exception as e:
            logger.warning(f"Failed to initialize OCR with GPU, trying CPU: {e}")
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("OCR reader initialized with CPU")
            except Exception as e2:
                logger.error(f"Failed to initialize OCR: {e2}")
                self.ocr_reader = None

        # Vehicle class mapping for YOLO COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        # Initialize license plate model (optional)
        self.license_plate_model = None
        if license_plate_model_path:
            try:
                self.license_plate_model = YOLO(license_plate_model_path)
                self.license_plate_model.verbose = False
                logger.info(f"License plate model loaded successfully from {license_plate_model_path}")
            except Exception as e:
                logger.error(f"Failed to load license plate model: {e}")

        # Results storage
        self.results = []

    def detect_vehicles(self, image_input):
        """
        Detect vehicles in the image.
        Args:
            image_input (str or np.array): Path to the image or a numpy array of the image.
        Returns:
            tuple: (original_image, list_of_detected_vehicles)
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                logger.error(f"Could not load image: {image_input}")
                return None, []
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            logger.error("Invalid image input type. Must be a path (str) or a numpy array.")
            return None, []

        if self.vehicle_model is None:
            logger.error("Vehicle detection model not loaded. Cannot detect vehicles.")
            return image, []

        results = self.vehicle_model(image, verbose=False)

        detected_vehicles = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id in self.vehicle_classes and confidence > self.vehicle_model.conf:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        vehicle_type = self.vehicle_classes[class_id]

                        detected_vehicles.append({
                            'type': vehicle_type,
                            'confidence': confidence,
                            'bbox': (int(x1), int(y1), int(x2), int(y2))
                        })

        return image, detected_vehicles

    def extract_license_plate_region(self, image, vehicle_bbox):
        """
        Extracts license plate regions using dedicated model or enhanced region-based detection.
        Args:
            image (np.array): The full image.
            vehicle_bbox (tuple): Bounding box of the detected vehicle (x1, y1, x2, y2).
        Returns:
            list: List of dictionaries with 'text', 'region', and 'confidence'.
        """
        x1_v, y1_v, x2_v, y2_v = vehicle_bbox
        vehicle_crop = image[y1_v:y2_v, x1_v:x2_v]

        if vehicle_crop.size < 1000:  # Skip if vehicle crop is too small
            return []

        license_plates = []

        # Use dedicated license plate model if available
        if self.license_plate_model:
            lp_results = self.license_plate_model(vehicle_crop, verbose=False)
            for lp_result in lp_results:
                lp_boxes = lp_result.boxes
                if lp_boxes is not None:
                    for lp_box in lp_boxes:
                        lp_x1, lp_y1, lp_x2, lp_y2 = lp_box.xyxy[0].cpu().numpy()
                        lp_confidence = float(lp_box.conf[0])

                        # Ensure coordinates are within bounds
                        lp_x1, lp_y1, lp_x2, lp_y2 = map(int, [lp_x1, lp_y1, lp_x2, lp_y2])
                        lp_x1 = max(0, lp_x1)
                        lp_y1 = max(0, lp_y1)
                        lp_x2 = min(vehicle_crop.shape[1], lp_x2)
                        lp_y2 = min(vehicle_crop.shape[0], lp_y2)

                        if lp_x2 > lp_x1 and lp_y2 > lp_y1:
                            plate_region_crop = vehicle_crop[lp_y1:lp_y2, lp_x1:lp_x2]
                            
                            # Enhanced preprocessing for OCR
                            plate_region_crop = self.preprocess_for_ocr(plate_region_crop)
                            text = self.extract_text_from_image(plate_region_crop)
                            
                            if self.is_valid_license_plate(text):
                                license_plates.append({
                                    'text': text,
                                    'region': (lp_x1 + x1_v, lp_y1 + y1_v, lp_x2 + x1_v, lp_y2 + y1_v),
                                    'confidence': lp_confidence
                                })

        # Fallback to region-based detection if no plates found
        if not license_plates:
            license_plates = self.region_based_plate_detection(image, vehicle_bbox)

        return license_plates

    def region_based_plate_detection(self, image, vehicle_bbox):
        """
        Enhanced region-based license plate detection when dedicated model fails.
        Args:
            image (np.array): The full image.
            vehicle_bbox (tuple): Bounding box of the detected vehicle.
        Returns:
            list: Detected license plates with text and confidence.
        """
        x1_v, y1_v, x2_v, y2_v = vehicle_bbox
        vehicle_crop = image[y1_v:y2_v, x1_v:x2_v]
        
        if vehicle_crop.size < 1000:
            return []

        # Convert to grayscale and apply multiple preprocessing techniques
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Filter based on aspect ratio and size (typical for license plates)
            if 2.0 < aspect_ratio < 6.0 and w > 50 and h > 15:
                plate_region = vehicle_crop[y:y+h, x:x+w]
                
                # Further validate by checking edge density
                edges = cv2.Canny(plate_region, 50, 150)
                edge_density = cv2.countNonZero(edges) / float(w * h)
                
                if edge_density > 0.1:  # Plate regions typically have high edge density
                    # Preprocess for better OCR
                    plate_region = self.preprocess_for_ocr(plate_region)
                    text = self.extract_text_from_image(plate_region)
                    
                    if self.is_valid_license_plate(text):
                        plates.append({
                            'text': text,
                            'region': (x + x1_v, y + y1_v, x + w + x1_v, y + h + y1_v),
                            'confidence': min(0.85 + (edge_density * 0.15), 0.95)  # Confidence based on edge density
                        })
        
        return plates

    def preprocess_for_ocr(self, image):
        """
        Enhanced preprocessing for license plate images before OCR.
        Args:
            image (np.array): Input image (BGR or grayscale).
        Returns:
            np.array: Preprocessed image optimized for OCR.
        """
        if len(image.shape) == 3:  # Convert to grayscale if color
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Binarization using adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def extract_text_from_image(self, image):
        """
        Extract text using EasyOCR with optimized parameters.
        Args:
            image (np.array): Preprocessed license plate image.
        Returns:
            str: Extracted and cleaned text.
        """
        if self.ocr_reader is None:
            return ""

        try:
            # Convert to PIL Image if not already
            if not isinstance(image, Image.Image):
                if len(image.shape) == 2:  # Grayscale
                    image = Image.fromarray(image)
                else:  # BGR
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Use detail=0 for faster processing (returns only text and confidence)
            results = self.ocr_reader.readtext(np.array(image), detail=0)
            
            # Filter and clean results
            valid_texts = []
            for text in results:
                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(cleaned) >= 3:  # Minimum length for license plates
                    valid_texts.append(cleaned)
            
            return ' '.join(valid_texts) if valid_texts else ''
        except Exception as e:
            logger.error(f"Error during OCR text extraction: {e}")
            return ''

    def is_valid_license_plate(self, text):
        """
        Robust license plate validation with multiple pattern checks.
        Args:
            text (str): Extracted license plate text.
        Returns:
            bool: True if text matches known license plate patterns.
        """
        if not text or len(text.strip()) < 3:
            return False
            
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Common license plate patterns (customize for your region)
        patterns = [
            r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$',  # Standard pattern (e.g., MH01AB1234)
            r'^[A-Z]{3}[0-9]{4}$',                       # Old pattern (e.g., ABC1234)
            r'^[0-9]{4}[A-Z]{3}$',                       # Some commercial vehicles
            r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$',         # Variants
            r'^[A-Z]{2}[0-9]{3,4}$',                     # Short forms
        ]

        for pattern in patterns:
            if re.fullmatch(pattern, clean_text):
                return True

        # Fallback checks
        has_letter = any(c.isalpha() for c in clean_text)
        has_number = any(c.isdigit() for c in clean_text)
        length_ok = 4 <= len(clean_text) <= 10
        
        return has_letter and has_number and length_ok

    def process_single_image(self, image_path):
        """Process a single image and extract vehicle info"""
        logger.info(f"Processing: {image_path}")

        image, vehicles = self.detect_vehicles(image_path)
        if image is None:
            return None

        image_results = {
            'image_path': image_path,
            'vehicles': []
        }

        unique_plates = set()

        for i, vehicle in enumerate(vehicles):
            license_plates = self.extract_license_plate_region(image, vehicle['bbox'])

            best_plate = None
            if license_plates:
                best_plate = max(
                    license_plates,
                    key=lambda x: (len(x['text'].replace(' ', '')) > 0,
                    x['confidence'] if x['confidence'] > 0 else 0
                ))
                clean_text = best_plate['text'].replace(' ', '')
                if self.is_valid_license_plate(clean_text):
                    unique_plates.add(clean_text)

            vehicle_info = {
                'vehicle_id': i + 1,
                'type': vehicle['type'],
                'confidence': vehicle['confidence'],
                'bbox': vehicle['bbox'],
                'license_plates': [best_plate] if best_plate and self.is_valid_license_plate(best_plate['text']) else []
            }

            image_results['vehicles'].append(vehicle_info)

        # Log results
        logger.info(f"Detection Results for {os.path.basename(image_path)}:")
        logger.info(f"Total vehicles detected: {len(vehicles)}")
        
        if unique_plates:
            logger.info(f"License plates found: {len(unique_plates)}")
            for plate in unique_plates:
                logger.info(f"  → {plate}")
        else:
            logger.info("No license plates detected")

        return image_results

    def process_video(self, video_path, output_video_path=None, frame_skip=5):
        """
        Process video file and extract vehicle info from frames.
        Args:
            video_path (str): Path to input video.
            output_video_path (str): Path to save annotated video (optional).
            frame_skip (int): Process every Nth frame (default: 5).
        Returns:
            dict: Video processing results.
        """
        logger.info(f"Starting video processing for {video_path}")

        if not os.path.exists(video_path):
            logger.error(f"Video file not found at {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file {video_path}")
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video Info: {width}x{height} at {fps} FPS, {total_frames} frames")

        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        video_results = {
            'video_path': video_path,
            'total_frames': total_frames,
            'processed_frames': 0,
            'unique_vehicles': set(),
            'unique_plates': set(),
            'frame_results': []
        }

        frame_count = 0
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % frame_skip == 0:
                    processed_count += 1

                    _, vehicles = self.detect_vehicles(frame)
                    frame_data = {
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'vehicles': []
                    }

                    annotated_frame = frame.copy()

                    for vehicle in vehicles:
                        video_results['unique_vehicles'].add(vehicle['type'])

                        license_plates = self.extract_license_plate_region(frame, vehicle['bbox'])
                        best_plate = None

                        if license_plates:
                            best_plate = max(
                                license_plates,
                                key=lambda x: (len(x['text'].replace(' ', '')) > 0,
                                x['confidence'] if x['confidence'] > 0 else 0
                            ))
                            clean_plate = best_plate['text'].replace(' ', '')
                            if self.is_valid_license_plate(clean_plate):
                                video_results['unique_plates'].add(clean_plate)

                        vehicle_data = {
                            'type': vehicle['type'],
                            'confidence': vehicle['confidence'],
                            'bbox': vehicle['bbox'],
                            'license_plate': best_plate['text'] if best_plate else None
                        }
                        frame_data['vehicles'].append(vehicle_data)

                        # Annotate the frame
                        annotated_frame = self.annotate_frame(annotated_frame, vehicle, best_plate)

                    video_results['frame_results'].append(frame_data)

                    if out is not None:
                        out.write(annotated_frame)

                    if processed_count % 10 == 0:
                        progress = (frame_count / total_frames) * 100
                        logger.info(f"Progress: {progress:.1f}%")

        except Exception as e:
            logger.error(f"Error during video processing: {e}")
        finally:
            cap.release()
            if out is not None:
                out.release()

        video_results['processed_frames'] = processed_count
        self.print_video_summary(video_results)
        return video_results

    def annotate_frame(self, frame, vehicle, plate_info=None):
        """Annotate a frame with vehicle and license plate information."""
        x1, y1, x2, y2 = vehicle['bbox']
        
        # Draw vehicle bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add vehicle type label
        label = f"{vehicle['type']}: {vehicle['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw license plate if detected
        if plate_info and plate_info['text']:
            px1, py1, px2, py2 = plate_info['region']
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(frame, plate_info['text'], (px1, py1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

    def print_video_summary(self, results):
        """Print summary of video processing results."""
        logger.info("\nVideo Processing Complete!")
        logger.info(f"  Frames processed: {results['processed_frames']}")
        logger.info(f"  Unique vehicle types detected: {len(results['unique_vehicles'])}")
        
        for vehicle_type in sorted(results['unique_vehicles']):
            logger.info(f"    → {vehicle_type.capitalize()}")
        
        logger.info(f"  Unique license plates found: {len(results['unique_plates'])}")
        if results['unique_plates']:
            for plate in sorted(results['unique_plates']):
                logger.info(f"    → {plate}")

    def save_results_to_csv(self, results, filename='detection_results.csv'):
        """Save detection results to CSV file."""
        if not results:
            logger.warning("No results to save.")
            return

        rows = []
        for result in results:
            image_path = result.get('image_path', 'unknown')
            for vehicle in result.get('vehicles', []):
                row = {
                    'image_path': image_path,
                    'vehicle_id': vehicle.get('vehicle_id', ''),
                    'vehicle_type': vehicle.get('type', ''),
                    'confidence': vehicle.get('confidence', 0.0),
                    'license_plate': '',
                    'plate_confidence': 0.0
                }
                
                if vehicle.get('license_plates'):
                    plate = vehicle['license_plates'][0]
                    row.update({
                        'license_plate': plate.get('text', ''),
                        'plate_confidence': plate.get('confidence', 0.0)
                    })
                
                rows.append(row)

        try:
            pd.DataFrame(rows).to_csv(filename, index=False)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    # Initialize the system
    detector = TollDetectionSystem(
        vehicle_model_path='yolov8x.pt',
        license_plate_model_path='license_plate.pt'  # Optional
    )

    # Process a single image
    image_result = detector.process_single_image('test_image.jpg')
    if image_result:
        detector.save_results_to_csv([image_result], 'image_results.csv')

    # Process a video file
    video_result = detector.process_video('test_video.mp4', 'annotated_video.mp4')
    if video_result:
        detector.save_results_to_csv(video_result['frame_results'], 'video_results.csv')

if __name__ == "__main__":
    main()