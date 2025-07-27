import cv2  # type: ignore
import numpy as np
from ultralytics import YOLO # type: ignore
import easyocr #type: ignore
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import re
from pathlib import Path
import logging
import time # For simulating delays and periodic checks
import random # For simulating new data selection for replay

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CarDetectionSystem:
    def __init__(self, vehicle_model_path='yolov8l.pt', license_plate_model_path=None):
        """
        Initialize the car detection system with all required models.
        Args:
            vehicle_model_path (str): Path to the YOLO vehicle detection model.
            license_plate_model_path (str): Path to the YOLO license plate detection model.
                                            If None, uses region-based OCR.
        """
        logger.info("Initializing CarDetectionSystem...")

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

        # Initialize OCR reader with better parameters
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
                self.ocr_reader = easyocr.Reader(
                ['en'],
                gpu=False,
                model_storage_directory='models',
                user_network_directory='models'
                )
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
            self.setup_license_plate_model(license_plate_model_path)
        else:
            logger.info("No custom license plate model path provided. Will use region-based OCR detection.")

        # Results storage
        self.results = []

        # Continuous Learning specific attributes
        self.replay_buffer = [] # Stores a subset of past data to prevent catastrophic forgetting
        self.max_replay_buffer_size = 500 # Max number of samples in replay buffer
        self.new_data_queue = [] # Queue for newly added data (e.g., from user feedback)
        self.retraining_threshold = 100 # Number of new data points to trigger retraining
        self.last_retraining_time = time.time()
        self.retraining_interval_hours = 24 # Retrain at least once every 24 hours if new data available

        logger.info("CarDetectionSystem initialization completed")

    def setup_license_plate_model(self, model_path=None):
        """
        Setup license plate detection model.
        Args:
            model_path (str): Path to the license plate model.
        """
        if model_path:
            try:
                self.license_plate_model = YOLO(model_path)
                self.license_plate_model.verbose = False
                logger.info(f"Custom license plate model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load license plate model from {model_path}: {e}")
                self.license_plate_model = None
        else:
            # Fallback to checking default locations if no path is given
            try:
                if os.path.exists('license_plate.pt'):
                    self.license_plate_model = YOLO('license_plate.pt')
                    logger.info("Custom license plate model loaded from 'license_plate.pt'")
                elif os.path.exists('models/license_plate.pt'):
                    self.license_plate_model = YOLO('models/license_plate.pt')
                    logger.info("License plate model loaded from 'models/license_plate.pt'")
                else:
                    logger.warning("License plate model not found in default locations. Using region-based OCR detection.")
                    self.license_plate_model = None
            except Exception as e:
                logger.error(f"Failed to load license plate model from default locations: {e}")
                self.license_plate_model = None


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
        Extracts license plate regions. Uses a dedicated license plate model if available,
        otherwise falls back to enhanced region-based OCR detection.
        Args:
            image (np.array): The full image.
            vehicle_bbox (tuple): Bounding box of the detected vehicle (x1, y1, x2, y2).
        Returns:
            list: List of dictionaries, each with 'text', 'region', and 'confidence'.
        """
        x1_v, y1_v, x2_v, y2_v = vehicle_bbox
        vehicle_crop = image[y1_v:y2_v, x1_v:x2_v]

        if vehicle_crop.size < 1000: # Skip if vehicle crop is too small
            return []

        license_plates = []

        if self.license_plate_model:
            # Use dedicated license plate model
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
                            text = self.extract_text_from_image(plate_region_crop)
                            if self.is_valid_license_plate(text):
                                # Convert plate region coordinates relative to the original image
                                license_plates.append({
                                    'text': text,
                                    'region': (lp_x1 + x1_v, lp_y1 + y1_v, lp_x2 + x1_v, lp_y2 + y1_v),
                                    'confidence': lp_confidence
                                })
        if not license_plates: # Fallback to region-based OCR if no dedicated model or no detections
            # Enhanced preprocessing pipeline for region-based OCR
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            processed_images = []
            processed_images.append(cv2.bilateralFilter(gray, 9, 75, 75))
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            processed_images.append(clahe.apply(gray))
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
            processed_images.append(adaptive_thresh)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            processed_images.append(morph)
            edges = cv2.Canny(gray, 50, 150)
            processed_images.append(edges)

            for processed_img in processed_images:
                contours, _ = cv2.findContours(processed_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    if 2.5 < aspect_ratio < 5.0 and w > 50 and h > 15: # Typical aspect ratio and size
                        plate_region_crop = processed_img[y:y+h, x:x+w]
                        edges = cv2.Canny(plate_region_crop, 50, 150)
                        edge_density = cv2.countNonZero(edges) / float(w * h)

                        if edge_density > 0.1:
                            text = self.extract_text_from_image(plate_region_crop)
                            if self.is_valid_license_plate(text):
                                license_plates.append({
                                    'text': text,
                                    'region': (x + x1_v, y + y1_v, x + w + x1_v, y + h + y1_v),
                                    'confidence': min(0.9 + (edge_density * 0.5), 0.99)
                                })
        if license_plates:
            # Return the plate with the highest confidence or best text length if multiple found
            return [max(license_plates, key=lambda x: x['confidence'] if x['confidence'] > 0 else len(x['text']))]
        return []


    def preprocess_for_ocr(self, gray_image):
        """Apply various preprocessing techniques for better OCR"""
        processed_images = []
        processed_images.append(gray_image) # Original
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(thresh1)
        adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
        processed_images.append(adaptive_thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        processed_images.append(morph)
        return processed_images

    def extract_text_from_image(self, image):
        """Extract text using EasyOCR"""
        try:
            # Convert to PIL Image if not already for EasyOCR compatibility
            if not isinstance(image, Image.Image):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for PIL

            results = self.ocr_reader.readtext(np.array(image))

            text_list = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:
                    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(cleaned_text) >= 3:
                        text_list.append(cleaned_text)
            return ' '.join(text_list) if text_list else ''
        except Exception as e:
            logger.error(f"Error during OCR text extraction: {e}")
            return ''

    def is_valid_license_plate(self, text):
        """More robust license plate validation"""
        if not text or len(text.strip()) < 3:
            return False
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Common Indian license plate patterns
        patterns = [
            r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$',  # Standard pattern (e.g., MH01AB1234)
            r'^[A-Z]{3}[0-9]{4}$',                       # Old pattern (e.g., ABC1234)
            r'^[0-9]{4}[A-Z]{3}$',                       # Some commercial vehicles
            r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$',         # Variants
            r'^[A-Z]{2}[0-9]{3,4}$',                     # Short forms
            r'^[A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{0,2}\s?[0-9]{4}$', # Patterns with spaces
            r'^[A-Z]{2}\s?[0-9]{2}\s?[A-Z]{2}\s?[0-9]{4}$', # Common format with spaces
        ]

        for pattern in patterns:
            if re.fullmatch(pattern, clean_text):
                return True

        has_letter = any(c.isalpha() for c in clean_text)
        has_number = any(c.isdigit() for c in clean_text)
        length_ok = 4 <= len(clean_text) <= 10 # Adjusted length for flexibility

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
                # Prioritize plates with text and then by confidence/length
                best_plate = max(license_plates, key=lambda x: (len(x['text'].replace(' ', '')) > 0, x['confidence'] if x['confidence'] > 0 else 0))
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

        total_vehicles = len(vehicles)
        detected_plates_count = len(unique_plates)

        logger.info(f"Detection Results for {os.path.basename(image_path)}:")
        logger.info(f"Total vehicles detected: {total_vehicles}")

        vehicle_types = {}
        for vehicle in vehicles:
            vtype = vehicle['type']
            vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1

        for vtype, count in vehicle_types.items():
            logger.info(f"  {vtype.capitalize()}: {count}")

        if unique_plates:
            logger.info(f"  License plates found: {detected_plates_count}")
            for plate in unique_plates:
                logger.info(f"      → {plate}")
        else:
            logger.info("  ❌ No license plates detected")

        logger.info("  " + "="*50)

        return image_results

    def process_video(self, video_path, output_video_path=None, frame_skip=5):
        """
        Process video file and extract vehicle info from frames.
        Args:
            video_path (str): Path to input video.
            output_video_path (str): Path to save annotated video (optional).
            frame_skip (int): Process every Nth frame (default: 5 for faster processing).
        Returns:
            dict: Dictionary containing video processing results.
        """
        logger.info(f"Starting video processing for {video_path}")

        if not os.path.exists(video_path):
            logger.error(f"Error: Video file not found at {video_path}")
            logger.info("Checking current directory for video files...")
            current_files = [f for f in os.listdir('.') if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
            if current_files:
                logger.info("Video files found in current directory:")
                for i, file in enumerate(current_files):
                    logger.info(f"  {i+1}. {file}")
            else:
                logger.info("No video files found in current directory.")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error: Could not open video file {video_path} using default backend. Trying alternative approaches.")
            return self.process_video_alternative(video_path, output_video_path, frame_skip)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"🎬 Video Info:")
        logger.info(f"  File: {os.path.basename(video_path)}")
        logger.info(f"  Duration: {duration:.1f} seconds")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Processing every {frame_skip} frames")
        logger.info("  " + "="*50)

        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
            try:
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                logger.info(f"Annotated video will be saved to: {output_video_path}")
            except Exception as e:
                logger.error(f"Failed to initialize video writer: {e}. Output video will not be saved.")
                out = None

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

                    # Process the current frame (numpy array) directly
                    _, vehicles = self.detect_vehicles(frame) # Pass numpy array directly

                    frame_data = {
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'vehicles': []
                    }

                    annotated_frame = frame.copy() # Start with a copy for annotation

                    for vehicle in vehicles:
                        video_results['unique_vehicles'].add(vehicle['type'])

                        license_plates = self.extract_license_plate_region(frame, vehicle['bbox'])
                        best_plate = None

                        if license_plates:
                            best_plate = max(license_plates, key=lambda x: (len(x['text'].replace(' ', '')) > 0, x['confidence'] if x['confidence'] > 0 else 0))
                            clean_plate = best_plate['text'].replace(' ', '')
                            if self.is_valid_license_plate(clean_plate):
                                video_results['unique_plates'].add(clean_plate)

                        vehicle_data = {
                            'type': vehicle['type'],
                            'confidence': vehicle['confidence'],
                            'bbox': vehicle['bbox'],
                            'license_plate': best_plate['text'] if best_plate and self.is_valid_license_plate(best_plate['text']) else None
                        }
                        frame_data['vehicles'].append(vehicle_data)

                        # Annotate the frame
                        annotated_frame = self.annotate_frame(annotated_frame, [vehicle]) # Pass single vehicle for annotation

                    video_results['frame_results'].append(frame_data)

                    if out is not None:
                        out.write(annotated_frame)

                    if processed_count % 10 == 0:
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        logger.info(f"Progress: {progress:.1f}% ({processed_count} frames processed)")

                elif out is not None:
                    # If not processing this frame, just write it to maintain video flow
                    out.write(frame)

        except KeyboardInterrupt:
            logger.info("\nProcessing interrupted by user.")
        except Exception as e:
            logger.error(f"An error occurred during video processing: {e}")

        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            logger.info("Video processing finished or interrupted.")

        video_results['processed_frames'] = processed_count
        self.print_video_summary(video_results)
        return video_results

    def annotate_frame(self, frame, vehicles):
        """Annotate a single frame with detection results"""
        annotated = frame.copy()

        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']

            # Draw vehicle bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add vehicle type label
            label = f"{vehicle['type']}: {vehicle['confidence']:.2f}"
            cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Extract and show license plate if available
            # Note: This part is called inside process_video, so it receives a list of vehicles.
            # However, for consistency with `process_single_image` and `visualize_results`,
            # `extract_license_plate_region` is called on the full frame for each vehicle.
            license_plates = self.extract_license_plate_region(frame, vehicle['bbox']) # Pass full frame here
            if license_plates:
                best_plate = max(license_plates, key=lambda x: (len(x['text'].replace(' ', '')) > 0, x['confidence'] if x['confidence'] > 0 else 0))
                if best_plate['text'] and self.is_valid_license_plate(best_plate['text']):
                    px1, py1, px2, py2 = best_plate['region']
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.putText(annotated, best_plate['text'], (px1, py1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return annotated

    def print_video_summary(self, results):
        """Print summary of video processing results"""
        logger.info("\nVideo Processing Complete!")
        logger.info("  " + "="*50)
        logger.info(f"  Frames processed: {results['processed_frames']}")
        logger.info(f"  Unique vehicle types detected: {len(results['unique_vehicles'])}")

        for vehicle_type in sorted(results['unique_vehicles']):
            logger.info(f"      → {vehicle_type.capitalize()}")

        logger.info(f"  Unique license plates found: {len(results['unique_plates'])}")

        if results['unique_plates']:
            for plate in sorted(results['unique_plates']):
                logger.info(f"      → {plate}")
        else:
            logger.info("      → No license plates detected")

        logger.info("  " + "="*50)

    def save_video_results_to_csv(self, video_results, filename='video_detection_results.csv'):
        """Save video processing results to CSV"""
        if not video_results or not video_results['frame_results']:
            logger.warning("No video results to save.")
            return

        rows = []
        for frame_data in video_results['frame_results']:
            frame_num = frame_data['frame_number']
            timestamp = frame_data['timestamp']

            if frame_data['vehicles']:
                for i, vehicle in enumerate(frame_data['vehicles']):
                    rows.append({
                        'video_path': video_results['video_path'],
                        'frame_number': frame_num,
                        'timestamp_seconds': timestamp,
                        'vehicle_id': i + 1,
                        'vehicle_type': vehicle['type'],
                        'confidence': vehicle['confidence'],
                        'license_plate': vehicle['license_plate'] or 'Not detected'
                    })
            else:
                rows.append({
                    'video_path': video_results['video_path'],
                    'frame_number': frame_num,
                    'timestamp_seconds': timestamp,
                    'vehicle_id': 0, # Indicate no specific vehicle
                    'vehicle_type': 'None',
                    'confidence': 0.0,
                    'license_plate': 'No vehicles'
                })

        df = pd.DataFrame(rows)
        try:
            df.to_csv(filename, index=False)
            logger.info(f"Video results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save video results to CSV: {e}")

    def process_video_alternative(self, video_path, output_video_path=None, frame_skip=5):
        """
        Alternative video processing using ffmpeg to extract frames.
        This method is a fallback if direct OpenCV video capture fails.
        """
        try:
            import subprocess
            import tempfile
            import shutil

            logger.info("Using alternative method: extracting frames with ffmpeg...")

            temp_dir = tempfile.mkdtemp(prefix="video_frames_") # Unique temp directory

            try:
                # Use ffmpeg to extract frames
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vf', f'select=not(mod(n\\,{frame_skip}))',
                    '-vsync', 'vfr', # Variable frame rate to prevent duplicates
                    '-q:v', '2', # Quality for output JPEG (2 is good, 1 is best)
                    f'{temp_dir}/frame_%08d.jpg' # Use %08d for more frames
                ]
                logger.debug(f"FFmpeg command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)

                if result.returncode != 0:
                    logger.error(f"FFmpeg frame extraction failed. Error: {result.stderr}")
                    return self.process_video_frame_by_frame(video_path, frame_skip)

                frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.jpg')])

                if not frame_files:
                    logger.warning("No frames extracted by ffmpeg.")
                    return None

                logger.info(f"Extracted {len(frame_files)} frames using ffmpeg.")

                # Estimate total frames if original cap failed, for progress calculation
                estimated_total_frames = len(frame_files) * frame_skip # Simple estimate

                video_results = {
                    'video_path': video_path,
                    'total_frames': estimated_total_frames, # Update based on estimate
                    'processed_frames': len(frame_files),
                    'unique_vehicles': set(),
                    'unique_plates': set(),
                    'frame_results': []
                }

                # If an output video is requested, we need to re-encode from processed frames
                # This can be resource intensive and might be better handled by a separate function
                # or by directly writing to video in the main loop if OpenCV capture works.
                # For this alternative path, we'll assume output_video_path is handled externally
                # or that annotation is just for visualization during processing.
                # If you truly need an annotated video from ffmpeg-extracted frames,
                # you'd need another ffmpeg pass or stitch frames with OpenCV after processing.
                if output_video_path:
                    logger.warning("Output video saving is not directly supported in this ffmpeg alternative path for efficiency. You might need to re-encode frames manually.")

                for i, frame_file in enumerate(frame_files):
                    frame_path = os.path.join(temp_dir, frame_file)

                    # Process frame
                    frame_original = cv2.imread(frame_path)
                    if frame_original is None:
                        logger.warning(f"Could not read extracted frame: {frame_path}. Skipping.")
                        continue

                    _, vehicles = self.detect_vehicles(frame_original)

                    frame_data = {
                        'frame_number': (i + 1) * frame_skip, # Approximate frame number
                        'timestamp': ((i + 1) * frame_skip) / 30.0, # Approximate timestamp (assuming 30fps)
                        'vehicles': []
                    }

                    # Annotate and add to results
                    annotated_frame = frame_original.copy()
                    for vehicle in vehicles:
                        video_results['unique_vehicles'].add(vehicle['type'])
                        license_plates = self.extract_license_plate_region(frame_original, vehicle['bbox'])
                        best_plate = None

                        if license_plates:
                            best_plate = max(license_plates, key=lambda x: (len(x['text'].replace(' ', '')) > 0, x['confidence'] if x['confidence'] > 0 else 0))
                            clean_plate = best_plate['text'].replace(' ', '')
                            if self.is_valid_license_plate(clean_plate):
                                video_results['unique_plates'].add(clean_plate)

                        vehicle_data = {
                            'type': vehicle['type'],
                            'confidence': vehicle['confidence'],
                            'bbox': vehicle['bbox'],
                            'license_plate': best_plate['text'] if best_plate and self.is_valid_license_plate(best_plate['text']) else None
                        }
                        frame_data['vehicles'].append(vehicle_data)
                        annotated_frame = self.annotate_frame(annotated_frame, [vehicle]) # Annotate this specific vehicle

                    video_results['frame_results'].append(frame_data)

                    if (i + 1) % 10 == 0:
                        progress = ((i + 1) / len(frame_files)) * 100
                        logger.info(f"  Progress (ffmpeg frames): {progress:.1f}%")

            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")

            self.print_video_summary(video_results)
            return video_results

        except ImportError:
            logger.error("`subprocess` module not available. Cannot use ffmpeg alternative. Trying manual frame extraction.")
            return self.process_video_frame_by_frame(video_path, frame_skip)
        except Exception as e:
            logger.error(f"Alternative (ffmpeg) method failed unexpectedly: {e}. Trying manual frame extraction.")
            return self.process_video_frame_by_frame(video_path, frame_skip)

    def process_video_frame_by_frame(self, video_path, frame_skip=5):
        """
        Last resort: try different OpenCV approaches for frame-by-frame processing.
        This is typically slower but more robust to various video formats/codecs.
        """
        logger.warning("Trying frame-by-frame processing with different OpenCV backends as a last resort...")
        approaches = [
            lambda path: cv2.VideoCapture(path, cv2.CAP_FFMPEG),
            lambda path: cv2.VideoCapture(path, cv2.CAP_ANY),
            lambda path: cv2.VideoCapture(path, cv2.CAP_GSTREAMER), # Might require GStreamer installed
        ]

        for i, approach_func in enumerate(approaches):
            try:
                cap = approach_func(video_path)
                if cap.isOpened():
                    logger.info(f"Success with OpenCV backend approach {i+1}.")

                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1

                        if frame_count % frame_skip == 0:
                            processed_count += 1

                            # Process frame (numpy array)
                            _, vehicles = self.detect_vehicles(frame)

                            frame_data = {
                                'frame_number': frame_count,
                                'timestamp': frame_count / fps,
                                'vehicles': []
                            }

                            for vehicle in vehicles:
                                video_results['unique_vehicles'].add(vehicle['type'])

                                license_plates = self.extract_license_plate_region(frame, vehicle['bbox'])
                                best_plate = None

                                if license_plates:
                                    best_plate = max(license_plates, key=lambda x: (len(x['text'].replace(' ', '')) > 0, x['confidence'] if x['confidence'] > 0 else 0))
                                    clean_plate = best_plate['text'].replace(' ', '')
                                    if self.is_valid_license_plate(clean_plate):
                                        video_results['unique_plates'].add(clean_plate)

                                vehicle_data = {
                                    'type': vehicle['type'],
                                    'confidence': vehicle['confidence'],
                                    'bbox': vehicle['bbox'],
                                    'license_plate': best_plate['text'] if best_plate and self.is_valid_license_plate(best_plate['text']) else None
                                }
                                frame_data['vehicles'].append(vehicle_data)

                            video_results['frame_results'].append(frame_data)

                            if processed_count % 5 == 0:
                                progress = (frame_count / max(total_frames, frame_count)) * 100 if total_frames > 0 else 0
                                logger.info(f"Progress (frame-by-frame): {progress:.1f}%")

                            # Limit processing for robustness in this fallback
                            if processed_count > 100 and total_frames > 200: # Process max 100 frames if video is long
                                logger.warning("Limiting frame-by-frame processing to 100 frames for safety due to potential performance issues.")
                                break

                    cap.release()
                    video_results['processed_frames'] = processed_count
                    self.print_video_summary(video_results)
                    return video_results

                else:
                    cap.release()

            except Exception as e:
                logger.error(f"OpenCV backend approach {i+1} failed: {e}")
                continue

        logger.error("❌ All video processing approaches failed. Cannot process video.")
        logger.info("💡 Suggestions:")
        logger.info("   1. Check if the video file is corrupted.")
        logger.info("   2. Ensure necessary video codecs and libraries (like FFmpeg, GStreamer) are installed.")
        logger.info("   3. Try converting the video to a common format like MP4 (H.264).")
        logger.info("   4. Reduce video size/resolution or use a different video file.")
        return None

    def process_dataset(self, dataset_path):
        """Process entire dataset of images"""
        dataset_path = Path(dataset_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f'*{ext}'))
            image_files.extend(dataset_path.glob(f'*{ext.upper()}'))

        logger.info(f"Found {len(image_files)} images to process in dataset: {dataset_path}")

        all_results = []
        for image_file in image_files:
            result = self.process_single_image(str(image_file))
            if result:
                all_results.append(result)
                # Extend self.results with detailed vehicle info including image_path
                for vehicle in result['vehicles']:
                    vehicle['image_path'] = result['image_path'] # Add image_path to each vehicle result
                    self.results.append(vehicle)
        return all_results

    def save_results_to_csv(self, filename='detection_results.csv'):
        """Save image processing results to CSV file"""
        if not self.results:
            logger.warning("No image detection results to save.")
            return

        rows = []
        for result in self.results:
            # Ensure all keys are present, providing defaults if missing
            image_path = result.get('image_path', 'unknown')
            vehicle_id = result.get('vehicle_id', 'unknown')
            vehicle_type = result.get('type', 'unknown')
            confidence = result.get('confidence', 0.0)
            bbox = result.get('bbox', (0, 0, 0, 0))

            license_plates = result.get('license_plates', [])

            if license_plates:
                for plate in license_plates:
                    rows.append({
                        'image_path': image_path,
                        'vehicle_id': vehicle_id,
                        'vehicle_type': vehicle_type,
                        'vehicle_confidence': confidence,
                        'vehicle_bbox': str(bbox),
                        'license_plate': plate.get('text', 'Not detected'),
                        'plate_confidence': plate.get('confidence', 0.0)
                    })
            else:
                rows.append({
                    'image_path': image_path,
                    'vehicle_id': vehicle_id,
                    'vehicle_type': vehicle_type,
                    'vehicle_confidence': confidence,
                    'vehicle_bbox': str(bbox),
                    'license_plate': 'Not detected',
                    'plate_confidence': 0.0
                })

        df = pd.DataFrame(rows)
        try:
            df.to_csv(filename, index=False)
            logger.info(f"Image detection results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save image results to CSV: {e}")

    def visualize_results(self, image_path, save_path=None):
        """
        Visualize detection results on image and save it.
        Args:
            image_path (str): Path to the input image.
            save_path (str): Path to save the annotated image. If None, saves to 'outputs/'.
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image for visualization: {image_path}")
            return

        file_id = os.path.splitext(os.path.basename(image_path))[0]

        # Detect vehicles
        _, vehicles = self.detect_vehicles(image_path) # Pass image_path for detection

        # Draw bounding boxes
        annotated_image = image.copy() # Work on a copy to draw
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{vehicle['type']}: {vehicle['confidence']:.2f}"
            cv2.putText(annotated_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Extract and show license plates
            license_plates = self.extract_license_plate_region(image, vehicle['bbox']) # Pass original image here
            if license_plates:
                best_plate = max(license_plates, key=lambda x: (len(x['text'].replace(' ', '')) > 0, x['confidence'] if x['confidence'] > 0 else 0))
                if best_plate['text'] and self.is_valid_license_plate(best_plate['text']):
                    px1, py1, px2, py2 = best_plate['region']
                    cv2.rectangle(annotated_image, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.putText(annotated_image, best_plate['text'], (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Ensure outputs directory exists
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_path:
            final_save_path = save_path
        else:
            final_save_path = output_dir / f"{file_id}_processed.jpg"

        try:
            cv2.imwrite(str(final_save_path), annotated_image)
            logger.info(f"Annotated image saved to {final_save_path}")
        except Exception as e:
            logger.error(f"Failed to save annotated image: {e}")

        # Display using matplotlib (optional, can be commented out for server environments)
        # image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        # plt.figure(figsize=(12, 8))
        # plt.imshow(image_rgb)
        # plt.axis('off')
        # plt.title(f"Processed Image: {os.path.basename(image_path)}")
        # plt.show() # Use plt.show() if you want to display, otherwise remove for non-interactive
        # plt.close()


    # --- New Methods for Model Training ---
    def train_model(self, data_config_path, epochs=50, batch_size=16, img_size=640, model_type='vehicle', save_dir='runs/train'):
        """
        Trains the YOLOv8 model (vehicle or license plate) on a custom dataset.
        Args:
            data_config_path (str): Path to the YAML data configuration file (e.g., 'data.yaml').
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            img_size (int): Image size for training (e.g., 640 for 640x640).
            model_type (str): 'vehicle' to train the vehicle model, 'license_plate' for license plate model.
            save_dir (str): Directory to save training results (weights, logs, etc.).
        Returns:
            ultralytics.yolo.engine.model.YOLO: The trained YOLO model.
        """
        logger.info(f"Starting training for {model_type} model...")
        logger.info(f"Data config: {data_config_path}, Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")

        if not os.path.exists(data_config_path):
            logger.error(f"Data configuration file not found at: {data_config_path}")
            return None

        model_to_train = None
        current_model_path = None # To save the current model path if it exists

        if model_type == 'vehicle':
            if self.vehicle_model and hasattr(self.vehicle_model, 'predictor') and hasattr(self.vehicle_model.predictor, 'model'):
                # Try to get the path of the currently loaded model to warm-start training
                # This is a bit tricky with YOLOv8 if it's loaded from .pt or .yaml dynamically.
                # A robust way is to always save the 'best.pt' and load from there.
                logger.info("Using current vehicle model as a starting point for training.")
                # If you want to continue training directly on the loaded model, pass its .pt path
                model_to_train = self.vehicle_model # Directly use the loaded YOLO object
            else:
                model_to_train = YOLO('yolov8x.pt') # Load a base pre-trained model
                logger.warning("Vehicle model not initialized or not a proper YOLO object. Loading yolov8x.pt for training.")
        elif model_type == 'license_plate':
            if self.license_plate_model and hasattr(self.license_plate_model, 'predictor') and hasattr(self.license_plate_model.predictor, 'model'):
                logger.info("Using current license plate model as a starting point for training.")
                model_to_train = self.license_plate_model
            else:
                model_to_train = YOLO('yolov8n.pt') # Start with a smaller base for LP
                logger.warning("License plate model not initialized or not a proper YOLO object. Loading yolov8n.pt for training.")
        else:
            logger.error(f"Invalid model_type: {model_type}. Must be 'vehicle' or 'license_plate'.")
            return None

        if model_to_train is None:
            logger.error("No model to train. Exiting training process.")
            return None

        try:
            # Train the model
            # For continuous learning, we want to fine-tune. `model_to_train` already points
            # to the loaded model (either initial or previously trained).
            results = model_to_train.train(
                data=data_config_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                name=f'{model_type}_training_continual', # Name for the run directory, different from initial
                project=save_dir,
                device=0,  # Use GPU 0 if available, or 'cpu'
                # Add more training arguments as needed, e.g., learning rate, optimizer
                # lr0=0.001, # Potentially lower learning rate for fine-tuning
                # optimizer='AdamW',
                # augment=True, # Enable data augmentation
                # cache=True, # Cache images for faster training
                # patience=10, # Shorter patience for incremental updates
            )
            logger.info(f"Training for {model_type} model completed successfully.")

            # After training, load the best model weights
            # The `results` object often contains the path to the best model.
            best_model_path_from_results = Path(results.save_dir) / "weights" / "best.pt"

            if best_model_path_from_results.exists():
                trained_model = YOLO(str(best_model_path_from_results))
                logger.info(f"Loaded best trained model from: {best_model_path_from_results}")
                # Update the system's models
                if model_type == 'vehicle':
                    self.vehicle_model = trained_model
                elif model_type == 'license_plate':
                    self.license_plate_model = trained_model
                return trained_model
            else:
                logger.warning(f"Best model weights not found at {best_model_path_from_results}. Returning the model object that completed training (last.pt).")
                return model_to_train # Return the model object that completed training

        except Exception as e:
            logger.error(f"An error occurred during {model_type} model training: {e}", exc_info=True)
            return None

    def validate_model(self, model_path, data_config_path, img_size=640, model_type='vehicle'):
        """
        Validates a trained YOLOv8 model.
        Args:
            model_path (str): Path to the trained model weights (e.g., 'runs/train/exp/weights/best.pt').
            data_config_path (str): Path to the YAML data configuration file for validation.
            img_size (int): Image size for validation.
            model_type (str): 'vehicle' or 'license_plate' for logging purposes.
        Returns:
            dict: Validation metrics.
        """
        logger.info(f"Starting validation for {model_type} model from {model_path}...")
        try:
            model = YOLO(model_path)
            model.verbose = False # Suppress excessive output
            metrics = model.val(data=data_config_path, imgsz=img_size, split='val')
            logger.info(f"Validation for {model_type} model completed.")
            logger.info(f"Validation Metrics: {metrics.results_dict}")
            return metrics.results_dict
        except Exception as e:
            logger.error(f"An error occurred during {model_type} model validation: {e}", exc_info=True)
            return None

    def export_model(self, model_path, format='onnx'):
        """
        Exports a YOLOv8 model to a different format (e.g., ONNX, OpenVINO, TFLite).
        Args:
            model_path (str): Path to the trained model weights.
            format (str): The format to export to (e.g., 'onnx', 'openvino', 'tflite').
        Returns:
            str: Path to the exported model.
        """
        logger.info(f"Exporting model {model_path} to {format} format...")
        try:
            model = YOLO(model_path)
            exported_path = model.export(format=format)
            logger.info(f"Model exported successfully to {exported_path}")
            return str(exported_path)
        except Exception as e:
            logger.error(f"Failed to export model {model_path}: {e}", exc_info=True)
            return None

    # --- New Methods for Continuous Self-Improvement ---

    def add_new_data_for_learning(self, image_path, label_path):
        """
        Adds a new data point (image and its corresponding YOLO label file)
        to an internal queue for future retraining.
        This simulates user adding data.
        Args:
            image_path (str): Path to the new image.
            label_path (str): Path to the YOLO format label file (.txt) for the image.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found at {image_path}. Cannot add to new data queue.")
            return
        if not os.path.exists(label_path):
            logger.error(f"Label file not found at {label_path}. Cannot add to new data queue.")
            return

        self.new_data_queue.append({'image': image_path, 'label': label_path})
        logger.info(f"Added new data point: Image '{os.path.basename(image_path)}', Label '{os.path.basename(label_path)}'. New data queue size: {len(self.new_data_queue)}")

    def _prepare_retraining_dataset(self, temp_data_dir='temp_retrain_data'):
        """
        Prepares a temporary dataset for retraining by combining new data and replay buffer.
        Returns:
            str: Path to the data.yaml file for the temporary dataset.
        """
        temp_data_dir_path = Path(temp_data_dir)
        temp_images_dir = temp_data_dir_path / 'images'
        temp_labels_dir = temp_data_dir_path / 'labels'
        temp_data_dir_path.mkdir(parents=True, exist_ok=True)
        temp_images_dir.mkdir(exist_ok=True)
        temp_labels_dir.mkdir(exist_ok=True)

        logger.info(f"Preparing temporary dataset for retraining in {temp_data_dir_path}")

        # Clear previous temporary data if it exists
        for item in temp_images_dir.iterdir():
            if item.is_file(): item.unlink()
        for item in temp_labels_dir.iterdir():
            if item.is_file(): item.unlink()

        # Combine new data from queue and selected data from replay buffer
        training_samples = list(self.new_data_queue) + random.sample(self.replay_buffer, min(len(self.replay_buffer), int(self.max_replay_buffer_size * 0.5))) # Use up to 50% of buffer

        if not training_samples:
            logger.warning("No data available for retraining (new data + replay buffer empty).")
            return None

        # Copy data to temporary directory and create data.yaml
        train_image_paths = []
        for i, sample in enumerate(training_samples):
            img_src = Path(sample['image'])
            label_src = Path(sample['label'])

            # Generate unique names to avoid conflicts (important if files have same names but different paths)
            img_dest_name = f"img_{i}_{img_src.name}"
            label_dest_name = f"label_{i}_{label_src.name}"

            img_dest = temp_images_dir / img_dest_name
            label_dest = temp_labels_dir / label_dest_name

            try:
                # Use shutil.copy to preserve permissions/metadata better than os.rename for this use case
                import shutil
                shutil.copy(img_src, img_dest)
                shutil.copy(label_src, label_dest)
                train_image_paths.append(str(img_dest)) # Add to list for data.yaml
            except Exception as e:
                logger.error(f"Failed to copy data for retraining: {e}")
                continue

        # Create data.yaml for YOLO training
        data_yaml_path = temp_data_dir_path / 'data.yaml'
        # Assuming your vehicle classes are defined, you'll need to know them here
        # For simplicity, we'll use a generic classes placeholder.
        # In a real scenario, you'd extract classes from your original data.yaml or define them.
        class_names = list(self.vehicle_classes.values()) # Assuming vehicle model classes are what we train for
        if self.license_plate_model: # If training LP model, ensure LP class is there
             class_names.append('license_plate') # Example, adjust based on your LP dataset
        class_names = sorted(list(set(class_names))) # Remove duplicates and sort

        data_yaml_content = f"""
path: {temp_data_dir_path.resolve()}
train: images
val: images # For simplicity, using same for train/val. In production, use a separate small val set.
nc: {len(class_names)}
names: {class_names}
"""
        try:
            with open(data_yaml_path, 'w') as f:
                f.write(data_yaml_content)
            logger.info(f"Temporary data.yaml created at {data_yaml_path}")
            return str(data_yaml_path)
        except Exception as e:
            logger.error(f"Failed to create data.yaml: {e}")
            return None

    def _update_replay_buffer(self):
        """
        Updates the replay buffer with a subset of the new data and maintains its size.
        This helps prevent catastrophic forgetting.
        """
        # Add new data to replay buffer
        for sample in self.new_data_queue:
            if len(self.replay_buffer) >= self.max_replay_buffer_size:
                # Remove oldest or random element if buffer is full (e.g., reservoir sampling)
                self.replay_buffer.pop(0) # Remove oldest
            self.replay_buffer.append(sample)
        logger.info(f"Replay buffer updated. Current size: {len(self.replay_buffer)}")

    def initiate_continuous_learning(self, vehicle_data_config_template, lp_data_config_template=None,
                                     training_epochs=5, training_batch_size=8, check_interval_seconds=3600):
        """
        Initiates a continuous learning loop.
        This function would typically run as a background service or a scheduled job.
        It simulates:
        1. Checking for new data.
        2. Triggering retraining based on data volume or time.
        3. Updating the model.
        4. (Optional) Monitoring model performance.

        Args:
            vehicle_data_config_template (str): Path to a *template* data.yaml for vehicle model.
                                                Used to get class names for _prepare_retraining_dataset.
            lp_data_config_template (str): Path to a *template* data.yaml for LP model (optional).
            training_epochs (int): Number of epochs for incremental training. Keep low for continuous.
            training_batch_size (int): Batch size for incremental training.
            check_interval_seconds (int): How often to check for new data/trigger retraining (e.g., 3600s = 1 hour).
        """
        logger.info("Initiating continuous learning loop...")
        logger.info(f"Retraining will be triggered with at least {self.retraining_threshold} new data points OR every {self.retraining_interval_hours} hours.")

        while True:
            current_time = time.time()
            elapsed_time_hours = (current_time - self.last_retraining_time) / 3600

            should_retrain = False
            if len(self.new_data_queue) >= self.retraining_threshold:
                logger.info(f"Retraining triggered: {len(self.new_data_queue)} new data points accumulated (threshold {self.retraining_threshold}).")
                should_retrain = True
            elif elapsed_time_hours >= self.retraining_interval_hours and self.new_data_queue:
                logger.info(f"Retraining triggered: Time interval ({self.retraining_interval_hours}h) elapsed and new data available.")
                should_retrain = True
            else:
                logger.info(f"No retraining trigger met. New data queue: {len(self.new_data_queue)}, Elapsed time: {elapsed_time_hours:.2f} hours.")

            if should_retrain:
                logger.info("Starting incremental retraining process...")

                # 1. Prepare combined dataset (new data + replay buffer)
                # We assume the new data added via `add_new_data_for_learning` is suitable for the
                # *vehicle* model for now. If you have separate LP data, you'd need two queues/logic.
                combined_data_config_path = self._prepare_retraining_dataset(temp_data_dir='temp_vehicle_retrain_data')
                if combined_data_config_path:
                    try:
                        # 2. Train the vehicle model incrementally
                        logger.info("Training vehicle model with new data and replay buffer.")
                        trained_vehicle_model = self.train_model(
                            data_config_path=combined_data_config_path,
                            epochs=training_epochs,
                            batch_size=training_batch_size,
                            img_size=640, # Keep consistent with initial training or adapt
                            model_type='vehicle',
                            save_dir='runs/continuous_train_vehicles'
                        )
                        if trained_vehicle_model:
                            logger.info("Vehicle model successfully updated through continuous learning.")
                            # Optional: Validate the updated model on a fresh validation set
                            # self.validate_model(str(trained_vehicle_model.pt), vehicle_data_config_template, model_type='vehicle')

                        # If you have a separate LP model and dedicated new LP data, repeat for LP model:
                        # lp_combined_data_config_path = self._prepare_retraining_dataset(temp_data_dir='temp_lp_retrain_data', for_lp=True)
                        # if lp_combined_data_config_path:
                        #     trained_license_plate_model = self.train_model(
                        #         data_config_path=lp_combined_data_config_path,
                        #         epochs=training_epochs, # Can be different
                        #         batch_size=training_batch_size, # Can be different
                        #         img_size=320, # Can be different
                        #         model_type='license_plate',
                        #         save_dir='runs/continuous_train_license_plates'
                        #     )
                        #     if trained_license_plate_model:
                        #         logger.info("License plate model successfully updated through continuous learning.")

                    except Exception as e:
                        logger.error(f"Error during continuous learning retraining: {e}", exc_info=True)
                    finally:
                        # Clean up temporary dataset after training
                        if Path(combined_data_config_path).parent.exists():
                            import shutil
                            shutil.rmtree(Path(combined_data_config_path).parent)
                            logger.info(f"Cleaned up temporary retraining directory: {Path(combined_data_config_path).parent}")
                else:
                    logger.warning("No data for retraining after preparing dataset. Skipping current retraining cycle.")

                # 3. Update replay buffer and clear new data queue
                self._update_replay_buffer()
                self.new_data_queue = [] # Clear the queue after successful (attempted) retraining
                self.last_retraining_time = current_time # Reset timer
                logger.info("Continuous learning cycle completed.")

            logger.info(f"Sleeping for {check_interval_seconds} seconds before next check...")
            time.sleep(check_interval_seconds)

    # --- Utility for continuous learning (mock user feedback or data stream) ---
    def simulate_new_data_arrival(self, num_samples=5, base_data_dir='path/to/your/sample_annotated_data'):
        """
        Simulates new annotated data arriving, which would typically come from:
        - User corrections/annotations.
        - High-confidence pseudo-labeled data (after human review).
        - New data captured from the system and manually annotated.

        For demonstration, this function picks random annotated files from a specified directory.
        YOU MUST REPLACE 'path/to/your/sample_annotated_data' with a real path
        containing images and corresponding YOLO .txt label files.
        """
        if not os.path.exists(base_data_dir):
            logger.warning(f"Base data directory for simulation not found: {base_data_dir}. Cannot simulate new data arrival.")
            logger.info("Please create a directory with images and their corresponding YOLO label files (.txt) and update `base_data_dir`.")
            return

        image_files = []
        label_files = {} # Store labels mapping to images

        for root, _, files in os.walk(base_data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(root, file)
                    label_path = os.path.join(root, Path(file).stem + '.txt')
                    if os.path.exists(label_path):
                        image_files.append(img_path)
                        label_files[img_path] = label_path
                    else:
                        logger.debug(f"No matching label found for {img_path}")

        if not image_files:
            logger.warning(f"No image-label pairs found in {base_data_dir} for simulation.")
            return

        logger.info(f"Simulating arrival of {num_samples} new data points from {base_data_dir}...")
        selected_samples = random.sample(image_files, min(num_samples, len(image_files)))

        for img_path in selected_samples:
            self.add_new_data_for_learning(img_path, label_files[img_path])
        logger.info(f"Simulated {len(selected_samples)} new data points.")


def main():
    # Example Usage:
    # 1. Initialize the system
    detector = CarDetectionSystem(
        vehicle_model_path='yolov8x.pt', # Start with a powerful pre-trained vehicle model
        license_plate_model_path=None # Can be 'license_plate.pt' if you have a custom one
    )


# Additional utility functions (from your original code, kept for completeness)
def check_video_file(video_path):
    """Check if video file exists and get info"""
    if not os.path.exists(video_path):
        logger.error(f"❌ File not found: {video_path}")
        current_dir = os.getcwd()
        video_files = [f for f in os.listdir(current_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'))]
        if video_files:
            logger.info("Video files in current directory:")
            for i, file in enumerate(video_files, 1):
                logger.info(f"  {i}. {file}")
            return video_files # Return list of files if found
        else:
            logger.info("No video files found in current directory.")
            return [] # Return empty list if no files found

    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            logger.info(f"✅ Video file OK: {os.path.basename(video_path)}")
            logger.info(f"  Resolution: {int(width)}x{int(height)}")
            logger.info(f"  FPS: {fps}")
            logger.info(f"  Frames: {int(frame_count)}")
            cap.release()
            return True # Return True for a valid single file
        else:
            logger.error(f"❌ Cannot open video: {video_path}")
            cap.release()
            return False
    except Exception as e:
        logger.error(f"❌ Error checking video: {e}")
        return False

def process_video_simple(video_path):
    """Simplified video processing function with better error handling"""
    logger.info(f"Starting simplified video processing for {video_path}...")

    if not check_video_file(video_path):
        return None

    detector = CarDetectionSystem()
    results = detector.process_video(video_path, frame_skip=10)

    if results:
        detector.save_video_results_to_csv(results, 'video_results.csv')
        logger.info("Video processing completed successfully!")
        return results
    else:
        logger.error("Video processing failed.")
        return None

def process_video_only(video_path, save_annotated=False):
    """Quick function to process only a video file"""
    logger.info(f"Quick processing video: {video_path}")
    detector = CarDetectionSystem()

    output_path = None
    if save_annotated:
        name, ext = os.path.splitext(video_path)
        output_path = f"{name}_annotated{ext}"
        logger.info(f"Annotated video will be saved to: {output_path}")

    # Use a faster frame_skip for quick processing (e.g., every 3rd frame)
    results = detector.process_video(video_path, output_path, frame_skip=3)

    if results:
        detector.save_video_results_to_csv(results, 'video_results.csv')
        logger.info("Video processing completed.")
        return results
    else:
        logger.error("Video processing failed.")
        return None

def batch_process_images(image_folder, output_folder):
    """Batch process images and save annotated results"""
    logger.info(f"Starting batch processing for images in {image_folder}")
    detector = CarDetectionSystem()

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    processed_count = 0
    total_files = 0
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                total_files += 1

    if total_files == 0:
        logger.warning(f"No image files found in {image_folder}.")
        return

    for root, _, files in os.walk(image_folder):
        for image_file in files:
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, image_file)
                relative_path = os.path.relpath(image_path, image_folder)
                output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))
                Path(output_subfolder).mkdir(parents=True, exist_ok=True)
                output_annotated_path = os.path.join(output_subfolder, f"annotated_{image_file}")

                try:
                    detector.visualize_results(image_path, output_annotated_path)
                    processed_count += 1
                    if processed_count % 10 == 0 or processed_count == total_files:
                        logger.info(f"Processed {processed_count}/{total_files} images ({processed_count/total_files*100:.1f}%)")
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
    logger.info(f"Batch processing completed. Annotated images saved to {output_folder}")


def analyze_results(csv_file):
    """Analyze the detection results from a CSV file"""
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return

    try:
        df = pd.read_csv(csv_file)
        logger.info("=== Detection Results Analysis ===")
        logger.info(f"Total entries: {len(df)}")

        if 'license_plate' in df.columns:
            vehicles_with_plates = df[df['license_plate'] != 'Not detected']
            logger.info(f"Entries with license plates detected: {len(vehicles_with_plates)}")
            if len(df) > 0:
                detection_rate = len(vehicles_with_plates) / len(df) * 100
                logger.info(f"Detection rate: {detection_rate:.1f}%")
        else:
            logger.warning(" 'license_plate' column not found in CSV. Cannot analyze plate detection rate.")

        if 'vehicle_type' in df.columns:
            logger.info("\nVehicle types distribution:")
            logger.info(df['vehicle_type'].value_counts().to_string()) # Use to_string for better logging
        else:
            logger.warning(" 'vehicle_type' column not found in CSV. Cannot analyze vehicle types.")

        if 'license_plate' in df.columns and len(vehicles_with_plates) > 0:
            logger.info("\nSample unique license plates detected:")
            # Use unique() to avoid duplicates in the sample
            valid_plates = vehicles_with_plates['license_plate'].unique()
            for i, plate in enumerate(valid_plates[:20]):  # Show first 20 unique plates
                logger.info(f"  - {plate}")
            if len(valid_plates) > 20:
                logger.info(f"  ...and {len(valid_plates) - 20} more unique plates.")
        elif 'license_plate' in df.columns:
            logger.info("\nNo license plates were detected in the CSV data.")

        logger.info("===============================")
    except Exception as e:
        logger.error(f"Error analyzing CSV file {csv_file}: {e}")

# Run the main function
if __name__ == "__main__":
    main()