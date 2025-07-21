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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarDetectionSystem:
    def __init__(self):
        """Initialize the car detection system with all required models"""
        logger.info("Initializing CarDetectionSystem...")

        # Initialize YOLOv8 models
        try:
            self.vehicle_model = YOLO('yolov8m.pt')  # Pre-trained model for vehicle detection
            self.vehicle_model.verbose = False
            logger.info(" Vehicle detection model loaded successfully")
        except Exception as e:
            logger.error(f" Failed to load vehicle model: {e}")
            self.vehicle_model = None

        self.license_plate_model = None

        # Initialize OCR reader
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if no GPU
            logger.info(" OCR reader initialized successfully")
        except Exception as e:
            logger.warning(f" Failed to initialize OCR with GPU, trying CPU: {e}")
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info(" OCR reader initialized with CPU")
            except Exception as e2:
                logger.error(f" Failed to initialize OCR: {e2}")
                self.ocr_reader = None

        # Vehicle class mapping for YOLO COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        # Results storage
        self.results = []

        logger.info("CarDetectionSystem initialization completed")

    def setup_license_plate_model(self):
        """
        Setup license plate detection model
        """
        try:
            # Try to load a custom license plate model if available
            if os.path.exists('license_plate.pt'):
                self.license_plate_model = YOLO('license_plate.pt')
                logger.info(" Custom license plate model loaded")
            elif os.path.exists('models/license_plate.pt'):
                self.license_plate_model = YOLO('models/license_plate.pt')
                logger.info("License plate model loaded from models folder")
            else:
                logger.warning(" License plate model not found. Using region-based OCR detection.")
                self.license_plate_model = None
        except Exception as e:
            logger.error(f" Failed to load license plate model: {e}")
            self.license_plate_model = None

    def detect_vehicles(self, image_path):
        """Detect vehicles in the image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None, []

        # Run vehicle detection
        results = self.vehicle_model(image, verbose=False)

        detected_vehicles = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id in self.vehicle_classes and confidence > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        vehicle_type = self.vehicle_classes[class_id]

                        detected_vehicles.append({
                            'type': vehicle_type,
                            'confidence': confidence,
                            'bbox': (int(x1), int(y1), int(x2), int(y2))
                        })

        return image, detected_vehicles

    def extract_license_plate_region(self, image, vehicle_bbox):
        """Extract potential license plate regions from vehicle"""
        x1, y1, x2, y2 = vehicle_bbox
        vehicle_crop = image[y1:y2, x1:x2]

        h, w = vehicle_crop.shape[:2]

        regions = [
            (0, int(h*0.6), w, h),  # Bottom region
            (0, int(h*0.4), w, int(h*0.8)),  # Middle-bottom region
        ]

        license_plates = []

        for region in regions:
            rx1, ry1, rx2, ry2 = region
            region_crop = vehicle_crop[ry1:ry2, rx1:rx2]

            gray = cv2.cvtColor(region_crop, cv2.COLOR_BGR2GRAY)

            processed_images = self.preprocess_for_ocr(gray)

            for processed_img in processed_images:
                text = self.extract_text_from_image(processed_img)
                if self.is_valid_license_plate(text):
                    license_plates.append({
                        'text': text,
                        'region': (rx1 + x1, ry1 + y1, rx2 + x1, ry2 + y1),
                        'confidence': 0.8
                    })

        return license_plates

    def preprocess_for_ocr(self, gray_image):
        """Apply various preprocessing techniques for better OCR"""

        processed_images = []

        # Original
        processed_images.append(gray_image)

        # Apply Gaussian blur and threshold
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(thresh1)

        # Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
        processed_images.append(adaptive_thresh)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        processed_images.append(morph)

        return processed_images

    def extract_text_from_image(self, image):
        """Extract text using EasyOCR"""
        try:
            results = self.ocr_reader.readtext(image)

            # Combine all detected text
            text_list = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence detections
                    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(cleaned_text) >= 3:
                        text_list.append(cleaned_text)

            return ' '.join(text_list) if text_list else ''
        except:
            return ''

    def is_valid_license_plate(self, text):
        """Check if extracted text looks like a license plate"""
        if not text or len(text.strip()) < 3:
            return False

        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

        patterns = [
            r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,2}$',
            r'^[0-9]{1,3}[A-Z]{1,3}[0-9]{1,4}$',
            r'^[A-Z0-9]{4,8}$',
        ]

        for pattern in patterns:
            if re.match(pattern, clean_text):
                return True

        has_letter = any(c.isalpha() for c in clean_text)
        has_number = any(c.isdigit() for c in clean_text)

        return has_letter and has_number and 3 <= len(clean_text) <= 10

    def process_single_image(self, image_path):
        """Process a single image and extract vehicle info"""
        print(f"Processing: {image_path}")

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
                best_plate = max(license_plates, key=lambda x: len(x['text'].replace(' ', '')))

                clean_text = best_plate['text'].replace(' ', '')
                unique_plates.add(clean_text)

            vehicle_info = {
                'vehicle_id': i + 1,
                'type': vehicle['type'],
                'confidence': vehicle['confidence'],
                'bbox': vehicle['bbox'],
                'license_plates': [best_plate] if best_plate else []
            }

            image_results['vehicles'].append(vehicle_info)

        total_vehicles = len(vehicles)
        detected_plates = len([v for v in vehicles if self.extract_license_plate_region(image, v['bbox'])])

        print(f"Detection Results for {os.path.basename(image_path)}:")
        print(f"Total vehicles detected: {total_vehicles}")

        vehicle_types = {}
        for vehicle in vehicles:
            vtype = vehicle['type']
            vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1

        for vtype, count in vehicle_types.items():
            print(f" {vtype.capitalize()}: {count}")

        if unique_plates:
            print(f"  License plates found: {len(unique_plates)}")
            for plate in unique_plates:
                print(f"      → {plate}")
        else:
            print("   ❌ No license plates detected")

        print("   " + "="*50)

        return image_results

    def process_video(self, video_path, output_video_path=None, frame_skip=5):
        """
        Process video file and extract vehicle info from frames

        Args:
            video_path: Path to input video
            output_video_path: Path to save annotated video (optional)
            frame_skip: Process every Nth frame (default: 5 for faster processing)
        """

        if not os.path.exists(video_path):
            print(f" Error: Video file not found at {video_path}")
            print(" Checking current directory...")
            current_files = [f for f in os.listdir('Models') if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
            if current_files:
                print(" Video files found in current directory:")
                for i, file in enumerate(current_files):
                    print(f"   {i+1}. {file}")
            else:
                print(" No video files found in current directory")
            return None

        cap = None
        backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]

        for backend in backends:
            try:
                cap = cv2.VideoCapture(video_path, backend)
                if cap.isOpened():
                    print(f" Video opened successfully with backend")
                    break
                else:
                    cap.release()
            except:
                continue

        if cap is None or not cap.isOpened():
            print(f" Error: Could not open video file {video_path}")
            print("Trying alternative approach...")
            return self.process_video_alternative(video_path, output_video_path, frame_skip)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps

        print(f"🎬 Video Info:")
        print(f"    File: {os.path.basename(video_path)}")
        print(f"    Duration: {duration:.1f} seconds")
        print(f"    Resolution: {width}x{height}")
        print(f"    FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        print(f"   Processing every {frame_skip} frames")
        print("   " + "="*50)

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

                    temp_frame_path = "temp_frame.jpg"
                    cv2.imwrite(temp_frame_path, frame)

                    _, vehicles = self.detect_vehicles(temp_frame_path)

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
                            best_plate = max(license_plates, key=lambda x: len(x['text'].replace(' ', '')))
                            clean_plate = best_plate['text'].replace(' ', '')
                            if len(clean_plate) >= 3:  # Valid plate
                                video_results['unique_plates'].add(clean_plate)

                        vehicle_data = {
                            'type': vehicle['type'],
                            'confidence': vehicle['confidence'],
                            'bbox': vehicle['bbox'],
                            'license_plate': best_plate['text'] if best_plate else None
                        }
                        frame_data['vehicles'].append(vehicle_data)

                    video_results['frame_results'].append(frame_data)

                    if out is not None:
                        annotated_frame = self.annotate_frame(frame, vehicles)
                        out.write(annotated_frame)

                    if processed_count % 10 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f" Progress: {progress:.1f}% ({processed_count} frames processed)")

                elif out is not None:
                    out.write(frame)

            if os.path.exists("temp_frame.jpg"):
                os.remove("temp_frame.jpg")

        except KeyboardInterrupt:
            print("\n Processing interrupted by user")

        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()

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

            # Extract and show license plate
            license_plates = self.extract_license_plate_region(frame, vehicle['bbox'])
            if license_plates:
                best_plate = max(license_plates, key=lambda x: len(x['text'].replace(' ', '')))
                if best_plate['text']:
                    px1, py1, px2, py2 = best_plate['region']
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.putText(annotated, best_plate['text'], (px1, py1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return annotated

    def print_video_summary(self, results):
        """Print summary of video processing results"""
        print("\n Video Processing Complete!")
        print("   " + "="*50)
        print(f"   Frames processed: {results['processed_frames']}")
        print(f"   Unique vehicle types detected: {len(results['unique_vehicles'])}")

        for vehicle_type in sorted(results['unique_vehicles']):
            print(f"      → {vehicle_type.capitalize()}")

        print(f"  Unique license plates found: {len(results['unique_plates'])}")

        if results['unique_plates']:
            for plate in sorted(results['unique_plates']):
                print(f"      → {plate}")
        else:
            print("      → No license plates detected")

        print("   " + "="*50)

    def save_video_results_to_csv(self, video_results, filename='video_detection_results.csv'):
        """Save video processing results to CSV"""
        if not video_results or not video_results['frame_results']:
            print("No video results to save")
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
                # Frame with no vehicles
                rows.append({
                    'video_path': video_results['video_path'],
                    'frame_number': frame_num,
                    'timestamp_seconds': timestamp,
                    'vehicle_id': 0,
                    'vehicle_type': 'None',
                    'confidence': 0,
                    'license_plate': 'No vehicles'
                })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f" Video results saved to {filename}")

    def process_video_alternative(self, video_path, output_video_path=None, frame_skip=5):
        """
        Alternative video processing using ffmpeg to extract frames
        """
        try:
            import subprocess
            import tempfile
            import shutil

            print(" Using alternative method: extracting frames with ffmpeg...")

            temp_dir = tempfile.mkdtemp()

            try:

                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vf', f'select=not(mod(n\\,{frame_skip}))',
                    '-vsync', 'vfr',
                    f'{temp_dir}/frame_%04d.jpg'
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(" FFmpeg extraction failed. Trying frame-by-frame approach...")
                    return self.process_video_frame_by_frame(video_path, frame_skip)

                frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.jpg')])

                if not frame_files:
                    print("No frames extracted")
                    return None

                print(f" Extracted {len(frame_files)} frames")

                video_results = {
                    'video_path': video_path,
                    'total_frames': len(frame_files) * frame_skip,
                    'processed_frames': len(frame_files),
                    'unique_vehicles': set(),
                    'unique_plates': set(),
                    'frame_results': []
                }

                for i, frame_file in enumerate(frame_files):
                    frame_path = os.path.join(temp_dir, frame_file)

                    # Process frame
                    _, vehicles = self.detect_vehicles(frame_path)

                    frame_data = {
                        'frame_number': (i + 1) * frame_skip,
                        'timestamp': ((i + 1) * frame_skip) / 30,
                        'vehicles': []
                    }

                    for vehicle in vehicles:
                        video_results['unique_vehicles'].add(vehicle['type'])

                        frame = cv2.imread(frame_path)
                        license_plates = self.extract_license_plate_region(frame, vehicle['bbox'])
                        best_plate = None

                        if license_plates:
                            best_plate = max(license_plates, key=lambda x: len(x['text'].replace(' ', '')))
                            clean_plate = best_plate['text'].replace(' ', '')
                            if len(clean_plate) >= 3:
                                video_results['unique_plates'].add(clean_plate)

                        vehicle_data = {
                            'type': vehicle['type'],
                            'confidence': vehicle['confidence'],
                            'bbox': vehicle['bbox'],
                            'license_plate': best_plate['text'] if best_plate else None
                        }
                        frame_data['vehicles'].append(vehicle_data)

                    video_results['frame_results'].append(frame_data)

                    if (i + 1) % 10 == 0:
                        print(f" Progress: {((i + 1) / len(frame_files)) * 100:.1f}%")

                self.print_video_summary(video_results)
                return video_results

            finally:

                shutil.rmtree(temp_dir)

        except ImportError:
            print("❌ subprocess not available. Trying manual frame extraction...")
            return self.process_video_frame_by_frame(video_path, frame_skip)
        except Exception as e:
            print(f"❌ Alternative method failed: {e}")
            return self.process_video_frame_by_frame(video_path, frame_skip)

    def process_video_frame_by_frame(self, video_path, frame_skip=5):
        """
        Last resort: try different OpenCV approaches
        """
        print(" Trying frame-by-frame processing with different settings...")
        approaches = [
            lambda: cv2.VideoCapture(video_path, cv2.CAP_FFMPEG),
            lambda: cv2.VideoCapture(video_path, cv2.CAP_ANY),
            lambda: cv2.VideoCapture(video_path, cv2.CAP_GSTREAMER),
        ]

        for i, approach in enumerate(approaches):
            try:
                cap = approach()
                if cap.isOpened():
                    print(f" Success with approach {i+1}")

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30

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

                            # Save frame temporarily
                            temp_frame = "temp_video_frame.jpg"
                            cv2.imwrite(temp_frame, frame)

                            # Process frame
                            _, vehicles = self.detect_vehicles(temp_frame)

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
                                    best_plate = max(license_plates, key=lambda x: len(x['text'].replace(' ', '')))
                                    clean_plate = best_plate['text'].replace(' ', '')
                                    if len(clean_plate) >= 3:
                                        video_results['unique_plates'].add(clean_plate)

                                vehicle_data = {
                                    'type': vehicle['type'],
                                    'confidence': vehicle['confidence'],
                                    'bbox': vehicle['bbox'],
                                    'license_plate': best_plate['text'] if best_plate else None
                                }
                                frame_data['vehicles'].append(vehicle_data)

                            video_results['frame_results'].append(frame_data)

                            if processed_count % 5 == 0:
                                progress = (frame_count / max(total_frames, frame_count)) * 100
                                print(f"Progress: {progress:.1f}%")

                        if processed_count > 100:  # Process max 100 frames
                            print(" Limiting processing to 100 frames for safety")
                            break

                    cap.release()

                    # Clean up
                    if os.path.exists("temp_video_frame.jpg"):
                        os.remove("temp_video_frame.jpg")

                    video_results['processed_frames'] = processed_count
                    self.print_video_summary(video_results)
                    return video_results

                else:
                    cap.release()

            except Exception as e:
                print(f"   Approach {i+1} failed: {e}")
                continue

        print("❌ All video processing approaches failed")
        print("💡 Suggestions:")
        print("   1. Check if the video file is corrupted")
        print("   2. Try converting the video to MP4 format")
        print("   3. Reduce video size/resolution")
        print("   4. Use a different video file")
        return None

    def process_dataset(self, dataset_path):
        """Process entire dataset"""
        dataset_path = Path(dataset_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f'*{ext}'))
            image_files.extend(dataset_path.glob(f'*{ext.upper()}'))

        print(f"Found {len(image_files)} images to process")

        all_results = []

        for image_file in image_files:
            result = self.process_single_image(str(image_file))
            if result:
                all_results.append(result)
                self.results.extend(result['vehicles'])

        return all_results

    def save_results_to_csv(self, filename='detection_results.csv'):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return

        rows = []
        for result in self.results:
            image_path = result.get('image_path', 'unknown')
            vehicle_id = result.get('vehicle_id', 'unknown')
            vehicle_type = result.get('type', 'unknown')
            confidence = result.get('confidence', 0)
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
                        'license_plate': plate['text'],
                        'plate_confidence': plate['confidence']
                    })
            else:
                rows.append({
                    'image_path': image_path,
                    'vehicle_id': vehicle_id,
                    'vehicle_type': vehicle_type,
                    'vehicle_confidence': confidence,
                    'vehicle_bbox': str(bbox),
                    'license_plate': 'Not detected',
                    'plate_confidence': 0
                })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def visualize_results(self, image_path, save_path=None):
        """Visualize detection results on image"""
        image = cv2.imread(image_path)
        if image is None:
            return

        # Detect vehicles
        _, vehicles = self.detect_vehicles(image_path)

        # Draw bounding boxes
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']

            # Draw vehicle bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add vehicle type label
            label = f"{vehicle['type']}: {vehicle['confidence']:.2f}"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Extract and show license plates
            license_plates = self.extract_license_plate_region(image, vehicle['bbox'])
            for plate in license_plates:
                if plate['text']:
                    # Draw license plate region (approximate)
                    px1, py1, px2, py2 = plate['region']
                    cv2.rectangle(image, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.putText(image, plate['text'], (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display or save result
        if save_path:
            cv2.imwrite(save_path, image)

        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        output_path = f"outputs/{file_id}_processed.jpg"
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        
def main():
        detector = CarDetectionSystem()

        detector.setup_license_plate_model()


# Run the main function
if __name__ == "__main__":
    main()

# Additional utility functions
def check_video_file(video_path):
    """Check if video file exists and get info"""
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")

        # Check current directory for video files
        current_dir = os.getcwd()
        video_files = [f for f in os.listdir(current_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'))]

        if video_files:
            print(" Video files in current directory:")
            for i, file in enumerate(video_files, 1):
                print(f"   {i}. {file}")
            return video_files
        else:
            print(" No video files found in current directory")
            return []

    # Try to get video info
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            print(f"✅ Video file OK: {os.path.basename(video_path)}")
            print(f"   Resolution: {int(width)}x{int(height)}")
            print(f"   FPS: {fps}")
            print(f"   Frames: {int(frame_count)}")
            cap.release()
            return True
        else:
            print(f"❌ Cannot open video: {video_path}")
            cap.release()
            return False
    except Exception as e:
        print(f"❌ Error checking video: {e}")
        return False

def process_video_simple(video_path):
    """Simplified video processing function with better error handling"""
    print(f" Starting video processing...")

    # First check if file exists and is valid
    if not check_video_file(video_path):
        return None

    detector = CarDetectionSystem()

    # Process with fallbacks
    results = detector.process_video(video_path, frame_skip=10)  # Skip more frames for speed

    if results:
        detector.save_video_results_to_csv(results, 'video_results.csv')
        print(" Video processing completed successfully!")
        return results
    else:
        print(" Video processing failed")
        return None

def process_video_only(video_path, save_annotated=False):
    """Quick function to process only a video file"""
    detector = CarDetectionSystem()

    output_path = None
    if save_annotated:
        name, ext = os.path.splitext(video_path)
        output_path = f"{name}_annotated{ext}"

    results = detector.process_video(video_path, output_path, frame_skip=3)

    if results:
        detector.save_video_results_to_csv(results, 'video_results.csv')
        return results
    return None

def batch_process_images(image_folder, output_folder):
    """Batch process images and save annotated results"""
    detector = CarDetectionSystem()

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_folder, image_file)
            output_path = os.path.join(output_folder, f"annotated_{image_file}")

            # Process and save annotated image
            detector.visualize_results(image_path, output_path)

def analyze_results(csv_file):
    """Analyze the detection results"""
    df = pd.read_csv(csv_file)

    print("=== Detection Results Analysis ===")
    print(f"Total vehicles detected: {len(df)}")
    print(f"Vehicles with license plates: {len(df[df['license_plate'] != 'Not detected'])}")
    print(f"Detection rate: {len(df[df['license_plate'] != 'Not detected']) / len(df) * 100:.1f}%")

    print("\nVehicle types distribution:")
    print(df['vehicle_type'].value_counts())

    print("\nSample license plates detected:")
    valid_plates = df[df['license_plate'] != 'Not detected']['license_plate'].unique()
    for plate in valid_plates[:10]:  # Show first 10
        print(f"  - {plate}")