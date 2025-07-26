
import os
import xml.etree.ElementTree as ET
from PIL import Image

def convert_bbox_to_yolo(size, box):
    # size = (width, height) of image
    # box = (xmin, xmax, ymin, ymax) - absolute pixel coordinates
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def process_split(xml_split_dir, image_split_dir, output_label_split_dir, class_mapping):
    os.makedirs(output_label_split_dir, exist_ok=True)
    print(f"\nProcessing XMLs in: {xml_split_dir}")
    print(f"Looking for images in: {image_split_dir}")
    print(f"Saving YOLO labels to: {output_label_split_dir}")

    processed_count = 0
    skipped_count = 0

    for xml_file in os.listdir(xml_split_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_split_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_file}: {e}. Skipping.")
            skipped_count += 1
            continue

        # Get image filename from XML
        img_filename_tag = root.find('filename')
        if img_filename_tag is None or img_filename_tag.text is None:
            print(f"Warning: <filename> tag not found or empty in {xml_file}. Skipping.")
            skipped_count += 1
            continue
        img_filename = img_filename_tag.text

        # Handle cases where image filename in XML might have different extension than actual image
        base_img_name = os.path.splitext(img_filename)[0]
        possible_img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        actual_img_path = None
        for ext in possible_img_extensions:
            temp_path = os.path.join(image_split_dir, base_img_name + ext)
            if os.path.exists(temp_path):
                actual_img_path = temp_path
                break

        if actual_img_path is None:
            print(f"Warning: Image not found for {xml_file} (tried multiple extensions). Skipping.")
            skipped_count += 1
            continue

        try:
            img = Image.open(actual_img_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {actual_img_path}: {e}. Skipping {xml_file}.")
            skipped_count += 1
            continue

        output_txt_path = os.path.join(output_label_split_dir, base_img_name + '.txt')
        with open(output_txt_path, 'w') as f:
            objects_found = False
            for obj in root.findall('object'):
                class_name_tag = obj.find('name')
                if class_name_tag is None or class_name_tag.text is None:
                    print(f"Warning: <name> tag not found or empty for an object in {xml_file}. Skipping this object.")
                    continue
                class_name = class_name_tag.text

                if class_name not in class_mapping:
                    print(f"Warning: Class '{class_name}' from {xml_file} not in specified class_mapping. Skipping object.")
                    continue
                class_id = class_mapping[class_name]

                bndbox = obj.find('bndbox')
                if bndbox is None:
                    print(f"Warning: <bndbox> tag not found for an object in {xml_file}. Skipping this object.")
                    continue
                try:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                except (ValueError, AttributeError) as e:
                    print(f"Error parsing bounding box coordinates in {xml_file}: {e}. Skipping this object.")
                    continue

                # Ensure valid coordinates (min <= max and within image bounds)
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_width, xmax)
                ymax = min(img_height, ymax)

                if xmax <= xmin or ymax <= ymin:
                    print(f"Warning: Invalid bounding box (xmax <= xmin or ymax <= ymin) in {xml_file}. Skipping object.")
                    continue

                # Convert to YOLO format
                b = (xmin, xmax, ymin, ymax)
                yolo_bbox = convert_bbox_to_yolo((img_width, img_height), b)

                f.write(f"{class_id} {' '.join([str(coord) for coord in yolo_bbox])}\n")
                objects_found = True
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} XML files...")

        if not objects_found:
            print(f"No valid objects found or written for {xml_file}. Empty label file created.")


    print(f"\nFinished processing. Converted {processed_count} XMLs. Skipped {skipped_count} XMLs due to errors/missing images.")

if __name__ == '__main__':
    
    base_dataset_dir = 'myDataset'

    my_class_mapping = {
        "car": 0,
        "license_plate": 1 # Adjust these based on the exact class names in your XMLs
        # Add other classes if you have them, e.g., "bus": 2, "truck": 3
    }

    # Process training set
    xml_train_dir = os.path.join(base_dataset_dir, 'annotations_xml', 'train')
    image_train_dir = os.path.join(base_dataset_dir, 'images', 'train')
    output_label_train_dir = os.path.join(base_dataset_dir, 'labels', 'train')
    process_split(xml_train_dir, image_train_dir, output_label_train_dir, my_class_mapping)

    # Process validation set
    xml_val_dir = os.path.join(base_dataset_dir, 'annotations_xml', 'val')
    image_val_dir = os.path.join(base_dataset_dir, 'images', 'val')
    output_label_val_dir = os.path.join(base_dataset_dir, 'labels', 'val')
    process_split(xml_val_dir, image_val_dir, output_label_val_dir, my_class_mapping)

    # Process test set (optional)
    xml_test_dir = os.path.join(base_dataset_dir, 'annotations_xml', 'test')
    image_test_dir = os.path.join(base_dataset_dir, 'images', 'test')
    output_label_test_dir = os.path.join(base_dataset_dir, 'labels', 'test')
    if os.path.exists(xml_test_dir) and os.path.exists(image_test_dir):
        process_split(xml_test_dir, image_test_dir, output_label_test_dir, my_class_mapping)
    else:
        print(f"Skipping test set conversion as '{xml_test_dir}' or '{image_test_dir}' not found.")