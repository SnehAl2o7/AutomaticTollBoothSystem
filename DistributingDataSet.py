import os
import shutil
from pathlib import Path


def separate_xml_and_images(source_dir, output_dir):
    """
    Separates XML and image files from a source directory into organized folders.

    Args:
        source_dir (str): Path to the directory containing mixed files
        output_dir (str): Path to the directory where organized files will be stored
    """

    # Define file extensions
    xml_extensions = {'.xml'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'}

    # Create output directories
    xml_dir = Path(output_dir) / 'xml_files'
    images_dir = Path(output_dir) / 'image_files'

    # Create directories if they don't exist
    xml_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Counters for tracking
    xml_count = 0
    image_count = 0

    # Process files in source directory
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist!")
        return

    print(f"Processing files from: {source_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    for file_path in source_path.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            file_name = file_path.name

            try:
                if file_ext in xml_extensions:
                    # Copy XML file
                    dest_path = xml_dir / file_name
                    shutil.copy2(file_path, dest_path)
                    print(f"XML: {file_name} → xml_files/")
                    xml_count += 1

                elif file_ext in image_extensions:
                    # Copy image file
                    dest_path = images_dir / file_name
                    shutil.copy2(file_path, dest_path)
                    print(f"Image: {file_name} → image_files/")
                    image_count += 1

                else:
                    # Skip files that are neither XML nor images
                    print(f"Skipping: {file_name} (not XML or image file)")

            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

    # Print summary
    print("-" * 50)
    print("SUMMARY:")
    print(f"XML files processed: {xml_count}")
    print(f"Image files processed: {image_count}")
    print(f"Total files processed: {xml_count + image_count}")


def create_sample_dataset():
    """
    Creates a sample dataset with XML and image files for testing.
    """
    sample_dir = Path("sample_dataset")
    sample_dir.mkdir(exist_ok=True)

    # Create sample XML content
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <folder>images</folder>
    <filename>sample_image.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>object1</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>200</ymax>
        </bndbox>
    </object>
</annotation>"""

    # Create sample files
    with open(sample_dir / "annotation1.xml", "w") as f:
        f.write(xml_content)

    with open(sample_dir / "annotation2.xml", "w") as f:
        f.write(xml_content.replace("sample_image.jpg", "sample_image2.jpg"))

    # Create dummy image files (just empty files for demo)
    for img_name in ["image1.jpg", "image2.png", "photo.jpeg"]:
        (sample_dir / img_name).touch()

    # Create other files
    with open(sample_dir / "readme.txt", "w") as f:
        f.write("This is a sample dataset")

    print(f"Sample dataset created in: {sample_dir}")
    return str(sample_dir)


if __name__ == "__main__":
    # Example usage
    print("XML and Image File Separator")
    print("=" * 40)

    # Option 1: Use with your own directories - ADD YOUR PATHS HERE
    source_directory =  "Dataset-2/Images"# Replace with your source path
    output_directory =  "Dataset-2/OnlyImages"# Replace with your output path

    # Separate the files
    separate_xml_and_images(source_directory, output_directory)

    print(f"\nFiles have been organized in: {output_directory}")
    print("Check the following folders:")
    print("- xml_files/")
    print("- image_files/")