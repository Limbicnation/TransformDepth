import argparse
import os
from transformers import pipeline
from PIL import Image
import numpy as np

# Function to process a single image
def process_image(image_path, output_path):
    # Load the image
    image = Image.open(image_path)

    # Load the pipeline for depth estimation
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    # Perform depth estimation
    result = pipe(image)

    # Convert depth data to a NumPy array if not already one
    depth_data = np.array(result["depth"])

    # Normalize and convert to uint8
    depth_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min())
    depth_uint8 = (255 * depth_normalized).astype(np.uint8)

    # Create an image from the processed depth data
    depth_image = Image.fromarray(depth_uint8)

    # Save the depth image
    depth_image.save(output_path)
    print(f"Processed and saved: {output_path}")

# Setup command line argument parsing
parser = argparse.ArgumentParser(description="Process images for depth estimation.")
parser.add_argument("--single", type=str, help="Path to a single image file to process.")
parser.add_argument("--batch", type=str, help="Path to directory of images to process in batch.")
args = parser.parse_args()

# Process based on the input arguments
if args.single:
    # Process a single image
    output_path = 'depth-' + os.path.basename(args.single)  # Naming the output file
    process_image(args.single, output_path)
elif args.batch:
    # Process all images in the directory
    for filename in os.listdir(args.batch):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(args.batch, filename)
            output_path = os.path.join(args.batch, 'depth-' + filename)
            process_image(image_path, output_path)
else:
    print("Please specify either --single <image_path> or --batch <directory_path> to process images.")

