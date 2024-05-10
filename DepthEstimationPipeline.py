import argparse
import os
from transformers import pipeline
from PIL import Image, ImageFilter
import numpy as np

# Function to process a single image
def process_image(image_path, output_path, blur_radius, median_size):
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

    # Apply Gaussian blur to smooth edges
    depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Apply a median filter to further reduce noise
    depth_image = depth_image.filter(ImageFilter.MedianFilter(size=median_size))

    # Save the depth image
    depth_image.save(output_path)
    print(f"Processed and saved: {output_path}")

# Setup command line argument parsing
parser = argparse.ArgumentParser(description="Process images for depth estimation.")
parser.add_argument("--single", type=str, help="Path to a single image file to process.")
parser.add_argument("--batch", type=str, help="Path to directory of images to process in batch.")
parser.add_argument("--output", type=str, help="Output directory for processed images.")
parser.add_argument("--blur_radius", type=float, default=4.0, help="Radius for Gaussian Blur. Default is 4.0. Can accept float values.")
parser.add_argument("--median_size", type=int, default=5, help="Size for Median Filter. Default is 5. Must be an integer.")
args = parser.parse_args()

# Ensure the output directory exists
if args.output and not os.path.exists(args.output):
    os.makedirs(args.output)

# Process based on the input arguments
if args.single:
    # Process a single image
    if args.output:
        output_path = os.path.join(args.output, 'depth-' + os.path.basename(args.single))
    else:
        output_path = 'depth-' + os.path.basename(args.single)
    process_image(args.single, output_path, args.blur_radius, args.median_size)
elif args.batch:
    # Process all images in the directory
    for filename in os.listdir(args.batch):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(args.batch, filename)
            if args.output:
                output_path = os.path.join(args.output, 'depth-' + filename)
            else:
                output_path = os.path.join(args.batch, 'depth-' + filename)
            process_image(image_path, output_path, args.blur_radius, args.median_size)
else:
    print("Please specify either --single <image_path> or --batch <directory_path> to process images.")
