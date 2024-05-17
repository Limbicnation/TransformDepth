import argparse
import os
from transformers import pipeline
from PIL import Image, ImageFilter, ImageChops, ImageOps
import numpy as np
import torch

def ensure_odd(value):
    """Ensure the value is an odd integer."""
    value = int(value)
    return value if value % 2 == 1 else value + 1

def process_image(image_path, output_path, blur_radius, median_size, device):
    # Ensure median_size is an odd integer
    median_size = ensure_odd(median_size)

    # Load the image
    image = Image.open(image_path)

    # Load the pipeline for depth estimation
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=device)

    # Perform depth estimation
    if device == 0:
        image = image.convert("RGB")  # Ensure image is in RGB format
        inputs = pipe.feature_extractor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = pipe.model(**inputs)
        result = pipe.post_process(outputs, (image.height, image.width))
    else:
        result = pipe(image)

    # Convert depth data to a NumPy array if not already one
    depth_data = np.array(result["depth"])

    # Normalize and convert to uint8
    depth_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min() + 1e-8)  # Avoid zero division
    depth_uint8 = (255 * depth_normalized).astype(np.uint8)

    # Create an image from the processed depth data
    depth_image = Image.fromarray(depth_uint8)

    # Apply a median filter to reduce noise
    depth_image = depth_image.filter(ImageFilter.MedianFilter(size=median_size))

    # Detect edges in the depth image with a higher threshold
    edges = depth_image.filter(ImageFilter.FIND_EDGES)
    edges = edges.point(lambda x: 255 if x > 100 else 0)  # Adjusted threshold

    # Create a mask from the edges
    mask = edges.convert("L")

    # Blur only the edges using the mask
    blurred_edges = depth_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Combine the blurred edges with the original depth image using the mask
    combined_image = Image.composite(blurred_edges, depth_image, mask)

    # Save the final depth image
    combined_image.save(output_path)
    print(f"Processed and saved: {output_path}")

def main():
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Process images for depth estimation.")
    parser.add_argument("--single", type=str, help="Path to a single image file to process.")
    parser.add_argument("--batch", type=str, help="Path to directory of images to process in batch.")
    parser.add_argument("--output", type=str, help="Output directory for processed images.")
    parser.add_argument("--blur_radius", type=float, default=4.0, help="Radius for Gaussian Blur. Default is 4.0. Can accept float values.")
    parser.add_argument("--median_size", type=int, default=5, help="Size for Median Filter. Default is 5. Must be an odd integer.")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="Device to use for inference: 'cpu' or 'gpu'. Default is 'cpu'.")
    args = parser.parse_args()

    # Ensure the output directory exists
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)

    # Determine the device (GPU or CPU)
    device = 0 if args.device == "gpu" and torch.cuda.is_available() else -1

    # Process based on the input arguments
    if args.single:
        # Process a single image
        output_path = os.path.join(args.output, 'depth-' + os.path.basename(args.single))
        process_image(args.single, output_path, args.blur_radius, args.median_size, device)
    elif args.batch:
        # Process all images in the directory
        for filename in os.listdir(args.batch):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                image_path = os.path.join(args.batch, filename)
                output_path = os.path.join(args.output, 'depth-' + filename)
                process_image(image_path, output_path, args.blur_radius, args.median_size, device)
    else:
        print("Please specify either --single <image_path> or --batch <directory_path> to process images.")

if __name__ == "__main__":
    main()
