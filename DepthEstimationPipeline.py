import argparse
import os
from transformers import pipeline
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import torch

def ensure_odd(value):
    """Ensure the value is an odd integer."""
    value = int(value)
    return value if value % 2 == 1 else value + 1

def convert_path(path):
    """Convert path for compatibility between Windows and WSL."""
    if os.name == 'nt':  # If running on Windows
        return path.replace('\\', '/')
    return path

def gamma_correction(image, gamma=1.0):
    """Apply gamma correction to the image."""
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return Image.fromarray(cv2.LUT(np.array(image), table))

def auto_gamma_correction(image):
    """Automatically adjust gamma correction for the image."""
    image_array = np.array(image).astype(np.float32) / 255.0
    mean_luminance = np.mean(image_array)
    gamma = np.log(0.5) / np.log(mean_luminance)
    return gamma_correction(image, gamma=gamma)

def process_image(image_path, output_path, blur_radius, median_size, device):
    # Ensure median_size is an odd integer
    median_size = ensure_odd(median_size)

    # Convert paths for compatibility
    image_path = convert_path(image_path)
    output_path = convert_path(output_path)

    # Check if the input image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The input image path does not exist: {image_path}")

    # Load the image
    image = Image.open(image_path)

    # Load the pipeline for depth estimation with the larger model
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", device=device)

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

    # Enhanced edge detection
    edges = depth_image.filter(ImageFilter.FIND_EDGES)
    edges = edges.point(lambda x: 255 if x > 50 else 0)  # Adjusted threshold

    # Create a mask from the edges
    mask = edges.convert("L")

    # Blur only the edges using the mask
    blurred_edges = depth_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Combine the blurred edges with the original depth image using the mask
    combined_image = Image.composite(blurred_edges, depth_image, mask)

    # Apply auto gamma correction
    gamma_corrected_image = auto_gamma_correction(combined_image)

    # Additional post-processing: Sharpen the final image
    final_image = gamma_corrected_image.filter(ImageFilter.SHARPEN)

    # Check if the output directory exists and create it if necessary
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the final depth image
    final_image.save(output_path)
    print(f"Processed and saved: {output_path}")

def main():
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Process images for depth estimation.")
    parser.add_argument("--single", type=str, help="Path to a single image file to process.")
    parser.add_argument("--batch", type=str, help="Path to directory of images to process in batch.")
    parser.add_argument("--output", type=str, help="Output directory for processed images.")
    parser.add_argument("--blur_radius", type=float, default=2.0, help="Radius for Gaussian Blur. Default is 2.0. Can accept float values.")
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
