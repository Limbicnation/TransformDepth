# file path: DepthEstimationPipeline.py
import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
import cv2

# Constants
MAX_PIXEL_VALUE = 255
EPSILON = 1e-8

def gamma_correction(img: Image.Image, gamma: float) -> Image.Image:
    """Apply gamma correction to the image."""
    if gamma <= 0:
        raise ValueError("Gamma value must be greater than zero.")
    
    inv_gamma = 1.0 / gamma
    max_pixel_value = 65535 if img.mode == "I;16" else 255
    table = [(i / max_pixel_value) ** inv_gamma * max_pixel_value for i in range(max_pixel_value + 1)]
    table = np.array(table, dtype=np.uint16 if max_pixel_value == 65535 else np.uint8)
    
    if img.mode == "I;16":
        img = img.point(lambda i: table[i], 'I;16')
    else:
        img = img.point(lambda i: table[i])
    
    return img

def auto_gamma_correction(image: Image.Image, gamma: float) -> Image.Image:
    """Automatically adjust gamma correction for the image."""
    return gamma_correction(image, gamma=gamma)

def ensure_odd(value: int) -> int:
    """Ensure the value is an odd integer."""
    return value if value % 2 != 0 else value + 1

def convert_path(path: str) -> str:
    """Convert paths for compatibility."""
    return os.path.normpath(path)

def load_image(image_path: str) -> Image.Image:
    """Load an image from the given path using PIL."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The input image path does not exist: {image_path}")
    return Image.open(image_path)

def process_depth_data(depth_data: np.ndarray) -> Image.Image:
    """Normalize and convert depth data to an image."""
    depth_min = depth_data.min()
    depth_max = depth_data.max()
    print(f"Depth Min: {depth_min}, Depth Max: {depth_max}")  # Debug print

    depth_normalized = (depth_data - depth_min) / (depth_max - depth_min + EPSILON)
    depth_uint8 = (MAX_PIXEL_VALUE * depth_normalized).astype(np.uint8)
    depth_uint8 = cv2.equalizeHist(depth_uint8)
    return Image.fromarray(depth_uint8)

def apply_median_filter(image: Image.Image, size: int) -> Image.Image:
    """Apply a median filter to the image."""
    return image.filter(ImageFilter.MedianFilter(size=size))

def detect_edges(image: Image.Image) -> Image.Image:
    """Detect edges in the image."""
    return image.filter(ImageFilter.CONTOUR)  # Use CONTOUR instead of FIND_EDGES for smoother edges

def process_image(image_path: str, output_path: str, blur_radius: float, median_size: int, flag: bool, no_post_processing: bool, apply_gamma: bool, gamma_value: float):
    """Process the image and save the output."""
    median_size = ensure_odd(median_size)

    image_path = convert_path(image_path)
    output_path = convert_path(output_path)

    # Load the image
    raw_img = load_image(image_path)

    # Convert image to RGB format to ensure compatibility with transformers pipeline
    raw_img = raw_img.convert("RGB")
    
    # Optionally convert to NumPy array (if needed)
    raw_img_np = np.array(raw_img)  # Convert to NumPy array in HWC format

    # Using transformers pipeline for Depth-Anything-V2-Small
    print("Using transformers pipeline for depth estimation.")
    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", channel_axis=-1)
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

    # Prepare the image for the model
    inputs = image_processor(images=raw_img_np, return_tensors="pt")  # Pass the NumPy array

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=raw_img.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Visualize the prediction
    depth_array = prediction.squeeze().cpu().numpy()
    formatted = (depth_array * 255 / np.max(depth_array)).astype("uint8")
    depth_image = Image.fromarray(formatted)

    # Apply gamma correction if specified
    if apply_gamma:
        depth_image = auto_gamma_correction(depth_image, gamma_value)

    # Apply post-processing filters only if no_post_processing is False
    if not no_post_processing:
        # Apply median filter
        depth_image = apply_median_filter(depth_image, median_size)

        # Apply Gaussian blur
        depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Detect edges
        edges = detect_edges(depth_image)

        # Convert both images to NumPy arrays for blending
        depth_array = np.array(depth_image)
        edges_array = np.array(edges)

        # Ensure both arrays are of the same dtype
        if depth_array.dtype != edges_array.dtype:
            edges_array = edges_array.astype(depth_array.dtype)

        # Apply add operation to combine the depth image and edges
        combined_array = np.clip((depth_array * 0.8 + edges_array * 0.2), 0, MAX_PIXEL_VALUE)

        # Convert back to PIL Image
        combined_image = Image.fromarray(combined_array.astype(np.uint8))

        # Set the final image to the combined image
        edges = combined_image
    else:
        edges = depth_image

    # Save the processed image with max bit depth for PNG
    edges.save(output_path, format="PNG", bits=16 if depth_image.mode == "I;16" else 8)
    print(f"Processed and saved: {output_path}")

    median_size = ensure_odd(median_size)

    image_path = convert_path(image_path)
    output_path = convert_path(output_path)

    # Load the image
    raw_img = load_image(image_path)

    # Using transformers pipeline for Depth-Anything-V2-Small
    print("Using transformers pipeline for depth estimation.")
    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

    # Prepare the image for the model
    inputs = image_processor(images=raw_img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=raw_img.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Visualize the prediction
    depth_array = prediction.squeeze().cpu().numpy()
    formatted = (depth_array * 255 / np.max(depth_array)).astype("uint8")
    depth_image = Image.fromarray(formatted)

    # Apply gamma correction if specified
    if apply_gamma:
        depth_image = auto_gamma_correction(depth_image, gamma_value)

    # Apply post-processing filters only if no_post_processing is False
    if not no_post_processing:
        # Apply median filter
        depth_image = apply_median_filter(depth_image, median_size)

        # Apply Gaussian blur
        depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Detect edges
        edges = detect_edges(depth_image)

        # Convert both images to NumPy arrays for blending
        depth_array = np.array(depth_image)
        edges_array = np.array(edges)

        # Ensure both arrays are of the same dtype
        if depth_array.dtype != edges_array.dtype:
            edges_array = edges_array.astype(depth_array.dtype)

        # Apply add operation to combine the depth image and edges
        combined_array = np.clip((depth_array * 0.8 + edges_array * 0.2), 0, MAX_PIXEL_VALUE)

        # Convert back to PIL Image
        combined_image = Image.fromarray(combined_array.astype(np.uint8))

        # Set the final image to the combined image
        edges = combined_image
    else:
        edges = depth_image

    # Save the processed image with max bit depth for PNG
    edges.save(output_path, format="PNG", bits=16 if depth_image.mode == "I;16" else 8)
    print(f"Processed and saved: {output_path}")

def main():
    """Main function to parse arguments and process images."""
    parser = argparse.ArgumentParser(description="Process images for depth estimation.")
    parser.add_argument("--single", type=str, help="Path to a single image file to process.")
    parser.add_argument("--batch", type=str, help="Path to directory of images to process in batch.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed images.")
    parser.add_argument("--blur_radius", type=float, default=2.0, help="Radius for Gaussian Blur. Default is 2.0. Can accept float values.")
    parser.add_argument("--median_size", type=int, default=5, help="Size for Median Filter. Default is 5. Must be an odd integer.")
    parser.add_argument("--depth-anything-v2-small", action='store_true', help="Flag to use Depth-Anything-V2-Small model.")
    parser.add_argument("--flag", action='store_true', help="A flag to trigger additional processing options.")
    parser.add_argument("--no-post-processing", action='store_true', help="Disable post-processing effects.")
    parser.add_argument("--apply-gamma", action='store_true', help="Apply gamma correction to the output.")
    parser.add_argument("--gamma-value", type=float, default=1.0, help="Gamma value for correction. Default is 1.0 (no correction).")
    args = parser.parse_args()

    if not args.depth_anything_v2_small:
        raise ValueError("The --depth-anything-v2-small flag is required to use the small model version.")

    if args.single:
        output_path = os.path.join(args.output, 'depth-' + os.path.basename(args.single))
        process_image(args.single, output_path, args.blur_radius, args.median_size, args.flag, args.no_post_processing, args.apply_gamma, args.gamma_value)
    elif args.batch:
        for filename in os.listdir(args.batch):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                image_path = os.path.join(args.batch, filename)
                output_path = os.path.join(args.output, 'depth-' + filename)
                process_image(image_path, output_path, args.blur_radius, args.median_size, args.flag, args.no_post_processing, args.apply_gamma, args.gamma_value)
    else:
        print("Please specify either --single <image_path> or --batch <directory_path> to process images.")

if __name__ == "__main__":
    main()
