import argparse
import os
from pathlib import Path
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
    return path.replace('\\', '/') if os.name == 'nt' else path

def gamma_correction(img, gamma=1.0):
    """Apply gamma correction to the image."""
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return Image.fromarray(np.array(img).astype(np.uint8)).point(lambda i: table[i])

def auto_gamma_correction(image):
    """Automatically adjust gamma correction for the image."""
    image_array = np.array(image).astype(np.float32) / 255.0
    mean_luminance = np.mean(image_array)
    gamma = np.log(0.5) / np.log(mean_luminance)
    return gamma_correction(image, gamma=gamma)

def auto_contrast(image):
    """Apply automatic contrast adjustment to the image."""
    return ImageOps.autocontrast(image)

def process_image(image_path, output_path, blur_radius, median_size, device, model_path, encoder):
    try:
        # Ensure median_size is an odd integer
        median_size = ensure_odd(median_size)

        # Convert paths for compatibility
        image_path = convert_path(image_path)
        output_path = convert_path(output_path)

        # Check if the input image path exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"The input image path does not exist: {image_path}")

        # Load the image
        image = Image.open(image_path)

        # Load the Depth Anything V2 model
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(encoder=encoder, features=256, out_channels=[256, 512, 1024, 1024])
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # Perform depth estimation
        raw_img = np.array(image.convert("RGB"))
        depth = model.infer_image(raw_img)  # HxW raw depth map

        # Convert depth data to a NumPy array if not already one
        depth_data = np.array(depth)

        # Normalize and convert to uint8
        depth_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min() + 1e-8)  # Avoid zero division
        depth_uint8 = (255 * depth_normalized).astype(np.uint8)

        # Create an image from the processed depth data
        depth_image = Image.fromarray(depth_uint8)

        # Apply a median filter to reduce noise
        depth_image = depth_image.filter(ImageFilter.MedianFilter(size=median_size))

        # Enhanced edge detection with more feathering
        edges = depth_image.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.GaussianBlur(radius=2 * blur_radius))
        edges = edges.point(lambda x: 255 if x > 20 else 0)  # Adjusted threshold

        # Create a mask from the edges
        mask = edges.convert("L")

        # Blur only the edges using the mask
        blurred_edges = depth_image.filter(ImageFilter.GaussianBlur(radius=blur_radius * 2))

        # Combine the blurred edges with the original depth image using the mask
        combined_image = Image.composite(blurred_edges, depth_image, mask)

        # Apply auto gamma correction with a lower gamma to darken the image
        gamma_corrected_image = gamma_correction(combined_image, gamma=0.7)

        # Apply auto contrast
        final_image = auto_contrast(gamma_corrected_image)

        # Additional post-processing: Sharpen the final image
        final_image = final_image.filter(ImageFilter.SHARPEN)

        # Ensure the output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the final depth image
        final_image.save(output_path)
        print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def process_images_in_batch(batch_path, output_dir, blur_radius, median_size, device, model_path, encoder):
    for filename in os.listdir(batch_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Case-insensitive check
            image_path = os.path.join(batch_path, filename)
            output_path = os.path.join(output_dir, 'depth-' + filename)
            process_image(image_path, output_path, blur_radius, median_size, device, model_path, encoder)

def main():
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Process images for depth estimation.")
    parser.add_argument("--single", type=str, help="Path to a single image file to process.")
    parser.add_argument("--batch", type=str, help="Path to directory of images to process in batch.")
    parser.add_argument("--output", type=str, help="Output directory for processed images.")
    parser.add_argument("--blur_radius", type=float, default=2.0, help="Radius for Gaussian Blur. Default is 2.0. Can accept float values.")
    parser.add_argument("--median_size", type=int, default=5, help="Size for Median Filter. Default is 5. Must be an odd integer.")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="Device to use for inference: 'cpu' or 'gpu'. Default is 'cpu'.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Depth Anything V2 model checkpoint.")
    parser.add_argument("--encoder", type=str, choices=["vits", "vitb", "vitl", "vitg"], default="vitl", help="Encoder type for the Depth Anything V2 model. Default is 'vitl'.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"

    if args.single:
        output_path = output_dir / ('depth-' + Path(args.single).name)
        process_image(args.single, str(output_path), args.blur_radius, args.median_size, device, args.model_path, args.encoder)
    elif args.batch:
        process_images_in_batch(args.batch, str(output_dir), args.blur_radius, args.median_size, device, args.model_path, args.encoder)
    else:
        print("Please specify either --single <image_path> or --batch <directory_path> to process images.")

if __name__ == "__main__":
    main()
