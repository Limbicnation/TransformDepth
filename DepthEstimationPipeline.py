import argparse
import cv2
import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import torch
import sys

# Add the Depth Anything V2 path
sys.path.insert(0, '/mnt/TurboTux/AnacondaWorkspace/Github/TransformDepth/Depth-Anything-V2')

# Import Depth Anything V2
from depth_anything_v2.dpt import DepthAnythingV2

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

def process_image(image_path, output_path, blur_radius, threshold, device, model_path, encoder, input_size):
    try:
        # Convert paths for compatibility
        image_path = convert_path(image_path)
        output_path = convert_path(output_path)

        # Check if the input image path exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"The input image path does not exist: {image_path}")

        # Load the image
        image = Image.open(image_path)

        # Resize image if input_size is specified
        if input_size:
            image = image.resize((input_size, input_size), Image.LANCZOS)

        # Load the Depth Anything V2 model
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        depth_anything = DepthAnythingV2(**model_configs[encoder])

        # Verifying the model path before loading
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model checkpoint does not exist: {model_path}")

        depth_anything.load_state_dict(torch.load(model_path, map_location='cpu'))
        depth_anything = depth_anything.to(device).eval()

        # Perform depth estimation
        raw_img = np.array(image.convert("RGB"))
        depth = depth_anything.infer_image(raw_img)  # HxW raw depth map

        # Convert depth data to a NumPy array if not already one
        depth_data = np.array(depth)

        # Normalize and convert to uint8
        depth_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min() + 1e-8)  # Avoid zero division
        depth_uint8 = (255 * depth_normalized).astype(np.uint8)

        # Create an image from the processed depth data
        depth_image = Image.fromarray(depth_uint8)

        # Enhanced edge detection with more feathering
        edges = depth_image.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        edges = edges.point(lambda x: 255 if x > threshold else 0)  # Adjusted threshold

        # Create a mask from the edges
        mask = edges.convert("L")

        # Combine the blurred edges with the original depth image using the mask
        combined_image = Image.composite(depth_image, depth_image.filter(ImageFilter.GaussianBlur(radius=blur_radius)), mask)

        # Apply auto gamma correction with a different gamma value to match the standard method
        gamma_corrected_image = gamma_correction(combined_image, gamma=1.0)

        # Apply auto contrast to match the standard method
        final_image = auto_contrast(gamma_corrected_image)

        # Ensure the output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the final depth image
        final_image.save(output_path)
        print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def process_images_in_batch(batch_path, output_dir, blur_radius, threshold, device, model_path, encoder, input_size):
    for filename in os.listdir(batch_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):  # Case-insensitive check
            image_path = os.path.join(batch_path, filename)
            output_path = os.path.join(output_dir, 'depth-' + filename)
            process_image(image_path, output_path, blur_radius, threshold, device, model_path, encoder, input_size)

def main():
    parser = argparse.ArgumentParser(description="Process images for depth estimation.")
    parser.add_argument("--single", type=str, help="Path to a single image file to process.")
    parser.add_argument("--batch", type=str, help="Path to directory of images to process in batch.")
    parser.add_argument("--output", type=str, help="Output directory for processed images.")
    parser.add_argument("--blur_radius", type=float, default=1.0, help="Radius for Gaussian Blur. Default is 1.0. Can accept float values.")
    parser.add_argument("--threshold", type=int, default=20, help="Threshold for edge detection. Default is 20.")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="Device to use for inference: 'cpu' or 'gpu'. Default is 'cpu'.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Depth Anything V2 model checkpoint.")
    parser.add_argument("--encoder", type=str, choices=["vits", "vitb", "vitl", "vitg"], default="vitl", help="Encoder type for the Depth Anything V2 model. Default is 'vitl'.")
    parser.add_argument("--input_size", type=int, help="Resize input images to this size before processing.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"

    if args.single:
        output_path = output_dir / ('depth-' + Path(args.single).name)
        process_image(args.single, str(output_path), args.blur_radius, args.threshold, device, args.model_path, args.encoder, args.input_size)
    elif args.batch:
        process_images_in_batch(args.batch, str(output_dir), args.blur_radius, args.threshold, device, args.model_path, args.encoder, args.input_size)
    else:
        print("Please specify either --single <image_path> or --batch <directory_path> to process images.")

if __name__ == "__main__":
    main()
