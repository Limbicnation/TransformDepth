from transformers import pipeline
from PIL import Image
import numpy as np

# Load the image
image_path = 'transform-depth.jpg'
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

# Save the depth image to the root directory
output_path = 'depth-transform-depth.jpg'  # Naming the output file
depth_image.save(output_path)

