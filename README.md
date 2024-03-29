## TransformDepth 🔄

**TransformDepth: Unleashing the Power of Transformers for Depth Estimation**

Dive into the world of depth estimation with the transformative power of state-of-the-art Transformer models. `TransformDepth` offers a streamlined approach to generate depth images from standard 2D pictures, leveraging the latest advancements in AI research.

## Environment Setup

To get started, you'll first need to create a virtual environment and activate it. Ensure you have [Conda](https://docs.conda.io/en/latest/) installed, then run the following commands:

```
conda create -n transform-depth python=3.8
conda activate transform-depth
```
## Installation

To install the necessary dependencies for `TransformDepth`, ensure you have set up and activated your virtual environment as described in the Environment Setup section. Then, you can easily install all required packages by running the following command:

```bash
pip install -r requirements.txt
```

# Load the image
```
image_path = 'transform-depth.jpg'
image = Image.open(image_path)
```
Replace 'transform-depth.jpg' with the path to your image file. For example, if your image is named myphoto.jpg and located in the same directory as the script, you would update the line to:

```
image_path = 'myphoto.jpg'
```
After updating the script with your image path, save the changes. You're now ready to run the depth estimation pipeline and transform your 2D image into a depth map.

## Running the Depth Estimation Pipeline

With the environment prepared and dependencies installed, you're now ready to run the depth estimation pipeline. 
Ensure you have an image ready for processing and execute the following command:

```
python DepthEstimationPipeline.py
```
This will generate a depth image based on your input, showcasing the capabilities of TransformDepth in transforming 2D images into their depth counterparts.
