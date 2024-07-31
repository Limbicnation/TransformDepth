

---

## TransformDepth V2 🔄

**TransformDepth: Unleashing the Power of Transformers for Depth Estimation**

Dive into the world of depth estimation with the transformative power of state-of-the-art Transformer models. `TransformDepth` offers a streamlined approach to generate depth images from standard 2D pictures, leveraging the latest advancements in AI research.

<p align="center">
  <img src="https://github.com/Limbicnation/TransformDepth/blob/main/images/transform-depth.jpg" alt="Original Image" width="48%"/>
  <img src="https://github.com/Limbicnation/TransformDepth/blob/main/images/depth-transform-depth.jpg" alt="Depth Image" width="48%"/>
</p>

<p align="center">
  <img src="https://github.com/Limbicnation/TransformDepth/blob/main/images/single.webp" alt="Original Image" width="48%"/>
  <img src="https://github.com/Limbicnation/TransformDepth/blob/main/images/depth-single.webp" alt="Depth Image" width="48%"/>
</p>

<p align="center">
  <img src="https://github.com/Limbicnation/TransformDepth/blob/main/images/resized_face_analysis_model.webp" alt="Original Image" width="48%"/> 
  <img src="https://github.com/Limbicnation/TransformDepth/blob/main/images/depth-resized_face_analysis_model.png" alt="Depth Image" width="48%"/>
</p>

## Environment Setup ✨

To get started, you'll first need to create a virtual environment and activate it. Ensure you have [Conda](https://docs.conda.io/en/latest/) installed, then run the following commands:

```
conda create -n depth-anything-2 python=3.11.9
conda activate depth-anything-v2
```

## Installation

To install the necessary dependencies for `TransformDepth`, ensure you have set up and activated your virtual environment as described in the Environment Setup section. Then, you can easily install all required packages by running the following command:

```bash
pip install -r requirements.txt
```

## Load the Image

```python
image_path = 'transform-depth.jpg'
image = Image.open(image_path)
```

Replace `'transform-depth.jpg'` with the path to your image file. For example, if your image is named `myphoto.jpg` and located in the same directory as the script, you would update the line to:

```python
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

## Batch Image Processing

For batch processing, specify the directory containing the images and the output directory where the processed images will be saved. This command will apply the depth estimation to all images in the specified directory:

## Run the Script for Batch/Single Image Processing with a Specified Output Directory 🌟

```
python DepthEstimationPipeline.py --batch ..\path\to\your\images --output ..\images\batch
```

## Single Image Processing

This command runs the `DepthEstimationPipeline.py` script to process a single image located at `..\images\single.webp`.

It specifies an output directory at `..\images` where the processed image will be saved.

```
python DepthEstimationPipeline.py --single ..\images\single.webp --output ..\images --blur_radius 0.3 --median_size 3
```

  ## Command Line Arguments

- `--single`: Path to a single image file to process.
- `--batch`: Path to directory of images to process in batch.
- `--output`: Output directory for processed images (required).
- `--blur_radius`: Radius for Gaussian Blur. Default is 2.0. Can accept float values.
- `--median_size`: Size for Median Filter. Default is 5. Must be an odd integer.
- `--depth-anything-v2-small`: Flag to use the Depth-Anything-V2-Small model.
- `--flag`: A flag to trigger additional processing options.
- `--no-post-processing`: Disable post-processing effects.
- `--apply-gamma`: Apply gamma correction to the output.
- `--gamma-value`: Gamma value for correction. Default is 1.0 (no correction).

### Example Command with Post-Processing

The Gaussian blur applied has a radius of 0.3, providing slight smoothing to reduce noise without significantly blurring the image details. A median filter of size 3x3 is also applied, which helps further reduce noise and smooth out small artifacts in the image. The command is ideal for refining depth estimations with minimal loss of detail in high-resolution or detailed images.

```
DepthEstimationPipeline.py --single images/resized_face_analysis_model.png --output . --depth-anything-v2-small --blur_radius 1 --median_size 3 --apply-gamma --gamma-value 1.5
```

## Citation

[DepthAnything](https://github.com/LiheYoung/Depth-Anything?tab=readme-ov-file)

```
@inproceedings{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      booktitle={CVPR},
      year={2024}
}
```

---
