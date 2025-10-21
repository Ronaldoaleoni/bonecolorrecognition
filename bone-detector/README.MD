# Bonedetector

This project was developed using Python 3.10.16 and includes libraries focused on computer vision, deep learning, and numerical data processing.

It is part of the Automated Bone Recognition pipeline and performs the detection and segmentation of bones using Mask R-CNN implemented with Detectron2.

## Libraries Used

- **Python**: 3.10.16
- **cudatoolkit**: 11.3
- **conda install -c nvidia cudnn**: 8.2.1
- **pytorch**: 1.12.0
- **torchvision**: 0.13.0
- **torchaudio**: 0.12.0
- **cython**:  3.1.0
- **pycocotools**: 2.0.8
- **Detectron2**: 0.6
- **OpenCV (cv2)**: 4.11.0
- **NumPy**: 1.26.4
- **Matplotlib:** 3.10.0
- **wheel** 0.45.1
- **pandas** 2.2.3
- **Random**
- **Math**
- **git**
- **git clone https://github.com/facebookresearch/detectron2.git**

## Requirements

Before running the project, make sure that:

Python 3.10.16 is installed

CUDA and cuDNN are properly configured for your GPU

You are working inside a virtual or Conda environment


## Usage

This module performs:

Detection and segmentation of bones in input images

Mask extraction using Detectron2

Cropping and normalization of each segmented bone

Export of standardized 1024×1024 PNG images

## Author

Developed by Leoni et al. (2025).

Part of the From Visual Perception to Quantitative Approach: Automated Bone Recognition, Color Clustering, and Munsell Matching for Objective Taphonomic Analysis project.