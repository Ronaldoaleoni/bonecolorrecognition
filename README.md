# Automated Bone Recognition and Colorimetric Analysis

> **From Instance Segmentation to Munsell-Based Quantification of Taphonomic Variation**

This repository provides a complete two-phase pipeline for the **automated recognition**, **segmentation**, and **quantitative color analysis** of bone images.  
It combines **deep learning** (Detectron2 + PyTorch) for segmentation and **colorimetric analysis** (ΔE2000 in CIELAB space) for objective, reproducible taphonomic interpretation.

---

## Project Structure

project/
│
├── module_1_detectron/ # Detectron2-based bone segmentation
│ ├── README.md # Detailed setup
│ ├── bonne-detector/ # main pipeline
│ └── weights/ # Custom trained model weights (.pth)
│
├── module_2_color_analysis/ # Quantitative color extraction and Munsell matching
│ ├── README.md # Algorithm descriptions and usage examples
│ ├── assets/ # Munsell reference tables (CSV)
│ └── scripts/ # Color analysis scripts
│
├── requirements.txt # file to install dependencies through pip
└── README.md # General documentation


## Module 1 — Automated Bone Recognition (Mask R-CNN + Detectron2)

> **Instance segmentation pipeline for automatic bone extraction from complex scenes**

This module performs **automatic detection, segmentation, and extraction of individual bones** using the **Mask R-CNN** architecture implemented in [Detectron2](https://github.com/facebookresearch/detectron2).

Each bone is detected, masked, cropped, and exported on a **black 1024×1024 background**, creating standardized inputs for **Phase 2 (Colorimetric Analysis)**.

---

## Overview

### Objectives
- Detect and segment bones from complex images (e.g., excavation, lab photos).  
- Export each detected bone as an independent, background-free image.  
- Standardize all outputs to a fixed **1024×1024 px** resolution for further quantitative analysis.

### Model
- Architecture: `Mask R-CNN R-50-FPN (3x)`  
- Framework: [Detectron2](https://github.com/facebookresearch/detectron2)  
- Trained on: Custom COCO-formatted dataset  
- Classes: **1 (bone)**  

---

## Dataset Setup

The dataset must be in COCO format:

assets/
│
├── images/               # All test or validation images (.jpg, .png)
└── annotations.json      # COCO-style annotations

Register the dataset in the script by editing:

annotation_file_path_test = "<your-path-here>/assets/annotations.json"
image_directory_path_test = "<your-path-here>/assets/images"

Running the Script

## Model Inference and Visualization

The first section:

register_coco_instances(...)
predictor = DefaultPredictor(test_cfg)


creates a predictor and saves visualized detections with bounding boxes and masks into:

./test_set_predictions/


Each output image shows the predicted bone regions overlayed on the original photo.

## Bone Cropping and Export

The second part of the script automatically:

Reads all images from input_dir.

Runs inference to get instance masks.

Crops each detected bone with optional padding (padding_percentage = 0.1).

Resizes and centers each bone onto a 1024×1024 black canvas.

Saves each bone as a numbered file:

0001.png
0002.png
0003.png
...


Modify these variables as needed:

input_dir = "<your-path-here>/images"
output_dir = "<your-path-here>/output"
output_size = 1024
padding_percentage = 0.1


Each file will appear in output_dir:

✅ Saved 0001.png
✅ Saved 0002.png
...

## Module 2 — Quantitative Color Analysis

**Goal:** Extract representative bone colors and match them to Munsell reference chips.

**Core Idea:** Compute the **ΔE2000 color difference** between bone clusters (in CIELAB) and a Munsell ground-truth dataset.

**Available Algorithms:**
- Method - Description

- `gmm.py` - Gaussian Mixture Model clustering 
- `gmm_weighted.py` - GMM with covariance-weighted ΔE2000 with Lab-space interpolation 
- `gmm_interpolated.py` - GMM with Lab-space interpolation 
- `kmeans.py` - K-Means clustering 
- `kmeans_interpolated.py` - K-Means + interpolation 
- `median_cut.py` - Classic Median Cut quantization 
- `median_cut_interpolated.py` - Median Cut + interpolation 

Each method outputs:
- Cluster statistics (`min`, `max`, `mean`, `std`, `median` of ΔE2000)
- Visualizations:
  - Cluster comparison plots
  - ΔE2000 distributions
  - 3D Lab-space scatter plots
- CSV summaries with **Munsell notations** and **color matches**

---


**Detailed installation**
- Windows 11 
- Nvidia GPU

 - conda create --name bonecolorecog python=3.10.16

 - conda activate bonecolorecog

 - cd Desktop

 - cd bonecolorrecognition

 - conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -c nvidia

 - conda install -c nvidia cudnn=8.2.1

 - python -m pip install -r requirements.txt

 - conda install git

 - git clone https://github.com/facebookresearch/detectron2.git

 - cd detectron2

 - python -m pip install .

 - (If the installation fails, you may need the C++ compiler to build C/CUDA extensions) https://visualstudio.microsoft.com/visual-cpp-build-tools/ and download build tools

 - python -m pip install "Pillow<10"
 
 **import dependencies and versions**

 - print(torch.__version__)
 # Should be 1.12.1            
 - print(torch.cuda.is_available())
 # Should return True    
 - print(torch.version.cuda)  
 # Should show 11.3 (PyTorch's CUDA version)         
 - print(torch.backends.cudnn.version())  
 # Should show 8200 (cuDNN 8.2) or something near
 - print("detectron2:", detectron2.__version__)
 # Should show 0.6
 - print(numpy.__version__)
 # Should be 1.26.4
 - print(cv2.__version__)
 # Should be 4.11.0


**Citation**

If you use this project in research, please cite:

Leoni et al. (2025).
From Visual Perception to Quantitative Approach: Automated Bone Recognition, Color Clustering, and Munsell Matching for Objective Taphonomic Analysis.