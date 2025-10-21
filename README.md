# Automated Bone Recognition and Colorimetric Analysis

> **From Instance Segmentation to Munsell-Based Quantification of Taphonomic Variation**

This repository provides a complete two-phase pipeline for the **automated recognition**, **segmentation**, and **quantitative color analysis** of bone images.  
It combines **deep learning** (Detectron2 + PyTorch) for segmentation and **colorimetric analysis** (ΔE2000 in CIELAB space) for objective, reproducible taphonomic interpretation.

---

# Project Structure

- project/
  - module_1_detectron/ # Detectron2-based bone segmentation
    - README.md # Detailed setup
    - bonne-detector/ # main pipeline
    - detectron2/ # Custom trained model weights (.pth)
    - assets/ # Images and annotations
  - module_2_color_analysis/ # Quantitative color extraction and Munsell matching
    - README.md # Algorithm descriptions and usage examples
    - assets/ # Munsell reference tables (CSV)
    - scripts/ # Color analysis scripts
  - requirements.txt # file to install dependencies through pip
  - README.md # General documentation


# Module 1 — Automated Bone Recognition (Mask R-CNN + Detectron2)

 **Instance segmentation pipeline for automatic bone extraction from complex scenes**

 This module performs **automatic detection, segmentation, and extraction of individual bones** using the **Mask R-CNN** architecture implemented in [Detectron2](https://github.com/facebookresearch/detectron2).

 Each bone is detected, masked, cropped, and exported on a **black 1024×1024 background**, creating standardized inputs for **Phase 2 (Colorimetric Analysis)**

 ## Overview

 ### **Objectives**
 - Detect and segment bones from complex images (e.g., excavation, lab photos).  
 - Export each detected bone as an independent, background-free image.  
 - Standardize all outputs to a fixed **1024×1024 px** resolution for further quantitative analysis.

 ### **Model**
 - Architecture: `Mask R-CNN R-50-FPN (3x)`  
 - Framework: Detectron2  
 - Trained on: Custom COCO-formatted dataset  
 - Classes: **1 (bone)**  

 ### **Usage**

 This module performs:

 - Detection and segmentation of bones in input images
 - Mask extraction using Detectron2
 - Cropping and normalization of each segmented bone
 - Export of standardized 1024×1024 PNG images

---
# Module 2 — Quantitative Color Analysis

 ## Overview 
 
 Extract representative bone colors and match them to Munsell reference chips, computing the **ΔE2000 color difference** between bone clusters (in CIELAB) and a Munsell ground-truth dataset.

 ### **Available Algorithms:**

Method - Description

 - [GMM](color-detector/scripts/gmm/README.md) - Gaussian Mixture Model clustering 
 - [GMM Weighted Interpolated](color-detector/scripts/gmm-weighted-interpolated/README.md) - GMM with covariance-weighted ΔE2000 with Lab-space interpolation 
 - [GMM Interpolated](color-detector/scripts/gmm-interpolated/README.md) - GMM with Lab-space interpolation 
 - [K-Means](color-detector/scripts/k-means/README.md) K-Means clustering 
 - [K-Means Interpolated](color-detector/scripts/k-means-interpolated/README.md) - K-Means + interpolation 
 - [Median Cut](color-detector/scripts/median-cut/README.md) - Classic Median Cut quantization 
 - [Median Cut Interpolated](color-detector/scripts/median-cut-interpolated/README.md) - Median Cut + interpolation 

 ### **Output files**

 - Color cluster metrics 
 - Plots 
 - Munsell matches

---
 ### **Requeriments**

 - The entire procedure were made in Windows 11 (64-bit). with a Nvidia GPU

 - The model was trained in a NVIDIA RTX 4060 Ti (8 GB VRAM), but the inference and analysis (this current project) can also be performed on entry-level GPUs, such as NVIDIA MX110 (2 GB VRAM)

 - Before running the project, also make sure that:

   - Python 3.10.16 is installed

   - CUDA and cuDNN are properly configured for your GPU

   - You are working inside a virtual or Conda environment
   

  ### **installation**

 - Detailed installation can be accessed [here](bone-detector/README.MD)

 - Color module requisites are detailed [here](color-detector/README.MD)

  ## Citation

 If you use this project in a scientific publication, please cite:

 - *Leoni, R. A. et al. (2025). From Visual Perception to Quantitative Approach: Automated Bone Recognition, Color Clustering, and Munsell Matching for Objective Taphonomic Analysis.*  
- *Detectron2:* Wu, Y. et al. (2019). Detectron2. https://github.com/facebookresearch/detectron2  
- *Munsell dataset:* Timofeev. V et al. (2025). Munsell Re-renotation Revised. IEEE Dataport.



