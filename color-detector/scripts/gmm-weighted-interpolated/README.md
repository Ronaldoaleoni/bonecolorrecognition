# Algorithm: GMM Weighted Interpolated

## Overview
This folder contains the implementation of the **GMM Weighted Interpolated** algorithm for bone color detection and quantitative analysis.  
It uses the input bone images and metadata (CSV) to extract dominant color clusters and match them with Munsell color chips using the CIELAB color space.

## Contents
- `gmmwi.ipynb` — Jupyter Notebook with the full implementation.
- [Input ground truth](../../../color-detector/assets/real_converted.csv)
- [Input images](../../../bone-detector/assets/images/)
- Output files (color cluster metrics, plots, Munsell matches).

## Usage
 - To use algorithm we need to set the configurations

     BACKGROUND_RGB_THRESH = <int value>  # threshold to exclude near-black background (0-255)

     MAX_K = <int value>  # Maximum clusters for BIC optimization

     TOP_N = <int value>  # Number of top matches to display (increased to 5)

     SUBSAMPLE_SIZE = <int value>  # For BIC calculation efficiency

     INTERPOLATION_RESOLUTION = <int value> # Interpolation grid resolution

     USE_INTERPOLATION = <boolean value> # Enable interpolation for denser Munsell space

     MATCHING_METHOD = <str value>  # Options: 'centroid', 'covariance_weighted', 'mahalanobis'

     COVARIANCE_SAMPLES = <int value>  # Number of samples for covariance-weighted matching

     COVARIANCE_SCALE = <float value>  # Scale factor for covariance (1.0 = actual covariance)

## Method outputs:
- Cluster statistics (`min`, `max`, `mean`, `std`, `median` of ΔE2000)
- Visualizations:
  - Cluster comparison plots
  - ΔE2000 distributions
  - 3D Lab-space scatter plots
- CSV summaries with **Munsell notations** and **color matches**

