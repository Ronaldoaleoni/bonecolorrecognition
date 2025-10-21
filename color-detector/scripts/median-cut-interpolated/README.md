# Algorithm: Median Cut Interpolated

## Overview
This folder contains the implementation of the **Median Cut Interpolated** algorithm for bone color detection and quantitative analysis.  
It uses the input bone images and metadata (CSV) to extract dominant color clusters and match them with Munsell color chips using the CIELAB color space.

## Contents
- `mediancuti.ipynb` — Jupyter Notebook with the full implementation.
- [Input ground truth](../../../color-detector/assets/real_converted.csv)
- [Input images](../../../bone-detector/assets/images/)
- Output files (color cluster metrics, plots, Munsell matches).

## Usage
 - To use algorithm we need to set the configurations

     BACKGROUND_RGB_THRESH = <int value>  # threshold to exclude near-black background (0-255)

     QUANTIZATION_LEVELS = <int value>  # Number of colors for median cut quantization (power of 2 works best)

     TOP_N = <int value>  # Number of top matches to display

     USE_INTERPOLATION = <boolean value>  # Enable interpolation for denser Munsell space

     INTERPOLATION_RESOLUTION = <int value>  # Controls interpolation density

## Method outputs:
- Cluster statistics (`min`, `max`, `mean`, `std`, `median` of ΔE2000)
- Visualizations:
  - Cluster comparison plots
  - ΔE2000 distributions
  - 3D Lab-space scatter plots
- CSV summaries with **Munsell notations** and **color matches**
