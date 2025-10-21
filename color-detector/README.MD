# Image Analysis and Processing Environment

This environment is configured for projects focused on **quantitative color analysis** and **scientific image processing**.  
It integrates widely used libraries in **data science**, **computer vision**, and **machine learning**, ensuring **reproducibility** and **standardization** of results.

---

## Main Dependencies

| Library | Version | Description |

- **re**: 2.32.3 — Text pattern manipulation using regular expressions  
- **numpy**: 1.26.4 — Numerical computation and matrix operations  
- **pandas**: 2.2.3 — Data analysis and tabular manipulation  
- **scikit-image (skimage)**: 0.25.2 — Image processing and analysis  
- **colour**: 0.4.6 — Color conversion and computation across multiple color spaces  
- **matplotlib**: 3.10.0 — Scientific plotting and data visualization  
- **scikit-learn (sklearn)**: 1.6.1 — Machine learning algorithms and clustering  
- **scipy**: 1.15.3 — Advanced mathematical, statistical, and scientific functions  
- **tqdm**: 4.67.1 — Progress bars for tracking long or iterative tasks  

---
 
## Contents

The following algorithms were implemented to detect and quantify bone colors.
Each folder contains the implementation and example outputs for the respective method.

| Algorithm | Folder | Description |
|------------|---------|-------------|
| **GMM** | [`scripts/gmm/`](scripts/gmm/) | Gaussian Mixture Model for unsupervised color clustering. |
| **GMM Interpolated** | [`scripts/gmm-interpolated/`](scripts/gmm-interpolated/) | GMM using interpolated Ground Truth for smoother cluster assignment. |
| **GMM Weighted Interpolated** | [`scripts/gmm-weighted-interpolated/`](scripts/gmm-weighted-interpolated/) | GMM with pixel-weighting and interpolated Ground Truth to improve local accuracy. |
| **K-Means** | [`scripts/k-means/`](scripts/k-means/) | K-Means clustering for basic color segmentation. |
| **K-Means Interpolated** | [`scripts/k-means-interpolated/`](scripts/k-means-interpolated/) | K-Means with interpolated Ground Truth for enhanced color smoothness. |
| **Median Cut** | [`scripts/median-cut/`](scripts/median-cut/) | Median Cut algorithm for palette extraction. |
| **Median Cut Interpolated** | [`scripts/median-cut-interpolated/`](scripts/median-cut-interpolated/) | Median Cut with interpolated Ground Truth for refined color distribution. |


