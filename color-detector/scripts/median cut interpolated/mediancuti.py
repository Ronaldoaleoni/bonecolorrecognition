"""
Enhanced compute ΔE2000 for bone images against Munsell ground-truth using median cut color quantization.
Designed for segmented bone images with black background.
Includes comprehensive statistics calculation (min, max, mean, std).
"""

import os
import re
import numpy as np
import pandas as pd
from skimage import io, color
from colour import xyY_to_XYZ
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from skimage.util import img_as_float
from skimage.color import deltaE_ciede2000, lab2rgb
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.interpolate import RegularGridInterpolator  
import colour  
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========== USER CONFIG ==========
BONE_IMAGES_DIR = r"<your-path-here>"
MUNSELL_CSV = r"<your-path-here>\assets\real_converted.csv"
OUTPUT_DIR = r"<your-path-here>"
BACKGROUND_RGB_THRESH = 10  # threshold to exclude near-black background (0-255)
QUANTIZATION_LEVELS = 32  # Number of colors for median cut quantization (power of 2 works best)
TOP_N = 5  # Number of top matches to display
USE_INTERPOLATION = True  # Enable interpolation for denser Munsell space
INTERPOLATION_RESOLUTION = 5  # Controls interpolation density
# =================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_munsell_hue_notation(hue_value):
    """
    Converts a H_GT value to a full Munsell hue notation.
    Handles both numeric and string inputs.
    """
    if pd.isna(hue_value):
        return "Unknown"
    
    # If it's already a string, return it as is
    if isinstance(hue_value, str):
        return hue_value
    
    # If it's numeric, use the mapping
    try:
        hue_number = float(hue_value)
        hue_map = {
            5: "5R", 7: "7.5R", 10: "10R", 12: "2.5YR", 15: "5YR",
            17: "7.5YR", 20: "10YR", 22: "2.5Y", 25: "5Y", 30: "10Y", 35: "5GY"
        }
        return hue_map.get(int(hue_number), f"{hue_number}H")
    except (ValueError, TypeError):
        return "Unknown"

def get_full_munsell_notation(row):
    """Get full Munsell notation (H V/C) from a row."""
    # Try different possible column names for H, V, C
    h = None
    v = None
    c = None
    
    # Check for various possible column names
    for h_col in ['h', 'H_GT', 'H', 'h_gt', 'Hue']:
        if h_col in row and not pd.isna(row[h_col]):
            h = row[h_col]
            break
            
    for v_col in ['V', 'V_GT', 'v_gt', 'v', 'Value']:
        if v_col in row and not pd.isna(row[v_col]):
            v = row[v_col]
            break
            
    for c_col in ['C', 'C_GT', 'c_gt', 'c', 'Chroma']:
        if c_col in row and not pd.isna(row[c_col]):
            c = row[c_col]
            break
    
    if h is None or v is None or c is None:
        return "Unknown"
    
    hue_notation = get_munsell_hue_notation(h)
    return f"{hue_notation} {v}/{c}"

def create_interpolated_munsell_space(gt_df, resolution=INTERPOLATION_RESOLUTION):
    """Creates an interpolated Munsell color space from the ground truth data."""
    print("Creating interpolated Munsell color space...")
    
    h_values = np.unique(gt_df['H_GT'].values)
    v_values = np.unique(gt_df['V_GT'].values)
    c_values = np.unique(gt_df['C_GT'].values)
    
    h_grid, v_grid, c_grid = np.meshgrid(h_values, v_values, c_values, indexing='ij')
    
    l_grid = np.full_like(h_grid, np.nan, dtype=float)
    a_grid = np.full_like(h_grid, np.nan, dtype=float)
    b_grid = np.full_like(h_grid, np.nan, dtype=float)
    
    for _, row in gt_df.iterrows():
        h_idx = np.where(h_values == row['H_GT'])[0][0]
        v_idx = np.where(v_values == row['V_GT'])[0][0]
        c_idx = np.where(c_values == row['C_GT'])[0][0]
        
        l_grid[h_idx, v_idx, c_idx] = row['L_gt']
        a_grid[h_idx, v_idx, c_idx] = row['a_gt']
        b_grid[h_idx, v_idx, c_idx] = row['b_gt']
    
    l_interpolator = RegularGridInterpolator(
        (h_values, v_values, c_values), 
        l_grid, 
        method='linear', 
        bounds_error=False, 
        fill_value=None
    )
    
    a_interpolator = RegularGridInterpolator(
        (h_values, v_values, c_values), 
        a_grid, 
        method='linear', 
        bounds_error=False, 
        fill_value=None
    )
    
    b_interpolator = RegularGridInterpolator(
        (h_values, v_values, c_values), 
        b_grid, 
        method='linear', 
        bounds_error=False, 
        fill_value=None
    )
    
    h_fine = np.linspace(min(h_values), max(h_values), resolution * len(h_values))
    v_fine = np.linspace(min(v_values), max(v_values), resolution * len(v_values))
    c_fine = np.linspace(min(c_values), max(c_values), resolution * len(c_values))
    
    h_fine_grid, v_fine_grid, c_fine_grid = np.meshgrid(h_fine, v_fine, c_fine, indexing='ij')
    
    points = np.column_stack((h_fine_grid.ravel(), v_fine_grid.ravel(), c_fine_grid.ravel()))
    
    l_fine = l_interpolator(points).reshape(h_fine_grid.shape)
    a_fine = a_interpolator(points).reshape(h_fine_grid.shape)
    b_fine = b_interpolator(points).reshape(h_fine_grid.shape)
    
    interpolated_data = []
    for i in range(len(h_fine)):
        for j in range(len(v_fine)):
            for k in range(len(c_fine)):
                if not np.isnan(l_fine[i, j, k]):
                    interpolated_data.append({
                        'H_GT': h_fine[i],
                        'V_GT': v_fine[j],
                        'C_GT': c_fine[k],
                        'L_gt': l_fine[i, j, k],
                        'a_gt': a_fine[i, j, k],
                        'b_gt': b_fine[i, j, k]
                    })
    
    interpolated_df = pd.DataFrame(interpolated_data)
    print(f"Created interpolated Munsell space with {len(interpolated_df)} points")
    
    # Add full Munsell notation to interpolated data
    interpolated_df['munsell_notation'] = interpolated_df.apply(get_full_munsell_notation, axis=1)
    
    # Add RGB values for display
    lab_values = interpolated_df[['L_gt', 'a_gt', 'b_gt']].values
    rgb_values = lab2rgb(lab_values.reshape(-1, 1, 3)).reshape(-1, 3)
    interpolated_df[['R_display', 'G_display', 'B_display']] = rgb_values
    
    return interpolated_df

def median_cut_quantization(rgb_pixels, num_colors):
    """
    Implement median cut algorithm for color quantization.
    
    Args:
        rgb_pixels: Array of RGB pixels (N, 3)
        num_colors: Number of colors to quantize to (should be power of 2)
    
    Returns:
        quantized_colors: Array of quantized colors (num_colors, 3)
        labels: Cluster labels for each pixel (N,)
    """
    # Make a copy to avoid modifying original data
    pixels = rgb_pixels.copy()
    
    # Initialize with all pixels in one bucket
    buckets = [pixels]
    
    # Split buckets until we reach the desired number of colors
    while len(buckets) < num_colors:
        new_buckets = []
        
        for bucket in buckets:
            if len(bucket) == 0:
                continue
                
            # Find the color channel with the greatest range
            r_range = np.max(bucket[:, 0]) - np.min(bucket[:, 0])
            g_range = np.max(bucket[:, 1]) - np.min(bucket[:, 1])
            b_range = np.max(bucket[:, 2]) - np.min(bucket[:, 2])
            
            # Determine which channel to split along
            if r_range >= g_range and r_range >= b_range:
                channel = 0  # Red
            elif g_range >= r_range and g_range >= b_range:
                channel = 1  # Green
            else:
                channel = 2  # Blue
            
            # Sort the bucket along the selected channel
            sorted_bucket = bucket[bucket[:, channel].argsort()]
            
            # Find the median and split
            median_idx = len(sorted_bucket) // 2
            new_buckets.append(sorted_bucket[:median_idx])
            new_buckets.append(sorted_bucket[median_idx:])
        
        buckets = new_buckets
        
        # If we've reached the desired number of buckets, break
        if len(buckets) >= num_colors:
            break
    
    # Calculate the average color for each bucket
    quantized_colors = []
    for bucket in buckets:
        if len(bucket) > 0:
            avg_color = np.mean(bucket, axis=0)
            quantized_colors.append(avg_color)
    
    # If we have fewer colors than requested, duplicate the last color
    while len(quantized_colors) < num_colors:
        quantized_colors.append(quantized_colors[-1])
    
    quantized_colors = np.array(quantized_colors)
    
    # Assign each pixel to the closest quantized color
    labels = np.zeros(len(pixels), dtype=int)
    for i, pixel in enumerate(pixels):
        distances = np.sqrt(np.sum((quantized_colors - pixel) ** 2, axis=1))
        labels[i] = np.argmin(distances)
    
    return quantized_colors, labels

def load_munsell_groundtruth(csv_path, use_interpolation=USE_INTERPOLATION):
    """Enhanced ground-truth loading with interpolation option."""
    df = pd.read_csv(csv_path)
    
    # Normalize column names for case-insensitive matching
    col_lower = {c.lower(): c for c in df.columns}
    
    # Check for different column naming conventions
    has_xyY = ('x' in col_lower and 'y' in col_lower and 'y' in col_lower)
    has_rgb = ('r' in col_lower and 'g' in col_lower and 'b' in col_lower)
    has_lab = ('l' in col_lower and 'a' in col_lower and 'b' in col_lower)

    if has_xyY and 'y' in col_lower:
        # xyY conversion
        x = df['x'].values.astype(float)
        y = df['y'].values.astype(float)
        Y = df['Y'].values.astype(float)
        
        xyz = np.array([xyY_to_XYZ((xx, yy, yy_val)) for xx, yy, yy_val in zip(x,y,Y)])
        lab = color.xyz2lab(xyz / 100.0)
        df[['L_gt','a_gt','b_gt']] = lab
        
    elif has_rgb:
        # RGB conversion
        rgb_cols = [col_lower['r'], col_lower['g'], col_lower['b']]
        rgb = df[rgb_cols].values.astype(float) / 255.0
        lab = color.rgb2lab(rgb.reshape(-1,1,3)).reshape(-1,3)
        df[['L_gt','a_gt','b_gt']] = lab
        
    elif has_lab:
        # Direct Lab values
        lab_cols = [col_lower['l'], col_lower['a'], col_lower['b']]
        df['L_gt'] = df[lab_cols[0]]
        df['a_gt'] = df[lab_cols[1]]
        df['b_gt'] = df[lab_cols[2]]
    else:
        raise ValueError(f"Ground-truth CSV doesn't contain recognizable color columns. Found: {list(df.columns)}")

    # Add full Munsell notation
    df['munsell_notation'] = df.apply(get_full_munsell_notation, axis=1)
    
    # Add RGB values for display
    lab_values = df[['L_gt', 'a_gt', 'b_gt']].values
    rgb_values = lab2rgb(lab_values.reshape(-1, 1, 3)).reshape(-1, 3)
    df[['R_display', 'G_display', 'B_display']] = rgb_values
    
    # Apply interpolation if requested and possible
    if use_interpolation and 'H_GT' in df.columns and 'V_GT' in df.columns and 'C_GT' in df.columns:
        return create_interpolated_munsell_space(df)
    
    return df

def extract_bone_pixels(img_path, bg_thresh=BACKGROUND_RGB_THRESH):
    """Enhanced bone pixel extraction with better handling."""
    img = io.imread(img_path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    
    # Create mask for non-background pixels
    mask = np.sum(img, axis=2) > bg_thresh * 3
    bone_pixels_rgb = img[mask]
    
    if bone_pixels_rgb.size == 0:
        return None, None
    
    # Convert to Lab color space using the same method as pipeline 1
    # FIXED: Changed ast to astype
    bone_pixels_rgb_float = bone_pixels_rgb.astype(np.float32) / 255.0
    bone_pixels_rgb_float_reshaped = bone_pixels_rgb_float.reshape(-1, 1, 3)
    
    # Use cv2-style conversion for consistency
    import cv2
    bone_lab = cv2.cvtColor(bone_pixels_rgb_float_reshaped, cv2.COLOR_RGB2Lab)
    bone_lab = bone_lab.reshape(-1, 3)
    
    return bone_lab, bone_pixels_rgb

def quantize_bone_pixels_median_cut(bone_rgb, bone_lab, num_colors=QUANTIZATION_LEVELS):
    """
    Apply median cut quantization to bone pixels.
    
    Args:
        bone_rgb: RGB values of bone pixels (N, 3)
        bone_lab: Lab values of bone pixels (N, 3)
        num_colors: Number of colors to quantize to
    
    Returns:
        centers: Quantized color centers in Lab space (num_colors, 3)
        cluster_sizes: Number of pixels in each quantized color
        labels: Cluster labels for each pixel
    """
    # Apply median cut to RGB pixels
    quantized_colors_rgb, labels = median_cut_quantization(bone_rgb, num_colors)
    
    # Convert quantized colors to Lab space
    quantized_colors_rgb_float = quantized_colors_rgb.astype(np.float32) / 255.0
    quantized_colors_rgb_float_reshaped = quantized_colors_rgb_float.reshape(-1, 1, 3)
    
    import cv2
    quantized_colors_lab = cv2.cvtColor(quantized_colors_rgb_float_reshaped, cv2.COLOR_RGB2Lab)
    quantized_colors_lab = quantized_colors_lab.reshape(-1, 3)
    
    # Calculate cluster sizes
    cluster_sizes = np.bincount(labels, minlength=num_colors)
    
    return quantized_colors_lab, cluster_sizes, labels

def find_top_matches_enhanced(cluster_center, gt_df, top_n=TOP_N):
    """Enhanced matching using CIEDE2000 from colour library."""
    gt_lab_arr = gt_df[['L_gt','a_gt','b_gt']].values.astype(float)
    
    # Use colour library for CIEDE2000 (more accurate)
    dE = colour.delta_E(cluster_center[np.newaxis, :], gt_lab_arr, method='CIE 2000')
    
    # Get indices of top N matches (lowest dE)
    top_indices = np.argsort(dE)[:top_n]
    
    results = []
    for idx in top_indices:
        row = gt_df.iloc[idx]
        rgb_values = (row.get('R_display', 0), row.get('G_display', 0), row.get('B_display', 0))
        
        # Extract H, V, C values using various possible column names
        h_val = None
        v_val = None
        c_val = None
        
        for h_col in ['h', 'H_GT', 'H', 'h_gt', 'Hue']:
            if h_col in row and not pd.isna(row[h_col]):
                h_val = row[h_col]
                break
                
        for v_col in ['V', 'V_GT', 'v_gt', 'v', 'Value']:
            if v_col in row and not pd.isna(row[v_col]):
                v_val = row[v_col]
                break
                
        for c_col in ['C', 'C_GT', 'c_gt', 'c', 'Chroma']:
            if c_col in row and not pd.isna(row[c_col]):
                c_val = row[c_col]
                break
        
        results.append({
            'dE': dE[idx],
            'H': h_val,
            'V': v_val,
            'C': c_val,
            'L_gt': row['L_gt'],
            'a_gt': row['a_gt'],
            'b_gt': row['b_gt'],
            'munsell_notation': row.get('munsell_notation', 'Unknown'),
            'rgb_values': rgb_values
        })
    
    return results

def plot_3d_scatter_best_cluster(bone_pixels_cluster, munsell_chip_lab, output_path):
    """Enhanced 3D scatter plot for the best cluster with perceptual coloring."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for visualization if too many points
    if len(bone_pixels_cluster) > 5000:
        indices = np.random.choice(len(bone_pixels_cluster), 5000, replace=False)
        plot_pixels = bone_pixels_cluster[indices]
    else:
        plot_pixels = bone_pixels_cluster
    
    # Calculate distance from each point to the Munsell chip
    distances = np.sqrt(np.sum((plot_pixels - munsell_chip_lab)**2, axis=1))
    
    # Normalize distances for colormap
    norm = plt.Normalize(vmin=distances.min(), vmax=distances.max())
    cmap = plt.cm.plasma  # Perceptually uniform sequential colormap
    
    # Plot points with color based on distance
    scatter = ax.scatter(plot_pixels[:, 0], plot_pixels[:, 1], plot_pixels[:, 2], 
                         c=distances, cmap=cmap, norm=norm, alpha=0.4, s=10)
    
    # Plot the Munsell chip
    ax.scatter(munsell_chip_lab[0], munsell_chip_lab[1], munsell_chip_lab[2], 
               c='red', marker='o', s=300, label='Munsell Chip', edgecolors='black')
    
    ax.set_xlabel('L*')
    ax.set_ylabel('a*')
    ax.set_zlabel('b*')
    ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Distance from Munsell Chip (ΔE)')
    
    # Title removed as requested
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_enhanced_visualization(img_path, cluster_centers, cluster_sizes, top_matches_list, 
                                optimal_k, bone_lab, cluster_labels, gt_df, output_dir, img_name):
    """Enhanced visualization for median cut quantization."""
    
    # 1. Main comparison plot (showing top 5 clusters)
    top_indices = np.argsort([matches[0]['dE'] for matches in top_matches_list])[:min(5, optimal_k)]
    
    fig, axes = plt.subplots(1, len(top_indices) + 1, figsize=(22, 5))
    
    # Original image
    img = io.imread(img_path)
    axes[0].imshow(img)
    axes[0].set_title(f"({optimal_k} colors via Median Cut)")
    axes[0].axis('off')
    
    # Top matches visualization
    for i, cluster_idx in enumerate(top_indices):
        match = top_matches_list[cluster_idx][0]  # Best match for this cluster
        
        # Get RGB color for display
        rgb_color = match['rgb_values']
        
        axes[i + 1].imshow([[rgb_color]])
        
        munsell_notation = match['munsell_notation']
        rgb_text = f"RGB: ({rgb_color[0]:.2f}, {rgb_color[1]:.2f}, {rgb_color[2]:.2f})"
        
        axes[i + 1].set_title(f"Color {cluster_idx + 1} Match\n{munsell_notation}\n{rgb_text}\nΔE2000: {match['dE']:.2f}\nSize: {cluster_sizes[cluster_idx]} px")
        axes[i + 1].axis('off')
        axes[i + 1].add_patch(Rectangle((0, 0), 1, 1, facecolor=rgb_color, edgecolor='black', linewidth=2))

    # Main title removed as requested
    main_plot_path = os.path.join(output_dir, f"{img_name}_median_cut_analysis.png")
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 3D scatter plot for best cluster
    best_cluster_idx = np.argmin([matches[0]['dE'] for matches in top_matches_list])
    best_cluster_pixels = bone_lab[cluster_labels == best_cluster_idx]
    best_match = top_matches_list[best_cluster_idx][0]
    best_munsell_lab = np.array([best_match['L_gt'], best_match['a_gt'], best_match['b_gt']])
    
    scatter_path = os.path.join(output_dir, f"{img_name}_3d_scatter.png")
    plot_3d_scatter_best_cluster(best_cluster_pixels, best_munsell_lab, scatter_path)
    
    # 3. Detailed analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Bone Image')  # Simple title
    axes[0, 0].axis('off')
    
    # Cluster centers in Lab space
    colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
    for i, (center, size) in enumerate(zip(cluster_centers, cluster_sizes)):
        rgb_color = lab2rgb(center.reshape(1, 1, 3)).reshape(3)
        rgb_color = np.clip(rgb_color, 0, 1)
        axes[0, 1].scatter(center[1], center[2], color=rgb_color, s=size/10, 
                          alpha=0.7, edgecolor='black', label=f'Color {i+1}')
    
    axes[0, 1].set_xlabel('a*')
    axes[0, 1].set_ylabel('b*')
    axes[0, 1].set_title('Color Centers in Lab Space (size = pixel count/10)')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # ΔE distribution by cluster
    for cluster_idx, (top_matches, color) in enumerate(zip(top_matches_list, colors)):
        dE_values = [match['dE'] for match in top_matches]
        axes[1, 0].plot(range(1, min(len(dE_values), TOP_N)+1), dE_values[:TOP_N], 
                       'o-', color=color, label=f'Color {cluster_idx+1}')
    
    axes[1, 0].set_xlabel('Match Rank')
    axes[1, 0].set_ylabel('ΔE2000')
    axes[1, 0].set_title('ΔE2000 for Top Matches by Color')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Color swatches for best cluster (top 5 matches)
    best_matches = top_matches_list[best_cluster_idx][:5]
    for i, match in enumerate(best_matches):
        rgb_color = match['rgb_values']
        axes[1, 1].add_patch(plt.Rectangle((i, 0), 0.8, 0.8, color=rgb_color, edgecolor='black'))
        axes[1, 1].text(i + 0.4, -0.15, f"ΔE={match['dE']:.2f}", ha='center', fontsize=10)
        munsell_text = match['munsell_notation']
        axes[1, 1].text(i + 0.4, 0.9, munsell_text, ha='center', fontsize=9, rotation=45)
    
    axes[1, 1].set_xlim(-0.5, 5.5)
    axes[1, 1].set_ylim(-0.5, 1.2)
    axes[1, 1].set_title(f'Top 5 Matches for Best Color ({best_cluster_idx + 1})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    detailed_path = os.path.join(output_dir, f"{img_name}_detailed_analysis.png")
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_single_image(img_path, gt_df, output_dir):
    """Process a single bone image with median cut quantization."""
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    
    try:
        # Extract bone pixels
        bone_lab, bone_rgb = extract_bone_pixels(img_path)
        if bone_lab is None:
            print(f"Skipping {img_path} - no bone pixels detected")
            return []
        
        print(f"  Found {len(bone_lab)} bone pixels")
        
        # Apply median cut quantization
        cluster_centers, cluster_sizes, cluster_labels = quantize_bone_pixels_median_cut(
            bone_rgb, bone_lab, QUANTIZATION_LEVELS)
        
        print(f"  Quantized into {QUANTIZATION_LEVELS} colors")
        
        # Find top matches for each color
        top_matches_list = []
        cluster_results = []
        
        for i, center in enumerate(cluster_centers):
            top_matches = find_top_matches_enhanced(center, gt_df, top_n=TOP_N)
            top_matches_list.append(top_matches)
            
            # Store results
            for j, match in enumerate(top_matches):
                cluster_results.append({
                    "image_path": img_path,
                    "image_name": img_name,
                    "color_id": i+1,
                    "color_size": cluster_sizes[i],
                    "color_size_pct": cluster_sizes[i] / len(bone_lab) * 100,
                    "quantization_levels": QUANTIZATION_LEVELS,
                    "color_L": center[0],
                    "color_a": center[1],
                    "color_b": center[2],
                    "match_rank": j+1,
                    "dE2000": match['dE'],
                    "gt_H": match['H'],
                    "gt_V": match['V'],
                    "gt_C": match['C'],
                    "gt_L": match['L_gt'],
                    "gt_a": match['a_gt'],
                    "gt_b": match['b_gt'],
                    "munsell_notation": match['munsell_notation'],
                    "rgb_R": match['rgb_values'][0],
                    "rgb_G": match['rgb_values'][1],
                    "rgb_B": match['rgb_values'][2],
                    "interpolation_used": USE_INTERPOLATION  # Added: Record interpolation status
                })
        
        # Calculate statistics for this image
        if cluster_results:
            dE_values = [result['dE2000'] for result in cluster_results if result['match_rank'] == 1]
            if dE_values:
                stats = {
                    "image_path": img_path,
                    "image_name": img_name,
                    "min_dE": np.min(dE_values),
                    "max_dE": np.max(dE_values),
                    "mean_dE": np.mean(dE_values),
                    "std_dE": np.std(dE_values),
                    "median_dE": np.median(dE_values),
                    "num_colors": QUANTIZATION_LEVELS
                }
                # Add these stats to your results or save separately
        
        # Create enhanced visualizations
        create_enhanced_visualization(img_path, cluster_centers, cluster_sizes, top_matches_list,
                                    QUANTIZATION_LEVELS, bone_lab, cluster_labels, gt_df, output_dir, img_name)
        
        print(f"  Generated visualizations and analysis")
        return cluster_results
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

def calculate_statistics(all_results):
    """Calculate comprehensive statistics from all results."""
    if not all_results:
        return None
    
    # Convert to DataFrame
    res_df = pd.DataFrame(all_results)
    
    # Calculate per-image statistics
    image_stats = []
    for img_name in res_df['image_name'].unique():
        img_data = res_df[res_df['image_name'] == img_name]
        best_matches = img_data[img_data['match_rank'] == 1]
        
        if len(best_matches) > 0:
            image_stats.append({
                "image_name": img_name,
                "min_dE": best_matches['dE2000'].min(),
                "max_dE": best_matches['dE2000'].max(),
                "mean_dE": best_matches['dE2000'].mean(),
                "std_dE": best_matches['dE2000'].std(),
                "median_dE": best_matches['dE2000'].median(),
                "num_colors": QUANTIZATION_LEVELS,
                "total_pixels": best_matches['color_size'].sum(),
                "interpolation_used": USE_INTERPOLATION  # Added: Record interpolation status
            })
    
    return pd.DataFrame(image_stats), res_df

def create_statistics_visualization(stats_df, output_dir):
    """Create visualization of the statistics."""
    if stats_df is None or len(stats_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ΔE distribution
    axes[0, 0].hist(stats_df['mean_dE'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Mean ΔE2000 per Image')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Mean ΔE2000 Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Min vs Max ΔE
    axes[0, 1].scatter(stats_df['min_dE'], stats_df['max_dE'], alpha=0.6)
    axes[0, 1].set_xlabel('Minimum ΔE2000')
    axes[0, 1].set_ylabel('Maximum ΔE2000')
    axes[0, 1].set_title('Minimum vs Maximum ΔE2000 per Image')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Standard Deviation
    axes[1, 0].hist(stats_df['std_dE'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Standard Deviation of ΔE2000')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of ΔE2000 Standard Deviations')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Pixel count vs ΔE
    axes[1, 1].scatter(stats_df['total_pixels'], stats_df['mean_dE'], alpha=0.6)
    axes[1, 1].set_xlabel('Total Pixels in Image')
    axes[1, 1].set_ylabel('Mean ΔE2000')
    axes[1, 1].set_title('Image Size vs Color Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    stats_plot_path = os.path.join(output_dir, "median_cut_statistics_visualization.png")
    plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Statistics visualization: {stats_plot_path}")

# ========== MAIN EXECUTION ==========
def main():
    print("Bone Color Analysis with Median Cut Quantization")
    print("================================================")
    print(f"Using interpolated Munsell space: {USE_INTERPOLATION}")
    if USE_INTERPOLATION:
        print(f"Interpolation resolution: {INTERPOLATION_RESOLUTION}")
    
    # Load ground truth
    print("Loading Munsell ground truth...")
    try:
        gt_df = load_munsell_groundtruth(MUNSELL_CSV, USE_INTERPOLATION)
        print(f"Loaded {len(gt_df)} Munsell color references")
        
        # Print column names to help debug
        print(f"Columns in ground truth: {list(gt_df.columns)}")
        
        # Check if Munsell notation is working
        print(f"Sample Munsell notations: {gt_df['munsell_notation'].head()}")
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return
    
    # Get image files
    image_paths = []
    for fname in os.listdir(BONE_IMAGES_DIR):
        if fname.lower().endswith(('.png','.jpg','.jpeg','.tiff','.bmp')):
            image_paths.append(os.path.join(BONE_IMAGES_DIR, fname))
    
    if not image_paths:
        print(f"No images found in {BONE_IMAGES_DIR}")
        return
    
    print(f"Found {len(image_paths)} bone images to process")
    
    # Process all images
    all_results = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        print(f"\nProcessing: {os.path.basename(img_path)}")
        results = process_single_image(img_path, gt_df, OUTPUT_DIR)
        all_results.extend(results)
    
    # Save comprehensive results
    if all_results:
        print(f"\nSaving results for {len(all_results)} color matches...")
        
        # Calculate statistics
        stats_df, res_df = calculate_statistics(all_results)
        
        # Save detailed results
        all_csv = os.path.join(OUTPUT_DIR, "median_cut_bone_deltae_results.csv")
        res_df.to_csv(all_csv, index=False)
        print(f"Detailed results: {all_csv}")
        
        # Save statistics
        if stats_df is not None and len(stats_df) > 0:
            stats_csv = os.path.join(OUTPUT_DIR, "median_cut_statistics_summary.csv")
            stats_df.to_csv(stats_csv, index=False)
            print(f"Statistics summary: {stats_csv}")
            
            # Create statistics visualization
            create_statistics_visualization(stats_df, OUTPUT_DIR)
            
            # Print comprehensive summary
            print(f"\n=== ANALYSIS COMPLETE ===")
            print(f"Processed: {len(image_paths)} images")
            print(f"Generated: {len(all_results)} color-match pairs")
            print(f"Best ΔE2000: {stats_df['min_dE'].min():.2f}")
            print(f"Worst ΔE2000: {stats_df['max_dE'].max():.2f}")
            print(f"Mean ΔE2000: {stats_df['mean_dE'].mean():.2f}")
            print(f"Median ΔE2000: {stats_df['median_dE'].median():.2f}")
            print(f"Average Standard Deviation: {stats_df['std_dE'].mean():.2f}")
            print(f"Interpolation used: {USE_INTERPOLATION}")
            print(f"Results saved to: {OUTPUT_DIR}")
        
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()