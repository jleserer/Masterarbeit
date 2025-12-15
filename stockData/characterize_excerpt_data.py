import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, Rbf
from mpl_toolkits.mplot3d import Axes3D


def find_column(columns, name):
    lower = {c.lower(): c for c in columns}
    return lower.get(name.lower())


def characterize_file(filepath, params, plots_dir):
    df = pd.read_csv(filepath)
    cols = []
    for p in params:
        col = find_column(df.columns, p)
        if col is None:
            raise ValueError(f"Column '{p}' not found in {os.path.basename(filepath)}")
        cols.append(col)

    # Extract numeric data for the three parameters
    data = df[cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(data) == 0:
        raise ValueError(f"No valid data found in {os.path.basename(filepath)}")
    
    # Assign parameters to axes: Close (X), High (Y) -> Low (Z)
    x_vals = data[cols[0]].values  # Close
    y_vals = data[cols[1]].values  # High
    z_vals = data[cols[2]].values  # Low
    
    # Create grid for interpolation
    grid_x = np.linspace(x_vals.min(), x_vals.max(), 100)
    grid_y = np.linspace(y_vals.min(), y_vals.max(), 100)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    
    # Interpolate Z values on the grid - use RBF for robustness
    try:
        rbf = Rbf(x_vals, y_vals, z_vals, function='thin_plate', smooth=0.1)
        grid_Z = rbf(grid_X, grid_Y)
    except Exception:
        # Fallback to nearest neighbor if RBF fails
        try:
            grid_Z = griddata((x_vals, y_vals), z_vals, (grid_X, grid_Y), method='nearest')
        except Exception:
            # Last resort: create flat surface with mean value
            grid_Z = np.full_like(grid_X, z_vals.mean(), dtype=float)
    
    # Plot surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_X, grid_Y, grid_Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_xlabel(cols[0])  # Close
    ax.set_ylabel(cols[1])  # High
    ax.set_zlabel(cols[2])  # Low
    ax.set_title(f"3D Surface: {cols[0]}, {cols[1]}, {cols[2]}\n{os.path.basename(filepath)}")
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    plt.tight_layout()

    basename = os.path.splitext(os.path.basename(filepath))[0]
    outpath = os.path.join(plots_dir, f"{basename}_3d.png")
    plt.savefig(outpath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Characterize CSV files in excerptData and create 3D surface plots')
    parser.add_argument('--dir', default=os.path.join(os.path.dirname(__file__), 'normalizedData'), help='Path to normalizedData folder')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'plots'), help='Output directory for plots')
    parser.add_argument('--params', nargs='+', default=['Close', 'High', 'Low'], help='Columns to use (default: Close High Low)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith('.csv')]
    all_stats = []
    for f in sorted(files):
        try:
            stats = characterize_file(f, args.params, args.out)
            all_stats.extend(stats)
            print(f"Processed: {os.path.basename(f)} -> plot saved")
        except Exception as e:
            print(f"Skipping {os.path.basename(f)}: {e}")

if __name__ == '__main__':
    main()
