"""
Plot Generator for Stock Data
==============================
Generates three types of plots for each CSV file in preprocessedData:
1. Close Price Histogram
2. High-Low Difference Histogram
3. 3D Surface Plot (Close, High, Low)

Output directories:
- plots/histograms_close/
- plots/histograms_high_low_diff/
- plots/3d_surfaces/
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, griddata
from mpl_toolkits.mplot3d import Axes3D


def find_column(columns, name):
    """Find column name case-insensitively."""
    lower = {c.lower(): c for c in columns}
    return lower.get(name.lower())


def load_data(filepath):
    """Load CSV and extract Close, High, Low columns."""
    df = pd.read_csv(filepath)

    close_col = find_column(df.columns, 'Close')
    high_col = find_column(df.columns, 'High')
    low_col = find_column(df.columns, 'Low')

    if not all([close_col, high_col, low_col]):
        raise ValueError(f"Required columns (Close, High, Low) not found in {os.path.basename(filepath)}")

    data = df[[close_col, high_col, low_col]].apply(pd.to_numeric, errors='coerce').dropna()

    if len(data) == 0:
        raise ValueError(f"No valid data found in {os.path.basename(filepath)}")

    return {
        'close': data[close_col].values,
        'high': data[high_col].values,
        'low': data[low_col].values,
        'columns': [close_col, high_col, low_col]
    }


def create_close_histogram(data, basename, output_dir):
    """Create histogram for Close price."""
    close_vals = data['close']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(close_vals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Close Price', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Close Price Distribution\n{basename}', fontsize=14)
    ax.axvline(close_vals.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {close_vals.mean():.2f}')
    ax.axvline(np.median(close_vals), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(close_vals):.2f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f'Std: {close_vals.std():.2f}\nMin: {close_vals.min():.2f}\nMax: {close_vals.max():.2f}'
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    outpath = os.path.join(output_dir, f"{basename}_close_histogram.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

    return outpath


def create_high_low_histogram(data, basename, output_dir):
    """Create histogram for High-Low difference."""
    high_low_diff = data['high'] - data['low']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(high_low_diff, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('High - Low Difference', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Daily Price Range (High - Low)\n{basename}', fontsize=14)
    ax.axvline(high_low_diff.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {high_low_diff.mean():.2f}')
    ax.axvline(np.median(high_low_diff), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(high_low_diff):.2f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f'Std: {high_low_diff.std():.2f}\nMin: {high_low_diff.min():.2f}\nMax: {high_low_diff.max():.2f}'
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    outpath = os.path.join(output_dir, f"{basename}_high_low_diff_histogram.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

    return outpath


def create_3d_surface(data, basename, output_dir):
    """Create 3D surface plot for Close, High, Low."""
    close_vals = data['close']
    high_vals = data['high']
    low_vals = data['low']
    cols = data['columns']

    # Create grid for interpolation
    grid_x = np.linspace(close_vals.min(), close_vals.max(), 100)
    grid_y = np.linspace(high_vals.min(), high_vals.max(), 100)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

    # Interpolate Z values on the grid
    try:
        rbf = Rbf(close_vals, high_vals, low_vals, function='thin_plate', smooth=0.1)
        grid_Z = rbf(grid_X, grid_Y)
    except Exception:
        try:
            grid_Z = griddata((close_vals, high_vals), low_vals, (grid_X, grid_Y), method='nearest')
        except Exception:
            grid_Z = np.full_like(grid_X, low_vals.mean(), dtype=float)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(grid_X, grid_Y, grid_Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_xlabel(cols[0], fontsize=11)  # Close
    ax.set_ylabel(cols[1], fontsize=11)  # High
    ax.set_zlabel(cols[2], fontsize=11)  # Low
    ax.set_title(f"3D Surface: {cols[0]}, {cols[1]}, {cols[2]}\n{basename}", fontsize=14)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=cols[2])

    plt.tight_layout()

    outpath = os.path.join(output_dir, f"{basename}_3d_surface.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

    return outpath


def process_file(filepath, output_dirs):
    """Process a single CSV file and generate all plots."""
    basename = os.path.splitext(os.path.basename(filepath))[0]

    # Load data
    data = load_data(filepath)

    # Generate plots
    close_path = create_close_histogram(data, basename, output_dirs['close'])
    high_low_path = create_high_low_histogram(data, basename, output_dirs['high_low'])
    surface_path = create_3d_surface(data, basename, output_dirs['3d'])

    # Calculate and return statistics
    close_vals = data['close']
    high_low_diff = data['high'] - data['low']

    stats = {
        'close_mean': close_vals.mean(),
        'close_std': close_vals.std(),
        'high_low_mean': high_low_diff.mean(),
        'high_low_std': high_low_diff.std()
    }

    return {
        'close_histogram': close_path,
        'high_low_histogram': high_low_path,
        '3d_surface': surface_path,
        'stats': stats
    }


def main():
    parser = argparse.ArgumentParser(description='Generate plots for preprocessed stock data')
    parser.add_argument('--dir', default=os.path.join(os.path.dirname(__file__), 'preprocessedData'),
                        help='Path to preprocessedData folder')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'plots'),
                        help='Base output directory for plots')
    args = parser.parse_args()

    # Create output directories
    output_dirs = {
        'close': os.path.join(args.out, 'histograms_close'),
        'high_low': os.path.join(args.out, 'histograms_high_low_diff'),
        '3d': os.path.join(args.out, '3d_surfaces')
    }

    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Find all CSV files
    files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith('.csv')]

    print("=" * 70)
    print("STOCK DATA PLOT GENERATOR")
    print("=" * 70)
    print(f"\nInput directory:  {args.dir}")
    print(f"Output directory: {args.out}")
    print(f"  - Close histograms:      {output_dirs['close']}")
    print(f"  - High-Low histograms:   {output_dirs['high_low']}")
    print(f"  - 3D surfaces:           {output_dirs['3d']}")
    print(f"\nFound {len(files)} CSV files to process.\n")
    print("-" * 70)

    for f in sorted(files):
        basename = os.path.splitext(os.path.basename(f))[0]
        try:
            result = process_file(f, output_dirs)
            stats = result['stats']
            print(f"\n{basename}:")
            print(f"  Close Price    - Mean: {stats['close_mean']:>10.2f}, Std: {stats['close_std']:>10.2f}")
            print(f"  High-Low Diff  - Mean: {stats['high_low_mean']:>10.2f}, Std: {stats['high_low_std']:>10.2f}")
            print(f"  -> Close histogram:    {os.path.basename(result['close_histogram'])}")
            print(f"  -> High-Low histogram: {os.path.basename(result['high_low_histogram'])}")
            print(f"  -> 3D surface:         {os.path.basename(result['3d_surface'])}")
        except Exception as e:
            print(f"\nSkipping {basename}: {e}")

    print("\n" + "=" * 70)
    print("DONE! All plots generated successfully.")
    print("=" * 70)


if __name__ == '__main__':
    main()
