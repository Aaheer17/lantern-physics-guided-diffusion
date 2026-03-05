#!/usr/bin/env python3
"""
Gradient Angle Analysis for ConFIG Multi-Objective Optimization

This script analyzes the angles between different gradient vectors during
multi-objective training with ConFIG method. It computes and visualizes:
1. Angle between diffusion loss and voxel energy loss gradients
2. Angle between combined gradient and diffusion loss gradient
3. Angle between combined gradient and voxel energy loss gradient

Usage:
    python plot_gradient_angles.py <path_to_csv_file>

Example:
    python plot_gradient_angles.py ./training_logs/metrics.csv
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_angle(dot_product, norm1, norm2):
    """
    Compute angle between two vectors given their dot product and norms.
    
    Args:
        dot_product: Dot product between the two vectors
        norm1: Norm of the first vector
        norm2: Norm of the second vector
    
    Returns:
        Angle in degrees (0-180)
    """
    cos_theta = dot_product / (norm1 * norm2)
    # Clip to handle numerical errors that might push values outside [-1, 1]
    cos_theta_clipped = np.clip(cos_theta, -1, 1)
    angle_rad = np.arccos(cos_theta_clipped)
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg


def plot_gradient_angles(csv_path):
    """
    Generate gradient angle analysis plots from training log CSV.
    
    Args:
        csv_path: Path to the CSV file containing training metrics
    """
    # Convert to Path object for easier manipulation
    csv_path = Path(csv_path)
    
    # Check if file exists
    if not csv_path.exists():
        print(f"Error: File not found - {csv_path}")
        sys.exit(1)
    
    # Load the CSV data
    print(f"Loading data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Verify required columns exist
    required_columns = [
        'epoch',
        'train_config_gdot_combined__diffusion_loss_epoch',
        'train_config_gdot_combined__voxel_energy_loss_epoch',
        'train_config_gdot_diffusion_loss__voxel_energy_loss_epoch',
        'train_config_gnorm_combined_epoch',
        'train_config_gnorm_diffusion_loss_epoch',
        'train_config_gnorm_voxel_energy_loss_epoch'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        sys.exit(1)
    
    print(f"Processing {len(df)} epochs of data...")
    
    # Compute angles (in degrees)
    # 1. Angle between diffusion_loss and voxel_energy_loss
    angle_diff_voxel = compute_angle(
        df['train_config_gdot_diffusion_loss__voxel_energy_loss_epoch'],
        df['train_config_gnorm_diffusion_loss_epoch'],
        df['train_config_gnorm_voxel_energy_loss_epoch']
    )
    
    # 2. Angle between combined and diffusion_loss
    angle_comb_diff = compute_angle(
        df['train_config_gdot_combined__diffusion_loss_epoch'],
        df['train_config_gnorm_combined_epoch'],
        df['train_config_gnorm_diffusion_loss_epoch']
    )
    
    # 3. Angle between combined and voxel_energy_loss
    angle_comb_voxel = compute_angle(
        df['train_config_gdot_combined__voxel_energy_loss_epoch'],
        df['train_config_gnorm_combined_epoch'],
        df['train_config_gnorm_voxel_energy_loss_epoch']
    )
    # Create a new DataFrame with angles
    angles_df = pd.DataFrame({
        'epoch': df['epoch'],
        'angle_diffusion_voxel': angle_diff_voxel,
        'angle_combined_diffusion': angle_comb_diff,
        'angle_combined_voxel': angle_comb_voxel,
        'angle_sum_combined': angle_comb_diff + angle_comb_voxel,  # Sum of combined angles
    })

    # Optional: Add additional statistics
    angles_df['angle_deviation_from_90'] = angles_df['angle_sum_combined'] - 90.0

    angles_df.to_csv('angles.csv',index=False)

    # Create the plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Diffusion vs Voxel Energy Loss
    axes[0].plot(df['epoch'], angle_diff_voxel, linewidth=2, color='#e74c3c', 
                 label='Diff ↔ Energy')
    axes[0].axhline(y=90, color='gray', linestyle='--', alpha=0.5, 
                    label='90° (orthogonal)')
    axes[0].set_ylabel('Angle (degrees)', fontsize=11)
    axes[0].set_title('Gradient Conflict: Diffusion Loss ↔ Voxel Energy Loss', 
                      fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 180])
    
    # Plot 2: Combined vs Diffusion Loss
    axes[1].plot(df['epoch'], angle_comb_diff, linewidth=2, color='#3498db', 
                 label='Combined ↔ Diff')
    axes[1].axhline(y=90, color='gray', linestyle='--', alpha=0.5, 
                    label='90° (orthogonal)')
    axes[1].set_ylabel('Angle (degrees)', fontsize=11)
    axes[1].set_title('ConFIG Balance: Combined ↔ Diffusion Loss', 
                      fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 180])
    
    # Plot 3: Combined vs Voxel Energy Loss
    axes[2].plot(df['epoch'], angle_comb_voxel, linewidth=2, color='#2ecc71', 
                 label='Combined ↔ Energy')
    axes[2].axhline(y=90, color='gray', linestyle='--', alpha=0.5, 
                    label='90° (orthogonal)')
    axes[2].set_ylabel('Angle (degrees)', fontsize=11)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_title('ConFIG Balance: Combined ↔ Voxel Energy Loss', 
                      fontsize=12, fontweight='bold')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 180])
    
    plt.tight_layout()
    
    # Save the plot in the same directory as the CSV file
    output_dir = csv_path.parent
    output_path = output_dir / 'gradient_angles_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF for publication quality
    output_path_pdf = output_dir / 'gradient_angles_analysis.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_path_pdf}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("GRADIENT ANGLE STATISTICS")
    print("="*60)
    
    print(f"\nDiffusion ↔ Energy Loss:")
    print(f"  Mean: {angle_diff_voxel.mean():.2f}°")
    print(f"  Std:  {angle_diff_voxel.std():.2f}°")
    print(f"  Min:  {angle_diff_voxel.min():.2f}°  (Epoch {df.loc[angle_diff_voxel.idxmin(), 'epoch']:.0f})")
    print(f"  Max:  {angle_diff_voxel.max():.2f}°  (Epoch {df.loc[angle_diff_voxel.idxmax(), 'epoch']:.0f})")
    
    print(f"\nCombined ↔ Diffusion Loss:")
    print(f"  Mean: {angle_comb_diff.mean():.2f}°")
    print(f"  Std:  {angle_comb_diff.std():.2f}°")
    print(f"  Min:  {angle_comb_diff.min():.2f}°  (Epoch {df.loc[angle_comb_diff.idxmin(), 'epoch']:.0f})")
    print(f"  Max:  {angle_comb_diff.max():.2f}°  (Epoch {df.loc[angle_comb_diff.idxmax(), 'epoch']:.0f})")
    
    print(f"\nCombined ↔ Energy Loss:")
    print(f"  Mean: {angle_comb_voxel.mean():.2f}°")
    print(f"  Std:  {angle_comb_voxel.std():.2f}°")
    print(f"  Min:  {angle_comb_voxel.min():.2f}°  (Epoch {df.loc[angle_comb_voxel.idxmin(), 'epoch']:.0f})")
    print(f"  Max:  {angle_comb_voxel.max():.2f}°  (Epoch {df.loc[angle_comb_voxel.idxmax(), 'epoch']:.0f})")
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("  0°-30°:   Gradients aligned (cooperative)")
    print(" 30°-60°:   Moderately aligned")
    print(" 60°-120°:  Conflicting gradients (90° = maximum conflict)")
    print("120°-150°:  Strongly conflicting")
    print("150°-180°:  Opposite directions (severe conflict)")
    print("="*60)
    
    # Save statistics to text file
    stats_path = output_dir / 'gradient_angles_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("GRADIENT ANGLE STATISTICS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Diffusion ↔ Energy Loss:\n")
        f.write(f"  Mean: {angle_diff_voxel.mean():.2f}°\n")
        f.write(f"  Std:  {angle_diff_voxel.std():.2f}°\n")
        f.write(f"  Min:  {angle_diff_voxel.min():.2f}°  (Epoch {df.loc[angle_diff_voxel.idxmin(), 'epoch']:.0f})\n")
        f.write(f"  Max:  {angle_diff_voxel.max():.2f}°  (Epoch {df.loc[angle_diff_voxel.idxmax(), 'epoch']:.0f})\n\n")
        
        f.write(f"Combined ↔ Diffusion Loss:\n")
        f.write(f"  Mean: {angle_comb_diff.mean():.2f}°\n")
        f.write(f"  Std:  {angle_comb_diff.std():.2f}°\n")
        f.write(f"  Min:  {angle_comb_diff.min():.2f}°  (Epoch {df.loc[angle_comb_diff.idxmin(), 'epoch']:.0f})\n")
        f.write(f"  Max:  {angle_comb_diff.max():.2f}°  (Epoch {df.loc[angle_comb_diff.idxmax(), 'epoch']:.0f})\n\n")
        
        f.write(f"Combined ↔ Energy Loss:\n")
        f.write(f"  Mean: {angle_comb_voxel.mean():.2f}°\n")
        f.write(f"  Std:  {angle_comb_voxel.std():.2f}°\n")
        f.write(f"  Min:  {angle_comb_voxel.min():.2f}°  (Epoch {df.loc[angle_comb_voxel.idxmin(), 'epoch']:.0f})\n")
        f.write(f"  Max:  {angle_comb_voxel.max():.2f}°  (Epoch {df.loc[angle_comb_voxel.idxmax(), 'epoch']:.0f})\n")
    
    print(f"\nStatistics saved to: {stats_path}")


def main():
    """Main entry point for the script."""
    if len(sys.argv) != 2:
        print(__doc__)
        print("\nError: Please provide the path to the CSV file.")
        print("Usage: python plot_gradient_angles.py <path_to_csv_file>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    plot_gradient_angles(csv_path)
    print("\nDone!")


if __name__ == "__main__":
    main()