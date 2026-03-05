#!/usr/bin/env python3
"""
Plot diffusion loss and voxel energy loss side by side with dynamic y-axis ranges
Focuses on the last N epochs for better zoom (TRAINING ONLY)

Usage:
    python plot_two_losses.py /path/to/train_val_metrics.csv
    python plot_two_losses.py /path/to/train_val_metrics.csv --last-epochs 200
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import argparse
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

def compute_ylim_with_padding(values, padding_pct=0.1):
    """
    Compute y-axis limits with padding
    
    Args:
        values: Array of values to compute limits from
        padding_pct: Percentage padding (0.1 = 10% padding on each side)
    
    Returns:
        (ymin, ymax) tuple
    """
    if len(values) == 0:
        return (0, 1)
    
    vmin = np.min(values)
    vmax = np.max(values)
    
    # Add padding
    value_range = vmax - vmin
    padding = value_range * padding_pct
    
    ymin = vmin - padding
    ymax = vmax + padding
    
    # Ensure non-negative for losses
    ymin = max(0, ymin)
    
    return (ymin, ymax)

def plot_diffusion_and_voxel(csv_path, output_path=None, last_n_epochs=200):
    """
    Plot diffusion loss and voxel energy loss side by side (TRAINING ONLY)
    
    Args:
        csv_path: Path to CSV file with training metrics
        output_path: Optional output path for PNG (default: same dir as CSV)
        last_n_epochs: Number of last epochs to focus on (default: 200)
    """
    # Load data
    df_full = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df_full)} epochs from {csv_path}")
    
    # Focus on last N epochs
    if last_n_epochs is not None and last_n_epochs < len(df_full):
        df = df_full.iloc[-last_n_epochs:].copy()
        epoch_start = df['epoch'].iloc[0]
        epoch_end = df['epoch'].iloc[-1]
        print(f"✓ Focusing on last {last_n_epochs} epochs (epoch {epoch_start:.0f} to {epoch_end:.0f})")
    else:
        df = df_full.copy()
        epoch_start = df['epoch'].iloc[0]
        epoch_end = df['epoch'].iloc[-1]
        actual_n = len(df)
        if last_n_epochs and last_n_epochs > actual_n:
            print(f"⚠️  Requested last {last_n_epochs} epochs, but only {actual_n} available")
        print(f"✓ Using all {len(df)} epochs (epoch {epoch_start:.0f} to {epoch_end:.0f})")
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========================================
    # LEFT PLOT: Diffusion Loss (TRAIN ONLY)
    # ========================================
    ax = axes[0]
    
    # Check for train column
    has_train_diff = 'train_diffusion_loss_epoch' in df.columns
    
    if has_train_diff:
        train_diff = df['train_diffusion_loss_epoch'].dropna()
        
        ax.plot(df['epoch'], df['train_diffusion_loss_epoch'], 
                'b-', linewidth=2.5, label='Train', alpha=0.8)
        final_train = df['train_diffusion_loss_epoch'].iloc[-1]
        initial_train = df['train_diffusion_loss_epoch'].iloc[0]
        
        # Compute dynamic y-limits
        ymin, ymax = compute_ylim_with_padding(train_diff.values, padding_pct=0.15)
        ax.set_ylim(ymin, ymax)
        print(f"  Diffusion loss y-range: [{ymin:.6f}, {ymax:.6f}]")
        
        # Add final value text
        ax.text(0.02, 0.98, 
                f'Initial: {initial_train:.6f}\nFinal: {final_train:.6f}\nChange: {final_train-initial_train:+.6f}', 
                transform=ax.transAxes, va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', 
                         edgecolor='blue', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No training diffusion loss data found', 
                ha='center', va='center', fontsize=12)
    
    # Set x-axis limits
    ax.set_xlim(epoch_start, epoch_end)
    
    # Styling
    ax.set_xlabel(f'Epoch (Showing {epoch_start:.0f}–{epoch_end:.0f})', fontsize=13, fontweight='bold')
    ax.set_ylabel('Diffusion Loss', fontsize=13, fontweight='bold')
    ax.set_title(f'Diffusion Loss (Epochs {epoch_start:.0f}–{epoch_end:.0f})', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.4, linewidth=0.8)
    
    # ========================================
    # RIGHT PLOT: Voxel Energy Loss (TRAIN ONLY, UNWEIGHTED)
    # ========================================
    ax = axes[1]
    
    # Check for train column
    has_train_vox = 'train_voxel_energy_loss_unweighted_epoch' in df.columns
    
    if has_train_vox:
        train_vox = df['train_voxel_energy_loss_unweighted_epoch'].dropna()
        
        ax.plot(df['epoch'], df['train_voxel_energy_loss_unweighted_epoch'], 
                'b-', linewidth=2.5, label='Train', alpha=0.8)
        final_train = df['train_voxel_energy_loss_unweighted_epoch'].iloc[-1]
        initial_train = df['train_voxel_energy_loss_unweighted_epoch'].iloc[0]
        
        # Compute dynamic y-limits
        ymin, ymax = compute_ylim_with_padding(train_vox.values, padding_pct=0.15)
        ax.set_ylim(ymin, ymax)
        print(f"  Voxel energy loss y-range: [{ymin:.2f}, {ymax:.2f}]")
        
        # Add final value text
        ax.text(0.02, 0.98, 
                f'Initial: {initial_train:.2f}\nFinal: {final_train:.2f}\nChange: {final_train-initial_train:+.2f}', 
                transform=ax.transAxes, va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', 
                         edgecolor='blue', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No training voxel energy loss data found', 
                ha='center', va='center', fontsize=12)
    
    # Set x-axis limits
    ax.set_xlim(epoch_start, epoch_end)
    
    # Styling
    ax.set_xlabel(f'Epoch (Showing {epoch_start:.0f}–{epoch_end:.0f})', fontsize=13, fontweight='bold')
    ax.set_ylabel('Voxel Energy Loss (Unweighted)', fontsize=13, fontweight='bold')
    ax.set_title(f'Voxel Energy Loss (Unweighted) (Epochs {epoch_start:.0f}–{epoch_end:.0f})', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.4, linewidth=0.8)
    
    # ========================================
    # Save figure
    # ========================================
    plt.tight_layout()
    
    if output_path is None:
        csv_path = Path(csv_path)
        suffix = f'_last{last_n_epochs}' if last_n_epochs else ''
        output_path = csv_path.parent / f'diffusion_and_voxel_losses_train_only{suffix}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print(f"TRAINING LOSS SUMMARY (Epochs {epoch_start:.0f}–{epoch_end:.0f})")
    print("="*80)
    
    if has_train_diff:
        final_train = df['train_diffusion_loss_epoch'].iloc[-1]
        initial_train = df['train_diffusion_loss_epoch'].iloc[0]
        pct_change = 100*(final_train-initial_train)/initial_train
        print(f"\nDiffusion Loss (Train):")
        print(f"  Start (epoch {epoch_start:.0f}): {initial_train:.6f}")
        print(f"  Final (epoch {epoch_end:.0f}): {final_train:.6f}")
        print(f"  Change: {final_train - initial_train:+.6f} ({pct_change:+.2f}%)")
    
    if has_train_vox:
        final_train = df['train_voxel_energy_loss_unweighted_epoch'].iloc[-1]
        initial_train = df['train_voxel_energy_loss_unweighted_epoch'].iloc[0]
        pct_change = 100*(final_train-initial_train)/initial_train
        print(f"\nVoxel Energy Loss Unweighted (Train):")
        print(f"  Start (epoch {epoch_start:.0f}): {initial_train:.2f}")
        print(f"  Final (epoch {epoch_end:.0f}): {final_train:.2f}")
        print(f"  Change: {final_train - initial_train:+.2f} ({pct_change:+.2f}%)")
    
    print("="*80 + "\n")
    
    return fig


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Plot diffusion and voxel energy losses (TRAINING ONLY)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Focus on last 200 epochs (default)
  python plot_two_losses.py results/train_val_metrics.csv
  
  # Focus on last 300 epochs
  python plot_two_losses.py results/train_val_metrics.csv --last-epochs 300
  
  # Use all epochs
  python plot_two_losses.py results/train_val_metrics.csv --last-epochs -1
        """
    )
    
    parser.add_argument('csv_path', type=str, help='Path to train_val_metrics.csv')
    parser.add_argument('--last-epochs', type=int, default=200, 
                        help='Number of last epochs to focus on (default: 200, use -1 for all epochs)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path (default: auto-generated in same directory as CSV)')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} does not exist!")
        sys.exit(1)
    
    # Handle "use all epochs" case
    last_n = args.last_epochs if args.last_epochs > 0 else None
    
    print(f"\n{'='*80}")
    print(f"PLOTTING DIFFUSION & VOXEL ENERGY LOSSES (TRAINING ONLY)")
    print(f"{'='*80}\n")
    print(f"Input CSV: {csv_path}")
    if last_n:
        print(f"Focus: Last {last_n} epochs")
    else:
        print(f"Focus: All epochs")
    print()
    
    # Create plot
    plot_diffusion_and_voxel(csv_path, output_path=args.output, last_n_epochs=last_n)
    
    print(f"{'='*80}")
    print("✅ PLOT GENERATED SUCCESSFULLY!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()