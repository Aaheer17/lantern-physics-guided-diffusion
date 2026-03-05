#!/usr/bin/env python3
"""
Plot training metrics from CSV file
Handles both weighted_sum and CONFIG multi-objective methods

Usage:
    python plot_training_metrics.py /path/to/train_val_metrics.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

from scipy.stats import linregress

def compute_loss_saturation_summary(
    df,
    last_n=300,
    compare_window=100,
    ema_span=20,
    delta_threshold=0.01,
    output_dir='',
    save_path=None
):
    """
    Computes late-training diagnostics for all train/val loss columns in df.

    Metrics computed:
        - slope_last_n
        - delta_compare (mean drop between two windows)
        - delta_window_ema (EMA-based drop)
        - noise_last_n (std of last window)
        - status (saturated or improving)

    """

    results = []

    # Find loss columns automatically
    loss_columns = [
        col for col in df.columns
        if ("train_" in col or "val_" in col) and col.endswith("_epoch")
    ]

    epochs = df["epoch"].values

    for col in loss_columns:
        values = df[col].values

        if len(values) < last_n:
            continue

        # Last N epochs
        y_last = values[-last_n:]
        x_last = epochs[-last_n:]

        # 1) Slope
        slope, _, _, _, _ = linregress(x_last, y_last)

        # 2) Window mean comparison
        window1 = values[-2*compare_window:-compare_window]
        window2 = values[-compare_window:]

        delta_compare = np.mean(window1) - np.mean(window2)

        # 3) EMA-based delta
        ema = pd.Series(values).ewm(span=ema_span).mean().values
        delta_window_ema = ema[-compare_window] - ema[-1]

        # 4) Noise estimate
        noise_last_n = np.std(y_last)

        # 5) Saturation decision
        status = "saturated" if abs(delta_window_ema) < delta_threshold else "improving"

        results.append({
            "loss_name": col,
            "slope_last_n": slope,
            "delta_compare": delta_compare,
            "delta_window_ema": delta_window_ema,
            "noise_last_n": noise_last_n,
            "status": status
        })

    summary_df = pd.DataFrame(results)
    save_path=output_dir/save_path
    if save_path is not None:
        summary_df.to_csv(save_path, index=False)

    return summary_df

def should_use_log_scale(values):
    """Determine if log scale should be used based on value range"""
    if len(values) == 0 or values.max() == 0:
        return False
    
    # Use log scale if:
    # 1. Values span more than 2 orders of magnitude
    # 2. OR max value > 100 and min value < 1
    value_range = values.max() / (values.min() + 1e-10)
    
    if value_range > 100:  # More than 2 orders of magnitude
        return True
    
    if values.max() > 100 and values.min() < 1:
        return True
    
    return False
def plot_voxel_cfd_vs_epoch(df, output_dir, col='val_voxel_cfd_epoch', use_log='auto'):
    """
    Plot voxel CFD vs epoch, ignoring NaNs (e.g., when CFD is computed every N epochs).
    Uses df['epoch'] as the x-axis so points land on the correct epoch numbers.
    """
    if col not in df.columns:
        print(f"⚠️  Skipping voxel CFD: column '{col}' not found")
        return

    df_cfd = df[['epoch', col]].dropna()
    if df_cfd.empty:
        print(f"⚠️  Skipping voxel CFD: all values are NaN in '{col}'")
        return

    # Auto-detect log scale if requested
    if use_log == 'auto':
        use_log = should_use_log_scale(df_cfd[col].values)

    fig, ax = plt.subplots(figsize=(10, 6))

    # markers only at computed epochs (+ connect with a line for readability)
    ax.plot(
        df_cfd['epoch'], df_cfd[col],
        linestyle='-', marker='o', linewidth=2, markersize=5,
        alpha=0.85, label='Val Voxel CFD'
    )

    ax.set_xlabel('Epoch', fontsize=12)
    ylabel = 'Voxel CFD'
    if use_log:
        ylabel += ' (log scale)'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Voxel CFD vs Epoch (NaNs ignored)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    if use_log:
        ax.set_yscale('log')

    # annotate last available CFD
    last_epoch = int(df_cfd['epoch'].iloc[-1])
    last_val = df_cfd[col].iloc[-1]
    ax.text(
        0.02, 0.98, f'Last: {last_val:.6f} @ epoch {last_epoch}',
        transform=ax.transAxes, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7)
    )

    plt.tight_layout()
    output_path = output_dir / 'val_voxel_cfd_vs_epoch.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved {output_path.name}{' (log scale)' if use_log else ''}")

def load_data(csv_path):
    """Load CSV file"""
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} epochs from {csv_path}")
    return df

def detect_mo_method(df):
    """Detect which multi-objective method was used"""
    config_cols = [col for col in df.columns if 'config_' in col]
    
    if config_cols:
        return 'config'
    else:
        return 'weighted_sum'

def plot_loss(df, loss_name, output_dir, use_log='auto'):
    """Plot a single loss over epochs"""
    train_col = f'train_{loss_name}_epoch'
    val_col = f'val_{loss_name}_epoch'
    
    # Check if columns exist
    has_train = train_col in df.columns
    has_val = val_col in df.columns
    
    if not has_train and not has_val:
        print(f"⚠️  Skipping {loss_name}: no data found")
        return
    
    # Auto-detect log scale
    if use_log == 'auto':
        if has_train:
            use_log = should_use_log_scale(df[train_col].dropna().values)
        elif has_val:
            use_log = should_use_log_scale(df[val_col].dropna().values)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot train
    if has_train:
        ax.plot(df['epoch'], df[train_col], 'b-', linewidth=2, label='Train', alpha=0.8)
    
    # Plot val
    if has_val:
        ax.plot(df['epoch'], df[val_col], 'r--', linewidth=2, label='Val', alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ylabel = f'{loss_name.replace("_", " ").title()}'
    if use_log:
        ylabel += ' (log scale)'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{loss_name.replace("_", " ").title()} vs Epoch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if use_log:
        ax.set_yscale('log')
    
    # Add final values as text
    if has_train:
        final_train = df[train_col].iloc[-1]
        ax.text(0.02, 0.98, f'Final Train: {final_train:.6f}', 
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    if has_val:
        final_val = df[val_col].iloc[-1]
        ax.text(0.02, 0.90, f'Final Val: {final_val:.6f}', 
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    
    output_path = output_dir / f'{loss_name}_vs_epoch.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log_str = " (log scale)" if use_log else ""
    print(f"✓ Saved {output_path.name}{log_str}")
def plot_loss_last_epochs(df, loss_name, output_dir, last_n_epochs=200, ylim=None, use_log=False):
    """Plot only the last N epochs for a given loss (zoomed view)."""
    train_col = f'train_{loss_name}_epoch'
    val_col = f'val_{loss_name}_epoch'

    has_train = train_col in df.columns
    has_val = val_col in df.columns

    if not has_train and not has_val:
        print(f"⚠️  Skipping {loss_name} (last {last_n_epochs}): no data found")
        return

    # Determine last-epoch window
    max_epoch = df['epoch'].max()
    start_epoch = max_epoch - last_n_epochs

    df_zoom = df[df['epoch'] >= start_epoch].copy()
    if df_zoom.empty:
        print(f"⚠️  No data in last {last_n_epochs} epochs for {loss_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if has_train:
        ax.plot(df_zoom['epoch'], df_zoom[train_col], 'b-', linewidth=2, label='Train', alpha=0.85)

    # Keep your validation epoch scaling convention (val logged every validate_every steps)
    if has_val:
        ax.plot(df_zoom['epoch'] , df_zoom[val_col], 'r--', linewidth=2, label='Val', alpha=0.85)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(loss_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(f'{loss_name.replace("_", " ").title()} (Last {last_n_epochs} Epochs)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if use_log:
        ax.set_yscale('log')

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    output_path = output_dir / f'{loss_name}_last_{last_n_epochs}_epochs.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved {output_path.name}")

def plot_loss_contributions(df, output_dir):
    """Plot loss contributions over time (weighted_sum method)"""
    contribution_cols = [col for col in df.columns if 'contribution' in col and 'train' in col]
    
    if not contribution_cols:
        print("⚠️  No contribution columns found (using CONFIG method)")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in contribution_cols:
        loss_type = col.replace('train_', '').replace('_contribution_epoch', '')
        ax.plot(df['epoch'], df[col], linewidth=2, label=loss_type, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Contribution to Total Loss', fontsize=12)
    ax.set_title('Loss Contributions Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'loss_contributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {output_path.name}")
    
    # Print contribution statistics
    print("\n" + "="*80)
    print("LOSS CONTRIBUTION STATISTICS (Final Epoch)")
    print("="*80)
    
    total_contribution = 0
    for col in contribution_cols:
        loss_type = col.replace('train_', '').replace('_contribution_epoch', '')
        final_contrib = df[col].iloc[-1]
        total_contribution += final_contrib
        print(f"{loss_type:30s}: {final_contrib:.6f}")
    
    print(f"{'Total':30s}: {total_contribution:.6f}")
    print()
    
    if total_contribution > 0:
        print("Percentage contributions:")
        for col in contribution_cols:
            loss_type = col.replace('train_', '').replace('_contribution_epoch', '')
            final_contrib = df[col].iloc[-1]
            pct = 100.0 * final_contrib / total_contribution
            print(f"{loss_type:30s}: {pct:6.2f}%")
    
    print("="*80 + "\n")
def plot_config_gradient_stats(df, output_dir):
    """Plot CONFIG-specific gradient statistics with separate subplots"""
    # Find CONFIG columns
    gdot_cols = [col for col in df.columns if 'config_gdot_' in col and 'train' in col]
    gnorm_cols = [col for col in df.columns if 'config_gnorm_' in col and 'combined' not in col and 'train' in col]
    
    if not gdot_cols and not gnorm_cols:
        return
    
    print("\nGenerating CONFIG gradient statistics plots...")
    
    # ========================================
    # Plot 1: Gradient angles - SEPARATE SUBPLOTS (FIXED)
    # ========================================
    if gdot_cols:
        n_pairs = len(gdot_cols)
        
        # Create subplots side by side
        fig, axes = plt.subplots(1, n_pairs, figsize=(8*n_pairs, 6))
        
        # Handle single subplot case
        if n_pairs == 1:
            axes = [axes]
        
        for idx, col in enumerate(gdot_cols):
            ax = axes[idx]
            
            # Extract objective names
            objectives = col.replace('train_config_gdot_', '').replace('_epoch', '').replace('__', ' vs ')
            
            # Parse objective pair from column name
            parts = col.replace('train_config_gdot_', '').replace('_epoch', '').split('__')
            
            if len(parts) == 2:
                obj1, obj2 = parts
                
                # Get gradient norm columns
                norm1_col = f'train_config_gnorm_{obj1}_epoch'
                norm2_col = f'train_config_gnorm_{obj2}_epoch'
                
                if norm1_col in df.columns and norm2_col in df.columns:
                    # ✓ CORRECT: Compute angle = arccos(dot / (norm1 * norm2))
                    dot_products = df[col].values
                    norm1 = df[norm1_col].values
                    norm2 = df[norm2_col].values
                    
                    # Compute cosine of angle
                    cos_theta = dot_products / (norm1 * norm2 + 1e-10)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    
                    # Convert to angle in degrees
                    angles = np.arccos(cos_theta) * 180.0 / np.pi
                    
                    # Plot
                    ax.plot(df['epoch'], angles, linewidth=2, color='blue', alpha=0.8)
                    
                    # Add reference lines
                    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='90° (orthogonal)')
                    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='0° (aligned)')
                    ax.axhline(y=180, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='180° (opposite)')
                    
                    # Labels and title
                    ax.set_xlabel('Epoch', fontsize=11)
                    ax.set_ylabel('Gradient Angle (degrees)', fontsize=11)
                    ax.set_title(f'{objectives}', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim([0, 180])
                    ax.legend(fontsize=9, loc='best')
                    
                    # Add final angle as text
                    final_angle = angles[-1]
                    ax.text(0.98, 0.02, f'Final: {final_angle:.1f}°\nDot: {df[col].iloc[-1]:.3f}\n({obj1}/{obj2})', 
                           transform=ax.transAxes, ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                           fontsize=9)
                else:
                    ax.text(0.5, 0.5, f'Missing norm data for\n{objectives}',
                           ha='center', va='center', fontsize=12)
            else:
                ax.text(0.5, 0.5, f'Could not parse\n{col}',
                       ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        output_path = output_dir / 'config_gradient_angles_separate.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {output_path.name}")
        
        # Print final angles with correct computation
        print("\n" + "="*80)
        print("GRADIENT ANGLES (Final Epoch) - CORRECTED")
        print("="*80)
        
        for col in gdot_cols:
            parts = col.replace('train_config_gdot_', '').replace('_epoch', '').split('__')
            if len(parts) == 2:
                obj1, obj2 = parts
                norm1_col = f'train_config_gnorm_{obj1}_epoch'
                norm2_col = f'train_config_gnorm_{obj2}_epoch'
                
                if norm1_col in df.columns and norm2_col in df.columns:
                    dot = df[col].iloc[-1]
                    norm1 = df[norm1_col].iloc[-1]
                    norm2 = df[norm2_col].iloc[-1]
                    
                    cos_theta = dot / (norm1 * norm2 + 1e-10)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle = np.arccos(cos_theta) * 180.0 / np.pi
                    
                    if angle < 30:
                        interp = "✓ Well aligned"
                    elif angle < 60:
                        interp = "Moderately aligned"
                    elif angle < 120:
                        interp = "⚠️  Conflicting"
                    else:
                        interp = "❌ Strongly conflicting"
                    
                    print(f"{obj1:25s} ↔ {obj2:25s}: {angle:6.2f}°  {interp}")
        
        print("="*80 + "\n")
    
    # Rest of the function remains the same...
    # (gradient norms plot, etc.)

def plot_gradient_angles_weighted_sum(df, output_dir):
    """Convert gradient dot products to angles for weighted_sum method"""
    dot_product_cols = [col for col in df.columns if 'dot_product' in col.lower() and 'train' in col]
    
    if not dot_product_cols:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in dot_product_cols:
        # Convert dot product to angle (in degrees)
        dot_products = df[col].values
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles = np.arccos(dot_products) * 180.0 / np.pi
        
        label = col.replace('train_', '').replace('_epoch', '').replace('_', ' ')
        ax.plot(df['epoch'], angles, linewidth=2, label=label, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gradient Angle (degrees)', fontsize=12)
    ax.set_title('Gradient Alignment Angles Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add reference lines
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90° (orthogonal)')
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='0° (aligned)')
    ax.axhline(y=180, color='orange', linestyle='--', alpha=0.5, label='180° (opposite)')
    
    ax.text(0.98, 0.98, 
            '0° = perfectly aligned\n90° = orthogonal\n180° = opposite', 
            transform=ax.transAxes, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
            fontsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / 'gradient_angles.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {output_path.name}")
    
    # Print final angles
    print("\n" + "="*80)
    print("GRADIENT ANGLES (Final Epoch)")
    print("="*80)
    
    for col in dot_product_cols:
        dot_product = df[col].iloc[-1]
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product) * 180.0 / np.pi
        
        label = col.replace('train_', '').replace('_epoch', '')
        
        if angle < 30:
            interp = "✓ Well aligned"
        elif angle < 60:
            interp = "Moderately aligned"
        elif angle < 120:
            interp = "⚠️  Somewhat conflicting"
        else:
            interp = "❌ Conflicting"
        
        print(f"{label:40s}: {angle:6.2f}° (dot={dot_product:7.4f})  {interp}")
    
    print("="*80 + "\n")

def plot_all_losses_combined(df, output_dir):
    """Plot all main losses on the same figure (normalized)"""
    loss_types = ['diffusion_loss', 'energy_loss','voxel_energy_loss' 'moment_loss', 'sparsity_match_loss']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['blue', 'red', 'orange', 'green', 'purple']
    
    for loss_type, color in zip(loss_types, colors):
        train_col = f'train_{loss_type}_epoch'
        if train_col in df.columns:
            # Normalize to [0, 1] for comparison
            values = df[train_col].values
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
            else:
                normalized = values
            
            ax.plot(df['epoch'], normalized, color=color, linewidth=2, 
                   label=f'{loss_type.replace("_", " ")}', alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Normalized Loss [0, 1]', fontsize=12)
    ax.set_title('All Losses (Normalized) vs Epoch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'all_losses_normalized.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {output_path.name}")

def main(csv_path):
    """Main function"""
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} does not exist!")
        sys.exit(1)
    
    output_dir = csv_path.parent
    
    print(f"\n{'='*80}")
    print(f"PLOTTING TRAINING METRICS")
    print(f"{'='*80}\n")
    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}\n")
    
    # Load data
    df = load_data(csv_path)
    
    # Detect MO method
    mo_method = detect_mo_method(df)
    print(f"Detected multi-objective method: {mo_method.upper()}\n")
    
    # Plot individual losses with auto log scale detection
    print("Generating individual loss plots...")
    plot_loss(df, 'diffusion_loss', output_dir, use_log='auto')
    plot_loss(df, 'energy_loss', output_dir, use_log='auto')
    plot_loss(df, 'moment_loss', output_dir, use_log='auto')
    plot_loss(df, 'sparsity_match_loss', output_dir, use_log='auto')
    plot_loss(df, 'total_loss', output_dir, use_log='auto')
    plot_loss(df, 'voxel_energy_loss', output_dir, use_log='auto')
    # Plot method-specific information
    plot_loss(df, 'voxel_energy_loss_unweighted', output_dir, use_log='auto')
    # Plot voxel CFD (computed intermittently; NaNs are ignored)
    plot_voxel_cfd_vs_epoch(df, output_dir, col='val_voxel_cfd_epoch', use_log='auto')

    summary = compute_loss_saturation_summary(
        df,
        last_n=300,
        compare_window=100,
        ema_span=20,
        delta_threshold=0.01,
        output_dir=output_dir,
        save_path="loss_saturation_summary.csv"
    )
    
    print(summary)
    # Plot voxel energy weight (warmup schedule) if available
    if 'train_voxel_energy_weight_epoch' in df.columns:
        print("Generating voxel energy weight plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df['epoch'], df['train_voxel_energy_weight_epoch'], 
                linewidth=2, color='darkorange', alpha=0.8)
        
        # Add horizontal line at max weight
        max_weight = df['train_voxel_energy_weight_epoch'].max()
        ax.axhline(y=max_weight, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.5, label=f'Target: {max_weight:.6f}')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Voxel Energy Loss Weight', fontsize=12)
        ax.set_title('Voxel Energy Loss Weight (Warmup Schedule)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Annotate warmup completion
        if max_weight > 0:
            warmup_done = df[df['train_voxel_energy_weight_epoch'] >= 0.99 * max_weight]
            if len(warmup_done) > 0:
                first_full = warmup_done.iloc[0]['epoch']
                ax.axvline(x=first_full, color='green', linestyle=':', 
                          linewidth=1.5, alpha=0.5)
                ax.text(first_full, max_weight * 0.5, 
                       f'Warmup Complete\n(Epoch {int(first_full)})',
                       ha='center', va='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        output_path = output_dir / 'voxel_energy_weight_vs_epoch.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {output_path.name}")
    if mo_method == 'weighted_sum':
        print("\nGenerating loss contribution plot...")
        plot_loss_contributions(df, output_dir)
        
        print("\nGenerating gradient angle plots...")
        plot_gradient_angles_weighted_sum(df, output_dir)
    
    elif mo_method == 'config':
        # CONFIG-specific plots
        plot_config_gradient_stats(df, output_dir)
    
    # Plot all losses combined
    print("\nGenerating combined losses plot...")
    plot_all_losses_combined(df, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"{'='*80}\n")
    print(f"Output location: {output_dir}")
    print(f"\nGenerated files:")
    for png_file in sorted(output_dir.glob('*.png')):
        print(f"  - {png_file.name}")
    print()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python plot_training_metrics.py /path/to/train_val_metrics.csv")
        sys.exit(1)
    
    main(sys.argv[1])