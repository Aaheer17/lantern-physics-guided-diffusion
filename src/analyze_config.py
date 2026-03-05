#!/usr/bin/env python3
"""
Analyze CONFIG training WITHOUT the ConFIG vs objectives dot products
Extract maximum information from available metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_config_behavior(csv_path):
    """Comprehensive analysis of CONFIG training"""
    
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    output_dir = csv_path.parent
    
    print("\n" + "="*80)
    print("CONFIG BEHAVIOR ANALYSIS (From Available Data)")
    print("="*80)
    
    # ========================================
    # 1. GRADIENT MAGNITUDE ANALYSIS
    # ========================================
    print("\n1. GRADIENT MAGNITUDE COMPARISON")
    print("-"*80)
    
    gnorm_diff = df['train_config_gnorm_diffusion_loss_epoch']
    gnorm_energy = df['train_config_gnorm_energy_loss_epoch']
    gnorm_combined = df['train_config_gnorm_combined_epoch']
    
    print(f"\nFirst epoch:")
    print(f"  Diffusion grad norm:  {gnorm_diff.iloc[0]:.6f}")
    print(f"  Energy grad norm:     {gnorm_energy.iloc[0]:.6f}")
    print(f"  Combined grad norm:   {gnorm_combined.iloc[0]:.6f}")
    print(f"  Ratio (diff/energy):  {gnorm_diff.iloc[0]/gnorm_energy.iloc[0]:.1f}x")
    
    print(f"\nFinal epoch:")
    print(f"  Diffusion grad norm:  {gnorm_diff.iloc[-1]:.6f}")
    print(f"  Energy grad norm:     {gnorm_energy.iloc[-1]:.6f}")
    print(f"  Combined grad norm:   {gnorm_combined.iloc[-1]:.6f}")
    print(f"  Ratio (diff/energy):  {gnorm_diff.iloc[-1]/gnorm_energy.iloc[-1]:.1f}x")
    
    # Check if energy is too weak
    avg_ratio = (gnorm_diff / gnorm_energy).mean()
    print(f"\nAverage ratio: {avg_ratio:.1f}x")
    if avg_ratio > 50:
        print("⚠️  WARNING: Energy gradients are 50x+ weaker than diffusion!")
        print("   ConFIG may be dominated by diffusion loss.")
    elif avg_ratio > 20:
        print("⚠️  Energy gradients are 20x+ weaker - significant imbalance")
    elif avg_ratio < 5:
        print("✓ Gradients are relatively balanced")
    
    # ========================================
    # 2. GRADIENT MAGNITUDE TRENDS
    # ========================================
    print("\n2. GRADIENT MAGNITUDE TRENDS")
    print("-"*80)
    
    diff_change = ((gnorm_diff.iloc[-1] - gnorm_diff.iloc[0]) / gnorm_diff.iloc[0]) * 100
    energy_change = ((gnorm_energy.iloc[-1] - gnorm_energy.iloc[0]) / gnorm_energy.iloc[0]) * 100
    
    print(f"Diffusion grad norm change: {diff_change:+.1f}%")
    print(f"Energy grad norm change:    {energy_change:+.1f}%")
    
    if energy_change > 100:
        print("✓ Energy gradients growing - good sign!")
    elif energy_change < -50:
        print("⚠️  Energy gradients shrinking - may be getting ignored")
    
    # ========================================
    # 3. COMBINED GRADIENT ANALYSIS
    # ========================================
    print("\n3. COMBINED GRADIENT COMPOSITION")
    print("-"*80)
    
    # If ConFIG perfectly balanced, combined norm would be between individual norms
    # If dominated by one, combined norm close to that one
    
    for epoch_idx, label in [(0, "First"), (-1, "Last")]:
        combined = gnorm_combined.iloc[epoch_idx]
        diff = gnorm_diff.iloc[epoch_idx]
        energy = gnorm_energy.iloc[epoch_idx]
        
        print(f"\n{label} epoch:")
        print(f"  Combined: {combined:.4f}")
        print(f"  Diffusion: {diff:.4f}")
        print(f"  Energy: {energy:.4f}")
        
        # Check which it's closer to
        dist_to_diff = abs(combined - diff)
        dist_to_energy = abs(combined - energy)
        
        if dist_to_diff < dist_to_energy:
            closeness = (1 - dist_to_diff/diff) * 100
            print(f"  → Closer to diffusion ({closeness:.1f}% similar)")
            if closeness > 90:
                print(f"  ⚠️  Almost identical to diffusion - energy may be ignored!")
        else:
            closeness = (1 - dist_to_energy/energy) * 100
            print(f"  → Closer to energy ({closeness:.1f}% similar)")
    
    # ========================================
    # 4. ESTIMATED CONTRIBUTION WEIGHTS
    # ========================================
    print("\n4. ESTIMATED OBJECTIVE INFLUENCE (Rough Approximation)")
    print("-"*80)
    print("Assuming ConFIG weights objectives by their gradient magnitude...")
    
    total_norm = gnorm_diff + gnorm_energy
    diff_weight_est = (gnorm_diff / total_norm * 100).mean()
    energy_weight_est = (gnorm_energy / total_norm * 100).mean()
    
    print(f"  Estimated diffusion influence: ~{diff_weight_est:.1f}%")
    print(f"  Estimated energy influence:    ~{energy_weight_est:.1f}%")
    
    if energy_weight_est < 10:
        print("  ⚠️  Energy has <10% influence - likely being ignored!")
    elif energy_weight_est < 25:
        print("  Energy is contributing but weakly")
    else:
        print("  ✓ Energy has reasonable influence")
    
    # ========================================
    # 5. ANGLE BETWEEN OBJECTIVES
    # ========================================
    print("\n5. ANGLE BETWEEN DIFFUSION AND ENERGY GRADIENTS")
    print("-"*80)
    
    gdot_col = 'train_config_gdot_diffusion_loss__energy_loss_epoch'
    
    if gdot_col in df.columns:
        gdot = df[gdot_col]
        
        # Compute angle
        cos_angle = gdot / (gnorm_diff * gnorm_energy + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        angles = np.arccos(cos_angle) * 180 / np.pi
        
        print(f"First epoch angle: {angles.iloc[0]:.2f}°")
        print(f"Last epoch angle:  {angles.iloc[-1]:.2f}°")
        print(f"Mean angle:        {angles.mean():.2f}°")
        print(f"Angle std:         {angles.std():.2f}°")
        
        if angles.std() < 1.0:
            print("⚠️  Angle barely changes - objectives not interacting!")
        
        mean_angle = angles.mean()
        if mean_angle < 30:
            print("✓ Objectives are aligned (cooperating)")
        elif mean_angle < 60:
            print("Objectives moderately aligned")
        elif mean_angle < 120:
            print("⚠️  Objectives are somewhat conflicting")
        else:
            print("❌ Objectives are strongly conflicting")
        
        # Check if stuck at 90°
        if abs(mean_angle - 90) < 5:
            print("\n⚠️  STUCK AT ~90° (ORTHOGONAL)!")
            print("   Possible reasons:")
            print("   1. Gradients are truly independent")
            print("   2. Energy gradients too small (numerical issue)")
            print("   3. Energy loss not actually affecting parameters")
    
    # ========================================
    # 6. LOSS VALUES CHECK
    # ========================================
    print("\n6. LOSS VALUES ANALYSIS")
    print("-"*80)
    
    diff_loss = df['train_diffusion_loss_epoch']
    energy_loss = df['train_energy_loss_epoch']
    
    print(f"\nDiffusion loss:")
    print(f"  Start: {diff_loss.iloc[0]:.6f}")
    print(f"  End:   {diff_loss.iloc[-1]:.6f}")
    print(f"  Change: {((diff_loss.iloc[-1] - diff_loss.iloc[0])/diff_loss.iloc[0]*100):+.1f}%")
    
    print(f"\nEnergy loss:")
    print(f"  Start: {energy_loss.iloc[0]:.6f}")
    print(f"  End:   {energy_loss.iloc[-1]:.6f}")
    print(f"  Change: {((energy_loss.iloc[-1] - energy_loss.iloc[0])/energy_loss.iloc[0]*100):+.1f}%")
    
    # Check if energy loss is improving
    energy_improving = energy_loss.iloc[-1] < energy_loss.iloc[0]
    diff_improving = diff_loss.iloc[-1] < diff_loss.iloc[0]
    
    if diff_improving and energy_improving:
        print("\n✓ Both losses improving!")
    elif diff_improving and not energy_improving:
        print("\n⚠️  Diffusion improving but energy getting worse!")
        print("   ConFIG may be prioritizing diffusion too heavily")
    elif not diff_improving and energy_improving:
        print("\n⚠️  Energy improving but diffusion getting worse!")
        print("   Unusual - check your setup")
    else:
        print("\n❌ Neither loss improving!")
    
    # ========================================
    # 7. VISUALIZATION
    # ========================================
    print("\n7. GENERATING VISUALIZATION...")
    print("-"*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Gradient norms over time
    ax = axes[0, 0]
    ax.plot(df['epoch'], gnorm_diff, label='Diffusion', linewidth=2)
    ax.plot(df['epoch'], gnorm_energy, label='Energy', linewidth=2)
    ax.plot(df['epoch'], gnorm_combined, label='Combined', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Magnitudes Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Gradient ratio over time
    ax = axes[0, 1]
    ratio = gnorm_diff / (gnorm_energy + 1e-10)
    ax.plot(df['epoch'], ratio, linewidth=2, color='purple')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10x threshold')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='1:1 balance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Ratio (Diffusion / Energy)')
    ax.set_title('Gradient Magnitude Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Estimated weights over time
    ax = axes[1, 0]
    total = gnorm_diff + gnorm_energy
    diff_pct = gnorm_diff / total * 100
    energy_pct = gnorm_energy / total * 100
    ax.plot(df['epoch'], diff_pct, label='Diffusion', linewidth=2)
    ax.plot(df['epoch'], energy_pct, label='Energy', linewidth=2)
    ax.axhline(y=50, color='black', linestyle='--', alpha=0.3, label='50% (balanced)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Estimated Influence (%)')
    ax.set_title('Estimated Objective Influence on ConFIG')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Plot 4: Loss values
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(df['epoch'], diff_loss, 'b-', label='Diffusion Loss', linewidth=2)
    ax2.plot(df['epoch'], energy_loss, 'r-', label='Energy Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Diffusion Loss', color='b')
    ax2.set_ylabel('Energy Loss', color='r')
    ax.set_title('Loss Values Over Time')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'config_analysis_from_available_data.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {output_path}")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    issues = []
    
    if avg_ratio > 50:
        issues.append("Energy gradients are 50x+ weaker - severe imbalance")
    
    if energy_weight_est < 10:
        issues.append(f"Energy has only ~{energy_weight_est:.1f}% influence")
    
    if not energy_improving:
        issues.append("Energy loss not improving")
    
    if angles.std() < 1.0:
        issues.append("Objective angles stuck (not interacting)")
    
    if issues:
        print("\n⚠️  POTENTIAL ISSUES DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   - Increase lambda_energy to strengthen energy gradients")
        print("   - Check that energy loss is properly connected to model")
        print("   - Verify .detach() is not breaking energy gradient flow")
        print("   - Consider grad_norm method instead of CONFIG if imbalance persists")
    else:
        print("\n✓ CONFIG appears to be working reasonably well!")
    
    print("="*80 + "\n")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_config.py /path/to/train_val_metrics.csv")
        sys.exit(1)
    
    analyze_config_behavior(sys.argv[1])