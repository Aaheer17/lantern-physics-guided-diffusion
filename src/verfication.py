import torch
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transforms import (
    ScaleVoxels, NormalizeByElayer, CutValues, SelectiveUniformNoise,
    ExclusiveLogitTransform, StandardizeFromFile, LogEnergy, ScaleEnergy,
    AddFeaturesToCond, Reshape
)

def load_dataset(filepath, n_samples=25_000):
    """Load a small batch from the dataset"""
    print(f"\n{'='*80}")
    print(f"Loading data from: {filepath}")
    print(f"{'='*80}\n")
    
    with h5py.File(filepath, 'r') as f:
        # Load incident energies
        incident_energies = torch.from_numpy(f['incident_energies'][:n_samples]).float()
        
        # Load showers (voxels)
        showers = torch.from_numpy(f['showers'][:n_samples]).float()
        
        print(f"Original Data:")
        print(f"  Showers shape: {showers.shape}")
        print(f"  Showers range: [{showers.min():.6f}, {showers.max():.6f}]")
        print(f"  Showers mean: {showers.mean():.6f}")
        print(f"  Energies shape: {incident_energies.shape}")
        print(f"  Energies range: [{incident_energies.min():.6f}, {incident_energies.max():.6f}]")
        print(f"  Energies mean: {incident_energies.mean():.6f}")
        
        # Check for zeros
        n_zero_voxels = (showers < 1e-10).sum().item()
        pct_zero = 100.0 * n_zero_voxels / showers.numel()
        print(f"  Zero voxels: {n_zero_voxels} ({pct_zero:.2f}%)")
        
        return showers, incident_energies

def create_transform_pipeline(model_dir="./test_stats"):
    """Create the preprocessing pipeline exactly as in config"""
    print(f"\n{'='*80}")
    print("Creating Transform Pipeline")
    print(f"{'='*80}\n")
    
    os.makedirs(model_dir, exist_ok=True)
    
    transforms = [
        ("ScaleVoxels", ScaleVoxels(factor=0.35)),
        ("NormalizeByElayer", NormalizeByElayer(
            ptype='/project/biocomplexity/fa7sa/calo_dreamer/src/challenge_files/binning_dataset_2.xml',
            xml_file= 'electron', 
            cut=1.0e-7
        )),
        ("CutValues", CutValues(cut=1.0e-7)),
        ("SelectiveUniformNoise", SelectiveUniformNoise(
            noise_width=0.0e-6,
            cut=True,
            exclusions=list(range(-45, 0))  # Last 45 features
        )),
        ("ExclusiveLogitTransform", ExclusiveLogitTransform(
            delta=1.0e-6,
            rescale=True
        )),
        ("StandardizeFromFile", StandardizeFromFile(model_dir=model_dir)),
        ("LogEnergy", LogEnergy()),
        ("ScaleEnergy", ScaleEnergy(e_min=6.907755, e_max=13.815510)),
        ("AddFeaturesToCond", AddFeaturesToCond(split_index=6480)),
        ("Reshape", Reshape(shape=[1, 45, 16, 9]))
    ]
    
    for name, transform in transforms:
        print(f"  ✓ {name}")
    
    return transforms

def apply_forward_transforms(showers, energies, transforms):
    """Apply forward preprocessing"""
    print(f"\n{'='*80}")
    print("FORWARD PASS (Preprocessing)")
    print(f"{'='*80}\n")
    
    x = showers.clone()
    c = energies.clone()
    
    for i, (name, transform) in enumerate(transforms):
        try:
            x, c = transform(x, c, rev=False)
            
            print(f"Step {i+1}: {name}")
            if x is not None:
                print(f"  x: {tuple(x.shape)} | range: [{x.min():.6e}, {x.max():.6e}] | mean: {x.mean():.6e}")
                if torch.isnan(x).any():
                    print(f"  ⚠️  WARNING: NaN detected in x!")
                if torch.isinf(x).any():
                    print(f"  ⚠️  WARNING: Inf detected in x!")
            if c is not None:
                print(f"  c: {tuple(c.shape)} | range: [{c.min():.6e}, {c.max():.6e}] | mean: {c.mean():.6e}")
                if torch.isnan(c).any():
                    print(f"  ⚠️  WARNING: NaN detected in c!")
                if torch.isinf(c).any():
                    print(f"  ⚠️  WARNING: Inf detected in c!")
            print()
            
        except Exception as e:
            print(f"  ❌ ERROR in {name}: {str(e)}")
            raise
    
    return x, c

def apply_reverse_transforms(x, c, transforms):
    """Apply reverse postprocessing"""
    print(f"\n{'='*80}")
    print("REVERSE PASS (Postprocessing)")
    print(f"{'='*80}\n")
    
    x_rev = x.clone()
    c_rev = c.clone()
    
    # Apply transforms in reverse order
    for i, (name, transform) in enumerate(reversed(transforms)):
        step_num = len(transforms) - i
        
        try:
            x_rev, c_rev = transform(x_rev, c_rev, rev=True)
            
            print(f"Step {step_num}: {name} (reverse)")
            if x_rev is not None:
                print(f"  x: {tuple(x_rev.shape)} | range: [{x_rev.min():.6e}, {x_rev.max():.6e}] | mean: {x_rev.mean():.6e}")
                if torch.isnan(x_rev).any():
                    print(f"  ⚠️  WARNING: NaN detected in x!")
                if torch.isinf(x_rev).any():
                    print(f"  ⚠️  WARNING: Inf detected in x!")
            if c_rev is not None:
                print(f"  c: {tuple(c_rev.shape)} | range: [{c_rev.min():.6e}, {c_rev.max():.6e}] | mean: {c_rev.mean():.6e}")
                if torch.isnan(c_rev).any():
                    print(f"  ⚠️  WARNING: NaN detected in c!")
                if torch.isinf(c_rev).any():
                    print(f"  ⚠️  WARNING: Inf detected in c!")
            print()
            
        except Exception as e:
            print(f"  ❌ ERROR in {name} (reverse): {str(e)}")
            raise
    
    return x_rev, c_rev

def verify_reconstruction(original_showers, original_energies, 
                         reconstructed_showers, reconstructed_energies):
    """Check if reconstruction matches original"""
    print(f"\n{'='*80}")
    print("RECONSTRUCTION VERIFICATION")
    print(f"{'='*80}\n")
    
    # CRITICAL: Verify shapes match
    print(f"Shape Verification:")
    print(f"  Original showers: {original_showers.shape}")
    print(f"  Reconstructed showers: {reconstructed_showers.shape}")
    print(f"  Original energies: {original_energies.shape}")
    print(f"  Reconstructed energies: {reconstructed_energies.shape}")
    
    if original_showers.shape != reconstructed_showers.shape:
        print(f"  ⚠️  WARNING: Shower shapes don't match!")
    if original_energies.shape != reconstructed_energies.shape:
        print(f"  ⚠️  WARNING: Energy shapes don't match!")
        reconstructed_energies = reconstructed_energies.unsqueeze(1)
    print()
    print("FIXED: ", reconstructed_energies.shape)
    print()
    
    # Flatten for comparison
    orig_flat = original_showers.flatten()
    recon_flat = reconstructed_showers.flatten()
    
    # Compute errors
    abs_error = torch.abs(recon_flat - orig_flat)
    rel_error = abs_error / (torch.abs(orig_flat) + 1e-10)
    
    print("Shower Reconstruction:")
    print(f"  Max absolute error: {abs_error.max():.6e}")
    print(f"  Mean absolute error: {abs_error.mean():.6e}")
    print(f"  Max relative error: {rel_error.max():.6e}")
    print(f"  Mean relative error: {rel_error.mean():.6e}")
    
    # Energy comparison
    energy_abs_error = torch.abs(reconstructed_energies - original_energies)
    energy_rel_error = energy_abs_error / (torch.abs(original_energies) + 1e-10)
    
    print("\nEnergy Reconstruction:")
    print(f"  Max absolute error: {energy_abs_error.max():.6e}")
    print(f"  Mean absolute error: {energy_abs_error.mean():.6e}")
    print(f"  Max relative error: {energy_rel_error.max():.6e}")
    print(f"  Mean relative error: {energy_rel_error.mean():.6e}")
    
    # Check for exact match (within tolerance)
    shower_match = torch.allclose(original_showers, reconstructed_showers, 
                                   rtol=1e-3, atol=1e-5)
    energy_match = torch.allclose(original_energies, reconstructed_energies,
                                   rtol=1e-3, atol=1e-5)
    
    print(f"\nReconstruction Status:")
    print(f"  Showers match (rtol=1e-3, atol=1e-4): {'✓ PASS' if shower_match else '✗ FAIL'}")
    print(f"  Energies match (rtol=1e-3, atol=1e-5): {'✓ PASS' if energy_match else '✗ FAIL'}")
    
    # Detailed comparison for first sample
    print(f"\nFirst Sample Detailed Comparison:")
    print(f"  Original energy: {original_energies[0, 0].item():.6f}")
    print(f"  Reconstructed energy: {reconstructed_energies[0, 0].item():.6f}")

    print(f"  Original shower sum: {original_showers[0].sum().item():.6f}")
    print(f"  Reconstructed shower sum: {reconstructed_showers[0].sum().item():.6f}")

    
    # Check layer energies (assuming shape is (B, 6480))
    n_layers = 45
    n_bins_per_layer = 144  # 16x9 for dataset_2
    
    print(f"\nPer-Layer Energy Comparison (first sample):")
    for layer_idx in range(min(10, n_layers)):  # Show first 10 layers
        start = layer_idx * n_bins_per_layer
        end = start + n_bins_per_layer
        orig_layer_E = original_showers[0, start:end].sum().item()
        recon_layer_E = reconstructed_showers[0, start:end].sum().item()
        error = abs(orig_layer_E - recon_layer_E)
        rel_err = error / (orig_layer_E + 1e-10) * 100
        print(f"  Layer {layer_idx:2d}: orig={orig_layer_E:8.3f}, recon={recon_layer_E:8.3f}, "
              f"abs_err={error:.3e}, rel_err={rel_err:6.2f}%")
    
    return shower_match and energy_match, {
        'abs_error': abs_error,
        'rel_error': rel_error,
        'energy_abs_error': energy_abs_error,
        'energy_rel_error': energy_rel_error
    }

def plot_reconstruction_analysis(original_showers, original_energies,
                                reconstructed_showers, reconstructed_energies,
                                errors, save_dir="./plots"):
    """Create comprehensive visualization of reconstruction quality"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # ========================================================================
    # 1. INCIDENT ENERGIES COMPARISON
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Scatter plot: Original vs Reconstructed
    axes[0, 0].scatter(original_energies.numpy(), reconstructed_energies.numpy(), 
                       alpha=0.3, s=10)
    axes[0, 0].plot([original_energies.min(), original_energies.max()],
                    [original_energies.min(), original_energies.max()], 
                    'r--', lw=2, label='Perfect reconstruction')
    axes[0, 0].set_xlabel('Original Energy (MeV)', fontsize=12)
    axes[0, 0].set_ylabel('Reconstructed Energy (MeV)', fontsize=12)
    axes[0, 0].set_title('Incident Energy: Original vs Reconstructed', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of energies
    axes[0, 1].hist(original_energies.numpy(), bins=50, alpha=0.5, 
                    label='Original', color='blue', density=True)
    axes[0, 1].hist(reconstructed_energies.numpy(), bins=50, alpha=0.5, 
                    label='Reconstructed', color='red', density=True)
    axes[0, 1].set_xlabel('Energy (MeV)', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Energy Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy error distribution
    energy_errors = (reconstructed_energies - original_energies).numpy()
    energy_errors=energy_errors.flatten()
    axes[1, 0].hist(energy_errors, bins=50, color='green', alpha=0.7)
    axes[1, 0].axvline(0, color='red', linestyle='--', lw=2, label='Zero error')
    axes[1, 0].set_xlabel('Reconstruction Error (MeV)', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Energy Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Relative error
    energy_rel_errors = (errors['energy_rel_error'] * 100).numpy()
    axes[1, 1].hist(energy_rel_errors, bins=50, color='orange', alpha=0.7)
    axes[1, 1].axvline(0, color='red', linestyle='--', lw=2, label='Zero error')
    axes[1, 1].set_xlabel('Relative Error (%)', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Energy Relative Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/energy_reconstruction.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/energy_reconstruction.png")
    plt.close()
    
    # ========================================================================
    # 2. SHOWER ENERGY COMPARISON
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total shower energies
    orig_shower_energies = original_showers.sum(dim=1).numpy()
    recon_shower_energies = reconstructed_showers.sum(dim=1).numpy()
    
    # Scatter plot
    axes[0, 0].scatter(orig_shower_energies, recon_shower_energies, alpha=0.3, s=10)
    axes[0, 0].plot([orig_shower_energies.min(), orig_shower_energies.max()],
                    [orig_shower_energies.min(), orig_shower_energies.max()], 
                    'r--', lw=2, label='Perfect reconstruction')
    axes[0, 0].set_xlabel('Original Shower Energy (MeV)', fontsize=12)
    axes[0, 0].set_ylabel('Reconstructed Shower Energy (MeV)', fontsize=12)
    axes[0, 0].set_title('Total Shower Energy: Original vs Reconstructed', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram
    axes[0, 1].hist(orig_shower_energies, bins=50, alpha=0.5, 
                    label='Original', color='blue', density=True)
    axes[0, 1].hist(recon_shower_energies, bins=50, alpha=0.5, 
                    label='Reconstructed', color='red', density=True)
    axes[0, 1].set_xlabel('Total Shower Energy (MeV)', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Shower Energy Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Voxel-level error histogram
    axes[1, 0].hist(errors['abs_error'].numpy(), bins=100, color='purple', alpha=0.7)
    axes[1, 0].set_xlabel('Absolute Error (MeV)', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Voxel-Level Absolute Error', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Relative error (clip for visualization)
    rel_err_clipped = np.clip(errors['rel_error'].numpy() * 100, 0, 100)
    axes[1, 1].hist(rel_err_clipped, bins=100, color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('Relative Error (%, clipped at 100%)', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Voxel-Level Relative Error', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shower_reconstruction.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/shower_reconstruction.png")
    plt.close()
    
    # ========================================================================
    # 3. PER-LAYER ENERGY COMPARISON (Multiple Samples)
    # ========================================================================
    n_layers = 45
    n_bins_per_layer = 144  # 16x9
    n_samples_to_plot = min(5, original_showers.shape[0])
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for sample_idx in range(n_samples_to_plot):
        # Compute layer energies
        orig_layer_energies = []
        recon_layer_energies = []
        
        for layer_idx in range(n_layers):
            start = layer_idx * n_bins_per_layer
            end = start + n_bins_per_layer
            orig_layer_energies.append(original_showers[sample_idx, start:end].sum().item())
            recon_layer_energies.append(reconstructed_showers[sample_idx, start:end].sum().item())
        
        # Plot
        ax = axes[sample_idx]
        x = np.arange(n_layers)
        width = 0.35
        
        ax.bar(x - width/2, orig_layer_energies, width, label='Original', alpha=0.7)
        ax.bar(x + width/2, recon_layer_energies, width, label='Reconstructed', alpha=0.7)
        
        ax.set_xlabel('Layer Index', fontsize=11)
        ax.set_ylabel('Energy (MeV)', fontsize=11)
        E0 = original_energies[sample_idx, 0].item()
        ax.set_title(
            f"Sample {sample_idx}: Layer Energies\n"
            f"(Incident E = {E0:.2f} MeV)",
            fontsize=12, fontweight="bold"
        )
        #ax.set_title(f'Sample {sample_idx}: Layer Energies\n'
                    # f'(Incident E = {original_energies[sample_idx]:.2f} MeV)', 
                    # fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Error plot in last subplot
    ax = axes[5]
    # Use first sample for error visualization
    orig_layer_energies = []
    recon_layer_energies = []
    for layer_idx in range(n_layers):
        start = layer_idx * n_bins_per_layer
        end = start + n_bins_per_layer
        orig_layer_energies.append(original_showers[0, start:end].sum().item())
        recon_layer_energies.append(reconstructed_showers[0, start:end].sum().item())
    
    layer_errors = np.array(recon_layer_energies) - np.array(orig_layer_energies)
    ax.bar(np.arange(n_layers), layer_errors, alpha=0.7, color='red')
    ax.axhline(0, color='black', linestyle='--', lw=2)
    ax.set_xlabel('Layer Index', fontsize=11)
    ax.set_ylabel('Reconstruction Error (MeV)', fontsize=11)
    ax.set_title('Sample 0: Layer-wise Reconstruction Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/layer_energies.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/layer_energies.png")
    plt.close()
    
    # ========================================================================
    # 4. SUMMARY STATISTICS
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    
    stats_text = f"""
    RECONSTRUCTION QUALITY SUMMARY
    {'='*60}
    
    INCIDENT ENERGIES:
      • Max absolute error: {errors['energy_abs_error'].max():.6e} MeV
      • Mean absolute error: {errors['energy_abs_error'].mean():.6e} MeV
      • Max relative error: {errors['energy_rel_error'].max()*100:.4f}%
      • Mean relative error: {errors['energy_rel_error'].mean()*100:.4f}%
    
    SHOWER VOXELS:
      • Total voxels: {errors['abs_error'].numel():,}
      • Max absolute error: {errors['abs_error'].max():.6e} MeV
      • Mean absolute error: {errors['abs_error'].mean():.6e} MeV
      • Max relative error: {errors['rel_error'].max()*100:.4f}%
      • Mean relative error: {errors['rel_error'].mean()*100:.4f}%
    
    TOTAL SHOWER ENERGIES:
      • Original mean: {orig_shower_energies.mean():.3f} MeV
      • Reconstructed mean: {recon_shower_energies.mean():.3f} MeV
      • Max difference: {abs(orig_shower_energies - recon_shower_energies).max():.6e} MeV
    
    PASS/FAIL (rtol=1e-3, atol=1e-5):
      • Energies: {'✓ PASS' if torch.allclose(original_energies, reconstructed_energies, rtol=1e-3, atol=1e-5) else '✗ FAIL'}
      • Showers: {'✓ PASS' if torch.allclose(original_showers, reconstructed_showers, rtol=1e-3, atol=1e-5) else '✗ FAIL'}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/summary_statistics.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/summary_statistics.png")
    plt.close()
    
    print(f"\n✓ All plots saved to: {save_dir}/")
def test_gradient_flow(transforms, device='cuda'):
    """
    Test that gradients flow correctly through the entire reverse pipeline
    """
    print("\n" + "="*80)
    print("GRADIENT FLOW TEST")
    print("="*80 + "\n")
    
    # Create test tensors with gradients enabled
    batch_size = 4
    x = torch.randn(batch_size, 1, 45, 16, 9, requires_grad=True, device=device)
    c = torch.randn(batch_size, 46, requires_grad=True, device=device)
    
    print(f"Input shapes:")
    print(f"  x: {x.shape} (requires_grad={x.requires_grad})")
    print(f"  c: {c.shape} (requires_grad={c.requires_grad})")
    
    try:
        # Apply reverse transforms
        print(f"\n{'─'*80}")
        print("Applying reverse transforms...")
        print(f"{'─'*80}\n")
        
        x_rev, c_rev = apply_reverse_transforms(x, c, transforms)
        
        print(f"Output shapes:")
        print(f"  x_rev: {x_rev.shape} (requires_grad={x_rev.requires_grad})")
        print(f"  c_rev: {c_rev.shape} (requires_grad={c_rev.requires_grad})")
        
        # Compute a simple loss
        print(f"\n{'─'*80}")
        print("Computing loss and backpropagating...")
        print(f"{'─'*80}\n")
        
        loss = x_rev.sum() + c_rev.sum()
        print(f"Loss value: {loss.item():.6f}")
        
        # Backpropagate
        loss.backward()
        
        # Check gradients
        print(f"\n{'─'*80}")
        print("Checking gradients...")
        print(f"{'─'*80}\n")
        
        issues = []
        
        # Check x gradients
        if x.grad is None:
            issues.append("❌ x.grad is None (no gradient computed)")
        else:
            if torch.isnan(x.grad).any():
                nan_count = torch.isnan(x.grad).sum().item()
                issues.append(f"❌ x.grad contains {nan_count} NaN values")
            if torch.isinf(x.grad).any():
                inf_count = torch.isinf(x.grad).sum().item()
                issues.append(f"❌ x.grad contains {inf_count} Inf values")
            
            if not issues:
                grad_stats = {
                    'mean': x.grad.mean().item(),
                    'std': x.grad.std().item(),
                    'min': x.grad.min().item(),
                    'max': x.grad.max().item(),
                    'abs_mean': x.grad.abs().mean().item()
                }
                print(f"✓ x.grad statistics:")
                print(f"  Shape: {x.grad.shape}")
                print(f"  Mean: {grad_stats['mean']:.6e}")
                print(f"  Std:  {grad_stats['std']:.6e}")
                print(f"  Min:  {grad_stats['min']:.6e}")
                print(f"  Max:  {grad_stats['max']:.6e}")
                print(f"  Abs Mean: {grad_stats['abs_mean']:.6e}")
        
        # Check c gradients
        if c.grad is None:
            issues.append("❌ c.grad is None (no gradient computed)")
        else:
            if torch.isnan(c.grad).any():
                nan_count = torch.isnan(c.grad).sum().item()
                issues.append(f"❌ c.grad contains {nan_count} NaN values")
            if torch.isinf(c.grad).any():
                inf_count = torch.isinf(c.grad).sum().item()
                issues.append(f"❌ c.grad contains {inf_count} Inf values")
            
            if not issues:
                grad_stats = {
                    'mean': c.grad.mean().item(),
                    'std': c.grad.std().item(),
                    'min': c.grad.min().item(),
                    'max': c.grad.max().item(),
                    'abs_mean': c.grad.abs().mean().item()
                }
                print(f"\n✓ c.grad statistics:")
                print(f"  Shape: {c.grad.shape}")
                print(f"  Mean: {grad_stats['mean']:.6e}")
                print(f"  Std:  {grad_stats['std']:.6e}")
                print(f"  Min:  {grad_stats['min']:.6e}")
                print(f"  Max:  {grad_stats['max']:.6e}")
                print(f"  Abs Mean: {grad_stats['abs_mean']:.6e}")
        
        # Report results
        print(f"\n{'='*80}")
        if issues:
            print("❌ GRADIENT FLOW TEST FAILED")
            print(f"{'='*80}\n")
            for issue in issues:
                print(f"  {issue}")
            return False
        else:
            print("✓✓✓ GRADIENT FLOW TEST PASSED ✓✓✓")
            print(f"{'='*80}\n")
            print("Gradients flow correctly through the entire pipeline!")
            return True
            
    except Exception as e:
        print(f"\n{'='*80}")
        print("❌ GRADIENT FLOW TEST FAILED WITH EXCEPTION")
        print(f"{'='*80}\n")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
def main():
    """Main test function"""
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE VERIFICATION TEST")
    print("="*80)
    
    # Configuration
    data_path = "/project/biocomplexity/fa7sa/calochallenge_datasets/dataset_2/dataset_2_2_25k_part1.hdf5"
    n_samples = 25_000
    model_dir = "./test_stats_ds2_25k_p1/"
    plot_dir = "./verification_plots_25k_p1/"
    
    try:
        # 1. Load data
        original_showers, original_energies = load_dataset(data_path, n_samples)
        
        # 2. Create pipeline
        transforms = create_transform_pipeline(model_dir)
        
        # 3. Forward pass
        x_processed, c_processed = apply_forward_transforms(
            original_showers, original_energies, transforms
        )
        
        print(f"\n{'='*80}")
        print("PROCESSED DATA SUMMARY")
        print(f"{'='*80}\n")
        print(f"Processed x shape: {x_processed.shape}")
        print(f"Processed c shape: {c_processed.shape}")
        
        # 4. Reverse pass
        reconstructed_showers, reconstructed_energies = apply_reverse_transforms(
            x_processed, c_processed, transforms
        )
        
        # Squeeze energy back to (B,) if needed
        if reconstructed_energies.dim() > 1:
            reconstructed_energies = reconstructed_energies.squeeze(1)
        
        # 5. Verify reconstruction
        success, errors = verify_reconstruction(
            original_showers, original_energies,
            reconstructed_showers, reconstructed_energies
        )
        
        # 6. Create visualizations
        print(f"\n{'='*80}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        plot_reconstruction_analysis(
            original_showers, original_energies,
            reconstructed_showers, reconstructed_energies,
            errors, save_dir=plot_dir
        )
        
        # Final verdict
        print(f"\n{'='*80}")
        print("FINAL RESULT")
        print(f"{'='*80}\n")
        if success:
            print("✓✓✓ PIPELINE TEST PASSED ✓✓✓")
            print("Forward and reverse transforms are correctly implemented!")
        else:
            print("✗✗✗ PIPELINE TEST FAILED ✗✗✗")
            print("There are issues with the transform pipeline.")
            print("Check the errors above and visualization plots for details.")
        print()
        gradient_success = test_gradient_flow(transforms, device='cuda')

        if gradient_success:
            print("  ✓ Gradients flow through full pipeline")
        else:
            print("  ✗ Gradient flow issues in pipeline")
    except Exception as e:
        print(f"\n{'='*80}")
        print("❌ TEST FAILED WITH EXCEPTION")
        print(f"{'='*80}\n")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print()

if __name__ == "__main__":
    main()