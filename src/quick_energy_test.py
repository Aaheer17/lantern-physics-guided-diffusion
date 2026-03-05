#!/usr/bin/env python3
"""
Quick Energy Loss Test

Simple script to check if layer energies match between:
1. Summing reconstructed voxels
2. Computing from condition u values

Usage:
    python quick_energy_test.py
"""

import torch
import numpy as np
from itertools import pairwise

def test_energy_reconstruction():
    """
    Test case: Create synthetic data and check if energy reconstruction works
    """
    print("\n" + "="*80)
    print("SYNTHETIC TEST: Energy Reconstruction")
    print("="*80 + "\n")
    
    batch_size = 10
    n_layers = 45
    voxels_per_layer = 144  # 16x9
    total_voxels = n_layers * voxels_per_layer  # 6480
    
    # Create synthetic incident energy
    E_inc = torch.rand(batch_size, 1) * 10000 + 1000  # 1000-11000 MeV
    
    # Create synthetic u values (energy fractions)
    u_values = torch.rand(batch_size, n_layers)
    u_values = u_values / u_values.sum(dim=1, keepdim=True)  # Normalize to sum to 1
    
    # Compute layer energies from u values (GROUND TRUTH)
    u_clipped = u_values.clone()
    u_clipped[:, 1:] = torch.clip(u_clipped[:, 1:], 0, 1)
    
    layer_Es_from_u = []
    total_E = E_inc.flatten() * u_clipped[:, 0]
    cum_sum = torch.zeros_like(total_E)
    
    for i in range(n_layers - 1):
        layer_E = (total_E - cum_sum) * u_clipped[:, i+1]
        layer_Es_from_u.append(layer_E)
        cum_sum += layer_E
    layer_Es_from_u.append(total_E - cum_sum)
    
    E_from_u = torch.stack(layer_Es_from_u, dim=1)  # (B, 45)
    
    # Create synthetic voxels that match these energies
    # Each layer's voxels should sum to the layer energy
    voxels = torch.zeros(batch_size, total_voxels)
    
    for b in range(batch_size):
        for layer in range(n_layers):
            start = layer * voxels_per_layer
            end = start + voxels_per_layer
            # Distribute layer energy across voxels randomly
            layer_voxel_values = torch.rand(voxels_per_layer)
            layer_voxel_values = layer_voxel_values / layer_voxel_values.sum() * E_from_u[b, layer]
            voxels[b, start:end] = layer_voxel_values
    
    # Now sum voxels per layer to get energies (PREDICTED)
    layer_boundaries = list(range(0, total_voxels + 1, voxels_per_layer))
    E_from_voxels = []
    
    for start, end in pairwise(layer_boundaries):
        layer_energy = voxels[:, start:end].sum(dim=-1)
        E_from_voxels.append(layer_energy)
    
    E_from_voxels = torch.stack(E_from_voxels, dim=1)  # (B, 45)
    
    # Compare
    print("First sample comparison:")
    print(f"  E_inc: {E_inc[0, 0].item():.2f}")
    print(f"  Sum of E_from_u: {E_from_u[0].sum().item():.2f}")
    print(f"  Sum of E_from_voxels: {E_from_voxels[0].sum().item():.2f}")
    print(f"  Sum of all voxels: {voxels[0].sum().item():.2f}")
    
    print(f"\nFirst 5 layers:")
    for i in range(5):
        print(f"  Layer {i}: E_u={E_from_u[0, i]:.2f}, E_voxels={E_from_voxels[0, i]:.2f}")
    
    # Compute relative error
    rel_error = (E_from_voxels - E_from_u) / (E_from_u.abs() + 1e-6)
    energy_loss = torch.mean(rel_error ** 2)
    
    print(f"\nEnergy Loss:")
    print(f"  Mean relative error: {rel_error.mean().item():.6f}")
    print(f"  RMS relative error: {torch.sqrt((rel_error**2).mean()).item():.6f}")
    print(f"  Energy loss (MSE): {energy_loss.item():.6f}")
    
    if energy_loss < 1e-6:
        print(f"\n✓ SYNTHETIC TEST PASSED: Energy reconstruction is correct")
        print(f"  This means the formulas are right")
        print(f"  Problem must be in the actual data/transforms")
    else:
        print(f"\n✗ SYNTHETIC TEST FAILED: Formula error detected")
    
    return energy_loss.item()

def check_your_functions():
    """
    Check if the actual functions produce the expected behavior
    """
    print("\n" + "="*80)
    print("CHECKING YOUR ACTUAL FUNCTIONS")
    print("="*80 + "\n")
    
    print("This would require importing your actual model code.")
    print("To run this test:")
    print("  1. Import your tbd_Diff model")
    print("  2. Load a checkpoint")
    print("  3. Get a batch from your data loader")
    print("  4. Run:")
    print("     - voxels, energy = model.apply_inverse_transforms(x_0_hat, condition)")
    print("     - E_pred = model.compute_layer_energies_from_voxels(voxels)")
    print("     - E_true = model.get_ground_truth_layer_energies(condition)")
    print("     - Compare E_pred and E_true")

if __name__ == '__main__':
    print("\n" + "="*80)
    print("ENERGY LOSS DIAGNOSTIC - QUICK TEST")
    print("="*80)
    
    # Run synthetic test
    loss = test_energy_reconstruction()
    
    if loss < 1e-6:
        print("\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        print("\n✓ The formulas are correct!")
        print("\n❌ The problem is in your actual data:")
        print("  1. apply_inverse_transforms() might be scaling wrong")
        print("  2. Or get_ground_truth_layer_energies() is computing wrong")
        print("  3. Or condition u values are in wrong units")
        print("\nRun diagnose_energy_loss.py on real data to find which one!")
    
    # Show how to check with real functions
    check_your_functions()
