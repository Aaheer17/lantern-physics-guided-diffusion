"""
Auxiliary Loss Components for Shape Network
Includes: Energy Conservation, MVN Regularization, CFD Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class EnergyConservationLoss(nn.Module):
    """
    Ensures total energy in each layer matches predicted layer energies
    
    With optional variance-stabilizing transformation:
        y = 2 * sqrt(E + E_0)
    
    This transformation balances loss across layers with different energy scales
    and is the standard Anscombe transform for Poisson-distributed data.
    
    Updated to handle two input formats:
    1. Layer energies directly: (B, 45)
    2. Full spatial structure: (B, 1, L, H, W) - will sum to get layer energies
    """
    def __init__(self, loss_type='huber', huber_delta=1.0, 
                 per_layer_var=False, num_layers=45,
                 use_transform=False, E_0=1e-3):
        """
        Args:
            loss_type: 'mse', 'huber', or 'heteroscedastic'
            huber_delta: delta for Huber loss
            per_layer_var: (deprecated, kept for compatibility)
            num_layers: number of layers
            use_transform: if True, apply y = 2*sqrt(E + E_0) before computing loss
            E_0: offset for sqrt transform (prevents sqrt(0))
        """
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.per_layer_var = per_layer_var
        self.num_layers = num_layers
        self.use_transform = use_transform
        self.E_0 = E_0
        self.eps= 1e-6
        if self.loss_type == 'heteroscedastic':
            # Learnable per-layer variance
            self.log_var = nn.Parameter(torch.zeros(num_layers))
    
    def forward(self, x_shape: torch.Tensor, E_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_shape: Either:(both from generated samples from shape network)
                - (B, L) - layer energies directly from apply_inverse_transforms
                - (B, 1, L, H, W) - predicted clean shape in physical energy domain
            E_pred: (B, L) - layer energies from  (condition)/or true data
        
        Returns:
            energy_loss: scalar loss value
        """
        # Compute layer energies from x_shape
        if x_shape.ndim == 2:  # (B, L) - already layer energies
            E_shape = x_shape
        elif x_shape.ndim == 5:  # (B, 1, L, H, W) - need to sum
            E_shape = x_shape.sum(dim=(-2, -1)).squeeze(1)  # (B, L)
        else:
            raise ValueError(f"Unexpected x_shape dimension: {x_shape.shape}. "
                           f"Expected (B, {self.num_layers}) or (B, 1, {self.num_layers}, H, W)")
        
        # Verify shape matches num_layers
        if E_shape.shape[1] != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {E_shape.shape[1]}")
        
        # Apply variance-stabilizing transformation if enabled
        if self.use_transform:
            # y = 2 * sqrt(E + E_0)
            # Balances loss across different energy scales
            E_shape_safe = torch.clamp(E_shape, min=0.0)
            E_pred_safe = torch.clamp(E_pred, min=0.0)
            
            y_shape = 2.0 * torch.sqrt(E_shape_safe + self.E_0)
            y_pred = 2.0 * torch.sqrt(E_pred_safe + self.E_0)
            
            # Compute residual in transformed space
            energy_residual = y_pred - y_shape  # (B, L)
        else:
            # Standard energy residual
            energy_residual = E_pred - E_shape  # (B, L)
        
        # Compute loss
        if self.loss_type == 'mse':
            loss = torch.mean(energy_residual ** 2)
        
        elif self.loss_type == 'huber':
            # For Huber, need to pass the actual values (not residual)
            if self.use_transform:
                loss = F.huber_loss(
                    y_shape, y_pred, 
                    reduction='mean', 
                    delta=self.huber_delta
                )
            else:
                loss = F.huber_loss(
                    E_shape, E_pred, 
                    reduction='mean', 
                    delta=self.huber_delta
                )
        
        elif self.loss_type == 'heteroscedastic':
            # Loss = ||residual||^2 / (2*σ^2) + log(σ^2)
            # Prevents σ → ∞ while learning task-specific uncertainty
            precision = torch.exp(-self.log_var)  # (L,)
            weighted_mse = precision.unsqueeze(0) * (energy_residual ** 2)
            regularizer = self.log_var.unsqueeze(0)
            loss = torch.mean(weighted_mse + regularizer)
            # NEW: Relative error losses
        elif self.loss_type == 'relative':
            # Relative error: |E_pred - E_shape| / (E_shape + eps)
            # Scale-invariant and naturally O(0.01-0.1)
            relative_error = torch.abs(E_pred - E_shape) / (E_shape + self.eps)
            loss = torch.mean(relative_error)
        
        elif self.loss_type == 'relative_squared':
            # Mean squared relative error
            relative_error = (E_pred - E_shape) / (E_shape + self.eps)
            loss = torch.mean(relative_error ** 2)
        
        elif self.loss_type == 'relative_huber':
            # Huber loss on relative error
            # Add this in your loss computation
            print(f"E_pred range: [{E_pred.min():.6f}, {E_pred.max():.6f}]")
            print(f"E_shape range: [{E_shape.min():.6f}, {E_shape.max():.6f}]")
            print(f"E_pred mean: {E_pred.mean():.6f}")
            print(f"Denominator (E_pred + eps): {(E_pred + self.eps).min():.6f}")
           
            # How many near-zero values?
            near_zero = (E_pred < 1e-3).sum()
            print(f"Near-zero E_pred values: {near_zero} / {E_pred.numel()}")
            #relative_error = (E_pred - E_shape) / (E_shape + self.eps)
            relative_error = (E_shape - E_pred) / (E_pred + self.eps)
            print(f"Relative error range: [{relative_error.min():.2f}, {relative_error.max():.2f}]")
            loss = F.huber_loss(
                torch.zeros_like(relative_error), 
                relative_error,
                reduction='mean',
                delta=0.1
            )
        elif self.loss_type == 'adaptive_relative_huber':
            # Adaptive: (true - generated) / generated
            # Prevents energy collapse, natural curriculum
            relative_error = (E_pred - E_shape) / (E_shape + self.eps)
            
            loss = F.huber_loss(
                relative_error,
                torch.zeros_like(relative_error),
                reduction='mean',
                delta=self.huber_delta
            )
        elif self.loss_type == 'log_relative_huber':
            # Work in log space - naturally handles wide dynamic range
            log_shape = torch.log(E_shape + 1.0)
            log_pred = torch.log(E_pred + 1.0)
            
            loss = F.huber_loss(
                log_shape,
                log_pred,
                reduction='mean',
                delta=self.huber_delta
            )
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss
    
    def get_constraint_violation(self, x_shape: torch.Tensor, 
                                 E_pred: torch.Tensor) -> torch.Tensor:
        """
        For Augmented Lagrangian method
        
        Args:
            x_shape: Either (B, L) or (B, 1, L, H, W)
            E_pred: (B, L)
        
        Returns:
            violations: (B, L) - energy mismatch per layer
        """
        # Compute layer energies from x_shape
        if x_shape.ndim == 2:  # (B, L)
            E_shape = x_shape
        elif x_shape.ndim == 5:  # (B, 1, L, H, W)
            E_shape = x_shape.sum(dim=(-2, -1)).squeeze(1)
        else:
            raise ValueError(f"Unexpected x_shape dimension: {x_shape.shape}")
        
        if self.use_transform:
            E_shape_safe = torch.clamp(E_shape, min=0.0)
            E_pred_safe = torch.clamp(E_pred, min=0.0)
            y_shape = 2.0 * torch.sqrt(E_shape_safe + self.E_0)
            y_pred = 2.0 * torch.sqrt(E_pred_safe + self.E_0)
            return y_pred - y_shape
        else:
            return E_pred - E_shape


# class MVNRegularizationLoss(nn.Module):
#     """
#     Regularize layer-wise energy distribution to match target MVN statistics.

#     Supports two input formats:
#       1) (B, L): layer energies directly
#       2) (B, 1, L, H, W): spatial tensor; sums over H,W to get (B, L)

#     Notes:
#       - EMA buffers are for monitoring only (no gradients).
#       - Loss is computed from current batch statistics to preserve gradients.
#       - Targets are stored as buffers and updated in-place with copy_().
#     """
#     def __init__(
#         self,
#         num_layers: int = 45,
#         lambda_mean: float = 1.0,
#         lambda_cov: float = 0.5,
#         shrinkage_gamma: float = 0.1,
#         ema_rho: float = 0.99,
#         normalize_cov_loss: bool = True,
#     ):
#         super().__init__()
#         self.num_layers = num_layers
#         self.lambda_mean = float(lambda_mean)
#         self.lambda_cov = float(lambda_cov)
#         self.shrinkage_gamma = float(shrinkage_gamma)
#         self.ema_rho = float(ema_rho)
#         self.normalize_cov_loss = bool(normalize_cov_loss)

#         # Target statistics (set once from real data)
#         self.register_buffer("mu_target", torch.zeros(num_layers))
#         self.register_buffer("Sigma_target", torch.eye(num_layers))

#         # Cached identity for shrinkage (will be cast/moved as needed in forward)
#         self.register_buffer("identity_matrix", torch.eye(num_layers))

#         # EMA buffers (monitoring only)
#         self.register_buffer("mu_ema", torch.zeros(num_layers))
#         self.register_buffer("Sigma_ema", torch.eye(num_layers))
#         self.register_buffer("initialized", torch.tensor(False))

#     def _extract_layer_energies(self, x_shape: torch.Tensor) -> torch.Tensor:
#         """
#         Returns:
#             E_batch: (B, L)
#         """
#         if x_shape.ndim == 2:
#             # (B, L)
#             E_batch = x_shape
#         elif x_shape.ndim == 5:
#             # (B, 1, L, H, W) -> (B, L)
#             E_batch = x_shape.sum(dim=(-2, -1)).squeeze(1)
#         else:
#             raise ValueError(
#                 f"Unexpected x_shape dim={x_shape.ndim} with shape {tuple(x_shape.shape)}. "
#                 f"Expected (B, {self.num_layers}) or (B, 1, {self.num_layers}, H, W)."
#             )

#         if E_batch.shape[1] != self.num_layers:
#             raise ValueError(
#                 f"Expected {self.num_layers} layers, got {E_batch.shape[1]} (shape={tuple(E_batch.shape)})."
#             )
#         return E_batch

#     @torch.no_grad()
#     def set_targets(self, mu: torch.Tensor, Sigma: torch.Tensor, device: str | torch.device | None = None):
#         """
#         Set target mean/covariance in-place.

#         Args:
#             mu: (L,)
#             Sigma: (L, L)
#             device: optional; if provided, targets are stored on this device
#         """
#         if mu.ndim != 1 or mu.numel() != self.num_layers:
#             raise ValueError(f"mu must be shape ({self.num_layers},), got {tuple(mu.shape)}")
#         if Sigma.ndim != 2 or Sigma.shape != (self.num_layers, self.num_layers):
#             raise ValueError(f"Sigma must be shape ({self.num_layers},{self.num_layers}), got {tuple(Sigma.shape)}")

#         if device is None:
#             device = self.mu_target.device

#         mu = mu.to(device=device, dtype=self.mu_target.dtype)
#         Sigma = Sigma.to(device=device, dtype=self.Sigma_target.dtype)

#         self.mu_target.copy_(mu)
#         self.Sigma_target.copy_(Sigma)

#     @torch.no_grad()
#     def set_targets_from_data(
#         self,
#         data_loader,
#         device: str | torch.device = "cuda",
#         accumulate_on_gpu: bool = False,
#     ):
#         """
#         Compute target mean/covariance from real data.

#         Args:
#             data_loader: yields batches where batch[0] is (B,1,L,H,W)
#             device: where to store targets
#             accumulate_on_gpu: if True, accumulate energies on GPU (faster, uses more VRAM)
#         """
#         all_energies = []

#         if accumulate_on_gpu:
#             for batch in data_loader:
#                 x_real = batch[0].to(device)
#                 E_real = x_real.sum(dim=(-2, -1)).squeeze(1)  # (B,L)
#                 all_energies.append(E_real)
#             all_energies = torch.cat(all_energies, dim=0)  # (N,L) on GPU
#             mu = all_energies.mean(dim=0)
#             E_centered = all_energies - mu.unsqueeze(0)
#             Sigma = (E_centered.T @ E_centered) / max(all_energies.size(0) - 1, 1)
#             self.set_targets(mu, Sigma, device=device)
#         else:
#             # Accumulate on CPU to avoid VRAM spikes
#             for batch in data_loader:
#                 x_real = batch[0].to(device)
#                 E_real = x_real.sum(dim=(-2, -1)).squeeze(1)  # (B,L)
#                 all_energies.append(E_real.detach().cpu())
#             all_energies = torch.cat(all_energies, dim=0)  # (N,L) on CPU
#             mu_cpu = all_energies.mean(dim=0)
#             E_centered = all_energies - mu_cpu.unsqueeze(0)
#             Sigma_cpu = (E_centered.T @ E_centered) / max(all_energies.size(0) - 1, 1)
#             self.set_targets(mu_cpu, Sigma_cpu, device=device)

#     def forward(self, x_shape: torch.Tensor) -> torch.Tensor:
#         """
#         Returns:
#             loss: scalar
#         """
#         E_batch = self._extract_layer_energies(x_shape)  # (B,L)
#         B = E_batch.size(0)

#         # Batch mean/cov (keeps gradients)
#         mu_batch = E_batch.mean(dim=0)  # (L,)
#         E_centered = E_batch - mu_batch.unsqueeze(0)  # (B,L)

#         denom = max(B - 1, 1)
#         Sigma_batch = (E_centered.T @ E_centered) / denom  # (L,L)

#         # Shrinkage (keeps gradients)
#         I = self.identity_matrix.to(device=Sigma_batch.device, dtype=Sigma_batch.dtype)
#         Sigma_batch_shrunk = (1.0 - self.shrinkage_gamma) * Sigma_batch + self.shrinkage_gamma * I

#         # EMA update for monitoring only
#         with torch.no_grad():
#             if not bool(self.initialized.item()):
#                 self.mu_ema.copy_(mu_batch)
#                 self.Sigma_ema.copy_(Sigma_batch_shrunk)
#                 self.initialized.fill_(True)
#             else:
#                 self.mu_ema.mul_(self.ema_rho).add_(mu_batch, alpha=(1.0 - self.ema_rho))
#                 self.Sigma_ema.mul_(self.ema_rho).add_(Sigma_batch_shrunk, alpha=(1.0 - self.ema_rho))

#         # Loss computed from current batch stats (preserves gradients)
#         diff_mean = mu_batch - self.mu_target.to(device=mu_batch.device, dtype=mu_batch.dtype)
#         loss_mean = torch.sum(diff_mean * diff_mean)

#         diff_cov = Sigma_batch_shrunk - self.Sigma_target.to(device=Sigma_batch_shrunk.device, dtype=Sigma_batch_shrunk.dtype)
#         loss_cov = torch.sum(diff_cov * diff_cov)
#         if self.normalize_cov_loss:
#             loss_cov = loss_cov / float(self.num_layers * self.num_layers)

#         return self.lambda_mean * loss_mean + self.lambda_cov * loss_cov

#     def get_constraint_violation(self, x_shape: torch.Tensor) -> torch.Tensor:
#         """
#         For Augmented Lagrangian: mean deviation from target mean.
#         Returns:
#             (L,)
#         """
#         E_batch = self._extract_layer_energies(x_shape)
#         mu_batch = E_batch.mean(dim=0)
#         return mu_batch - self.mu_target.to(device=mu_batch.device, dtype=mu_batch.dtype)



class MVNMahalanobisLoss(nn.Module):
    """
    Per-sample Mahalanobis distance squared:
        d^2(x) = || L^{-1} (x - mu) ||^2
    where L = chol(Sigma_stable).

    Expects x as (B, L).
    Returns a scalar (mean over batch) by default.
    """

    def __init__(
        self,
        targets_pt_path: str,
        num_layers: int = 45,
        reduction: str = "mean",   # "mean" or "none"
        clip_max: float | None = None,  # robustify heavy tails
        use_log1p: bool = False,         # optional robustify
        eps: float = 1e-12,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.reduction = reduction
        self.clip_max = clip_max
        self.use_log1p = use_log1p
        self.eps = eps

        d = torch.load(targets_pt_path, map_location="cpu")

        mu = d["mu"]                          # (L,)
        chol = d["chol"]                      # (L,L) lower-tri

        if mu.numel() != num_layers or chol.shape != (num_layers, num_layers):
            raise ValueError(
                f"Target shapes mismatch: mu {tuple(mu.shape)}, chol {tuple(chol.shape)}, "
                f"expected ({num_layers},) and ({num_layers},{num_layers})"
            )

        # store as buffers so they move with .to(device)
        self.register_buffer("mu", mu.to(dtype=dtype))
        self.register_buffer("chol", chol.to(dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.num_layers:
            raise ValueError(f"Expected x shape (B,{self.num_layers}), got {tuple(x.shape)}")
        # inside MVNMahalanobisLoss.forward, temporarily
        with torch.no_grad():
            print("[MVN forward] x mean/std:", float(x.mean()), float(x.std(unbiased=False)))

        x = x.to(device=self.mu.device, dtype=self.mu.dtype)   # <— recommended
        diff = x - self.mu.unsqueeze(0)  # (B,L)
    
        y = torch.linalg.solve_triangular(self.chol, diff.T, upper=False)  # (L,B)
        d2 = (y * y).sum(dim=0)  # (B,)
    
        if self.clip_max is not None:
            d2 = torch.clamp(d2, max=float(self.clip_max))
        if self.use_log1p:
            d2 = torch.log1p(d2 + self.eps)
    
        if self.reduction == "mean":
            return d2.mean()
        if self.reduction == "none":
            return d2
        raise ValueError(f"Unknown reduction: {self.reduction}")


class CFDLoss(nn.Module):
    """
    Correlation Frobenius Distance Loss
    Preserves inter-layer correlation structure
    """
    def __init__(self, num_layers=45, shrinkage_gamma=0.1, ema_rho=0.99,
                 warmup_steps=2000, lambda_cfd_max=0.1, off_diagonal_only=True):
        super().__init__()
        self.num_layers = num_layers
        self.shrinkage_gamma = shrinkage_gamma
        self.ema_rho = ema_rho
        self.warmup_steps = warmup_steps
        self.lambda_cfd_max = lambda_cfd_max
        self.off_diagonal_only = off_diagonal_only
        
        # Target correlation from real data
        self.register_buffer('C_real', torch.eye(num_layers))
        
        # EMA correlation
        self.register_buffer('C_gen_ema', torch.eye(num_layers))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('step_count', torch.tensor(0))
        
        # Mask for off-diagonal elements
        if off_diagonal_only:
            mask = 1 - torch.eye(num_layers)
            self.register_buffer('mask', mask)
            self.normalization = num_layers * (num_layers - 1)
        else:
            self.register_buffer('mask', torch.ones(num_layers, num_layers))
            self.normalization = num_layers ** 2
    
    def set_target_correlation_from_data(self, data_loader, device='cuda'):
        """
        Compute target correlation matrix from real data
        
        Args:
            data_loader: DataLoader with real samples
        """
        print("Computing CFD target correlation from real data...")
        all_energies = []
        
        for batch in data_loader:
            x_real = batch[0].to(device)
            E_real = x_real.sum(dim=(-2, -1)).squeeze(1)
            all_energies.append(E_real.cpu())
        
        all_energies = torch.cat(all_energies, dim=0)  # (N, L)
        
        # Compute Pearson correlation
        E_centered = all_energies - all_energies.mean(dim=0, keepdim=True)
        cov = (E_centered.T @ E_centered) / (all_energies.size(0) - 1)
        
        # Stabilize: std = sqrt(diag(cov))
        std = torch.sqrt(torch.diag(cov) + 1e-8)
        corr = cov / (torch.ger(std, std) + 1e-8)
        
        self.C_real = corr.to(device)
        
        # Print diagnostics
        off_diag_values = corr[~torch.eye(self.num_layers, dtype=bool)]
        print(f"Target correlation off-diagonal range: [{off_diag_values.min():.3f}, "
              f"{off_diag_values.max():.3f}]")
        print(f"Mean off-diagonal correlation: {off_diag_values.mean():.3f}")
    
    def forward(self, x_shape: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_shape: (B, 1, L, H, W) - predicted shape
        
        Returns:
            cfd_loss: scalar loss value with warmup
        """
        # Compute layer energies
        E_batch = x_shape.sum(dim=(-2, -1)).squeeze(1)  # (B, L)
        B = E_batch.size(0)
        
        # Compute batch correlation
        E_centered = E_batch - E_batch.mean(dim=0, keepdim=True)
        cov_batch = (E_centered.T @ E_centered) / max(B - 1, 1)
        
        # Stabilize correlation computation
        std_batch = torch.sqrt(torch.diag(cov_batch) + 1e-8)
        C_batch = cov_batch / (torch.ger(std_batch, std_batch) + 1e-8)
        
        # Apply shrinkage
        C_shrink = ((1 - self.shrinkage_gamma) * C_batch + 
                   self.shrinkage_gamma * torch.eye(self.num_layers, device=x_shape.device))
        
        # Update EMA
        if not self.initialized:
            self.C_gen_ema.copy_(C_shrink.detach())
            self.initialized.copy_(torch.tensor(True))
        else:
            with torch.no_grad():
                self.C_gen_ema.mul_(self.ema_rho).add_(C_shrink, alpha=1 - self.ema_rho)
        
        # CFD loss (masked to off-diagonal if specified)
        diff = (self.C_gen_ema - self.C_real) * self.mask
        loss_cfd = torch.sum(diff ** 2) / self.normalization
        
        # Warm-up schedule
        self.step_count += 1
        warmup_factor = min(self.step_count.float() / self.warmup_steps, 1.0)
        
        return warmup_factor * self.lambda_cfd_max * loss_cfd
    
    def get_current_correlation(self) -> torch.Tensor:
        """
        Get current EMA correlation matrix (for monitoring)
        """
        return self.C_gen_ema.detach()
    
    def compute_validation_cfd(self, generated_samples: torch.Tensor) -> float:
        """
        Compute validation CFD without EMA (unbiased assessment)
        
        Args:
            generated_samples: (N, 1, L, H, W) - large set of generated samples
        
        Returns:
            cfd_rmse: off-diagonal RMSE
        """
        E_gen = generated_samples.sum(dim=(-2, -1)).squeeze(1)  # (N, L)
        
        # Compute correlation
        E_centered = E_gen - E_gen.mean(dim=0, keepdim=True)
        cov = (E_centered.T @ E_centered) / (E_gen.size(0) - 1)
        std = torch.sqrt(torch.diag(cov) + 1e-8)
        C_gen_val = cov / (torch.ger(std, std) + 1e-8)
        
        # Off-diagonal RMSE
        diff = (C_gen_val - self.C_real) * self.mask
        cfd_rmse = torch.sqrt(torch.sum(diff ** 2) / self.normalization)
        
        return cfd_rmse.item()


class AuxiliaryLossManager(nn.Module):
    """
    Manages all auxiliary losses for the shape network
    """
    def __init__(self, config: Dict):
        super().__init__()
        
        self.use_energy = config.get('use_energy_loss', False)
        self.use_mvn = config.get('use_mvn_loss', False)
        self.use_cfd = config.get('use_cfd_loss', False)
        
        num_layers = config.get('num_layers', 45)
        
        # Initialize loss components
        if self.use_energy:
            self.energy_loss = EnergyConservationLoss(
                loss_type=config.get('energy_loss_type', 'huber'),
                huber_delta=config.get('huber_delta', 1.0),
                num_layers=num_layers,
                use_transform=config.get('energy_use_transform', False),
                E_0=config.get('energy_E_0', 1e-3)
            )
        
        if self.use_mvn:
            # self.mvn_loss = MVNRegularizationLoss(
            #     num_layers=num_layers,
            #     lambda_mean=config.get('mvn_lambda_mean', 1.0),
            #     lambda_cov=config.get('mvn_lambda_cov', 0.5),
            #     shrinkage_gamma=config.get('mvn_shrinkage', 0.1),
            #     ema_rho=config.get('mvn_ema_rho', 0.99)
            # )
            #"/project/biocomplexity/fa7sa/Modified_Training_Objective/LANTERN_new/src/Models/mvn_shape_targets_ds2.pt"
            self.mvn_loss = MVNMahalanobisLoss(
                targets_pt_path="/project/biocomplexity/fa7sa/Modified_Training_Objective/LANTERN_new/src/Models/mvn_shape_targets_ds2.pt",
                num_layers=45,
                reduction="mean",
            
                clip_max=None,
                use_log1p=False,   # soft, smooth, no cutoff
            
                eps=1e-12,
                dtype=torch.float32,
            )

        
        if self.use_cfd:
            self.cfd_loss = CFDLoss(
                num_layers=num_layers,
                shrinkage_gamma=config.get('cfd_shrinkage', 0.1),
                ema_rho=config.get('cfd_ema_rho', 0.99),
                warmup_steps=config.get('cfd_warmup_steps', 2000),
                lambda_cfd_max=config.get('lambda_cfd', 0.1),
                off_diagonal_only=config.get('cfd_off_diagonal_only', True)
            )
    
    def set_targets_from_data(self, data_loader, device='cuda'):
        """
        Pre-compute target statistics from real data
        """
        if self.use_mvn:
            self.mvn_loss.set_targets_from_data(data_loader, device)
        
        if self.use_cfd:
            self.cfd_loss.set_target_correlation_from_data(data_loader, device)
    
    def compute_losses(self, x_shape_physical: torch.Tensor, 
                      E_pred: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all auxiliary losses
        
        Args:
            x_shape_physical: (B, 1, L, H, W) - predicted shape in PHYSICAL energy domain
            E_pred: (B, L) - predicted layer energies (required if use_energy=True)
        
        Returns:
            losses_dict: dictionary of loss values
        """
        losses = {}
        
        if self.use_energy:
            assert E_pred is not None, "E_pred required for energy loss"
            losses= self.energy_loss(x_shape_physical, E_pred)
        
        if self.use_mvn:
            losses = self.mvn_loss(x_shape_physical)
        
        if self.use_cfd:
            losses = self.cfd_loss(x_shape_physical)
        
        return losses
    
    def get_constraint_violations(self, x_shape_physical: torch.Tensor,
                                  E_pred: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Get constraint violations for Augmented Lagrangian
        
        Returns:
            violations_dict: dictionary of violation tensors
        """
        violations = {}
        
        if self.use_energy and E_pred is not None:
            violations['energy'] = self.energy_loss.get_constraint_violation(
                x_shape_physical, E_pred
            )
        
        if self.use_mvn:
            violations['mvn'] = self.mvn_loss.get_constraint_violation(
                x_shape_physical
            )
        
        return violations


if __name__ == "__main__":
    # Example usage
    config = {
        'use_energy_loss': True,
        'use_mvn_loss': True,
        'use_cfd_loss': True,
        'num_layers': 45,
        'energy_loss_type': 'huber',
        'huber_delta': 1.0,
        'mvn_lambda_mean': 1.0,
        'mvn_lambda_cov': 0.5,
        'lambda_cfd': 0.1,
        'cfd_warmup_steps': 2000
    }
    
    aux_manager = AuxiliaryLossManager(config).cuda()
    
    # Dummy data
    x_shape = torch.randn(8, 1, 45, 16, 9).cuda()  # Batch of 8
    E_pred = torch.randn(8, 45).cuda()
    
    # Compute losses
    losses = aux_manager.compute_losses(x_shape, E_pred)
    
    print("Auxiliary Losses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
