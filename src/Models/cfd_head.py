# cfd_head.py
# Layer-correlation auxiliary loss head for DS2 (CaloChallenge 2022)
# Works directly from processed (B,45,1) predictions — no voxel data required.

from __future__ import annotations
from typing import Optional, Union
import os
import torch

# Must match your dataset pipeline
from transforms import (
    Reshape, ScaleEnergy, LogEnergy,
    StandardizeFromFile, ExclusiveLogitTransform,ScaleTotalEnergy
)



# cfd_head.py (append near your CFD code)
import torch
from dataclasses import dataclass

@dataclass
class MVNStats:
    mu: torch.Tensor         # (45,)
    L: torch.Tensor          # (45,45) Cholesky of Sigma (Sigma = L L^T)
    inv_logdet: torch.Tensor # scalar tensor with -log|Sigma| for logging if you want

def _load_mvn_stats(npz_path: str, device: torch.device) -> MVNStats:
    import numpy as np
    z = np.load(npz_path)
    mu = torch.tensor(z["mu"], dtype=torch.float32, device=device)               # (45,)
    Sigma = torch.tensor(z["Sigma"], dtype=torch.float32, device=device)         # (45,45)
    # tiny jitter for numerical safety
    eps = 1e-6 * torch.mean(torch.diag(Sigma))
    Sigma_j = Sigma + eps * torch.eye(Sigma.shape[0], device=device)
    L = torch.linalg.cholesky(Sigma_j)                                           # (45,45)
    inv_logdet = -2.0 * torch.sum(torch.log(torch.diag(L)))                      # since |Sigma| = (prod diag(L))^2
    return MVNStats(mu=mu, L=L, inv_logdet=inv_logdet)
def neg_stats(x, e, tag=""):
    xa = _np(x).reshape(-1)
    pct = 100.0 * (xa < 0).sum() / max(1, xa.size)
    print(f"{tag} negatives in features: {pct:.6f}%")
class MVNHead(torch.nn.Module):
    """
    Provides two penalties:
      - batch-mean (MCLT) term:  (B/2) * (mean - mu)^T Sigma^{-1} (mean - mu)
      - per-sample MVN term:     (1/2) * mean_i (y_i - mu)^T Sigma^{-1} (y_i - mu)
    Uses Cholesky solves; never inverts Sigma explicitly.
    """
    def __init__(self, stats_path_or_dict, device, mode="mean", lambda_weight=1e-3):
        super().__init__()
        self.device = device
        self.mode   = mode          # "mean" or "sample"
        self.weight = lambda_weight

        if isinstance(stats_path_or_dict, str):
            self.stats = _load_mvn_stats(stats_path_or_dict, device)
            # expect stats.mu (45,), stats.L (45x45 chol of Sigma+epsI)
        else:
            mu = torch.as_tensor(stats_path_or_dict["mu"], dtype=torch.float32, device=device)
            Sigma = torch.as_tensor(stats_path_or_dict["Sigma"], dtype=torch.float32, device=device)
            eps = 1e-6 * torch.mean(torch.diag(Sigma))
            L = torch.linalg.cholesky(Sigma + eps * torch.eye(Sigma.shape[0], device=device))
            inv_logdet = -2.0 * torch.sum(torch.log(torch.diag(L)))
            self.stats = MVNStats(mu=mu, L=L, inv_logdet=inv_logdet)

        # Optional: cache Sigma^{-1} (not needed, but sometimes handy for logging)
        with torch.no_grad():
            self.Sigma_inv = torch.cholesky_inverse(self.stats.L)  # constant w.r.t. model

    @torch.no_grad()
    def _mahalanobis_no_grad(self, diff: torch.Tensor) -> torch.Tensor:
        y = torch.cholesky_solve(diff.unsqueeze(-1), self.stats.L)
        return (diff.unsqueeze(-2) @ y).squeeze(-1).squeeze(-1)

    def _mahalanobis(self, diff: torch.Tensor) -> torch.Tensor:
        y = torch.cholesky_solve(diff.unsqueeze(-1), self.stats.L)
        return (diff.unsqueeze(-2) @ y).squeeze(-1).squeeze(-1)

    def forward(self, vecs_45: torch.Tensor) -> torch.Tensor:
        """
        vecs_45: [B,45]  (per-sample layer-sum vector)
        Returns: scalar MVN/MCLT penalty
        """
        mu = self.stats.mu
        B  = vecs_45.shape[0]

        if self.mode == "mean":
            mean_b = vecs_45.mean(dim=0)
            diff   = mean_b - mu
            md2    = self._mahalanobis(diff)      # scalar
            loss   = 0.5 * B * md2
        elif self.mode == "sample":
            diff   = vecs_45 - mu                  # [B,45]
            md2    = self._mahalanobis(diff)       # [B]
            loss   = 0.5 * md2.mean()
        else:
            raise ValueError("mode must be 'mean' or 'sample'")

        return self.weight * loss

    def trace_penalty(self, vecs_45: torch.Tensor, eta: float = 0.0,
                      unit_sum: bool = False) -> torch.Tensor:
        """
        Lightweight covariance alignment: 0.5 * eta * tr(Sigma^{-1} S_batch).
        vecs_45: [B,45] in the SAME space used for (mu, Sigma).
        Returns a scalar (differentiable w.r.t. vecs_45).
        """
        if eta is None or eta <= 0.0:
            # Return a zero tensor on the right device/dtype to keep graphs tidy
            return vecs_45.new_zeros((), dtype=vecs_45.dtype)

        Y = vecs_45
        if unit_sum:
            denom = Y.sum(dim=1, keepdim=True).clamp_min(1e-12)
            Y = Y / denom

        B = Y.shape[0]
        if B <= 1:
            # avoid division by zero; penalize nothing for degenerate batch
            return vecs_45.new_zeros((), dtype=vecs_45.dtype)

        Yc = Y - Y.mean(dim=0, keepdim=True)                # [B,45]
        S_batch = (Yc.transpose(0, 1) @ Yc) / float(B - 1)  # [45,45]

        # Compute Sigma^{-1} S_batch via two triangular solves (stable, differentiable)
        L = self.stats.L                                    # [45,45], lower-tri
        X = torch.linalg.solve_triangular(L, S_batch, upper=False)              # solve L X = S
        Ymat = torch.linalg.solve_triangular(L.transpose(-1, -2), X, upper=True)  # solve L^T Y = X
        tr_term = torch.trace(Ymat)                         # tr(Sigma^{-1} S_batch)

        return 0.5 * eta * tr_term

class CFDHead:
    """
    Compute a correlation-based loss on *layer energies* for DS2, starting from
    x0_hat in *processed* space (B,45,1) and processed incident energies.

    Pipeline (reverse only the last 5 transforms):
        x_proc (B,45,1) --Reshape^-1--> (B,45)
                            --ScaleEnergy^-1--> raw-log energy
                            --LogEnergy^-1--> raw energy
                            --StandardizeFromFile^-1--> unstandardized u
                            --ExclusiveLogitTransform^-1--> u in (delta,1-delta)

    Then reconstruct per-layer energies with the same stick-breaking used in
    NormalizeByElayer.rev, compute (45×45) correlation across the batch, and
    return off-diagonal MSE vs C_real.

    Notes:
    - This head assumes DS2 (no angle conditions).
    - You MUST pass the same delta/e_min/e_max/stats_dir used in your dataset.
    - stats_dir should already contain means.npy/stds.npy computed from TRAIN ONLY.
    """

    def __init__(
        self,
        *,
        stats_dir: str,
        delta: float,
        e_min: float,
        e_max: float,
        C_real: Union[torch.Tensor, str],
        device: Optional[torch.device] = None,
        n_layers: int = 45,
        require_stats: bool = True,
        # NEW:
        ema_beta = 0.0,     # e.g., 0.9 or 0.99; None disables EMA
        shrink_alpha: float = 0.0,            # 0 disables shrinkage; try 0.05–0.15
        shrink_target: Literal["identity", "creal"] = "identity",
    ):
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_layers = int(n_layers)

        if require_stats:
            mean_p = os.path.join(stats_dir, "means.npy")
            std_p  = os.path.join(stats_dir, "stds.npy")
            if not (os.path.exists(mean_p) and os.path.exists(std_p)):
                raise RuntimeError(
                    f"StandardizeFromFile stats not found in '{stats_dir}'. "
                    "Create them from the TRAIN split before using CFDHead."
                )

        # Build the *reverse* chain's forward instances with identical params
        self.inv_fns = [
            Reshape(shape=[self.n_layers, 1]),
            ScaleEnergy(e_min=e_min, e_max=e_max),
            LogEnergy(),
            StandardizeFromFile(model_dir=stats_dir),
            ExclusiveLogitTransform(delta=delta, rescale=True),
            ScaleTotalEnergy(factor=0.35)
        ]
        
        self.ema_beta = ema_beta if ema_beta is not 0.0 else 0.0
        self.shrink_alpha = float(shrink_alpha)
        assert 0.0 <= self.shrink_alpha < 1.0, "shrink_alpha must be in [0,1)"
        assert shrink_target in ("identity", "creal")
        self.shrink_target = shrink_target

        # Load target correlation
        if isinstance(C_real, str):
            C_real_t = torch.tensor(
                __import__("numpy").load(C_real),
                dtype=torch.float32, device=self.device
            )
        else:
            C_real_t = C_real.to(device=self.device, dtype=torch.float32)
        assert C_real_t.shape == (self.n_layers, self.n_layers), \
            f"C_real must be {(self.n_layers, self.n_layers)}, got {tuple(C_real_t.shape)}"
        self.C_real = C_real_t

        eye = torch.eye(self.n_layers, dtype=torch.bool, device=self.device)
        self.offdiag = ~eye
        # --- EMA state (buffer, no grad) ---
        self.C_ema = None   # lazily initialized on first call
    # ------------- helper: shrinkage of a correlation matrix -------------
    def _shrink_corr(self, C: torch.Tensor) -> torch.Tensor:
        """
        Linear shrinkage: C_shrunk = (1 - alpha) * C + alpha * T
          where T = I (identity) or C_real
        """
        if self.shrink_alpha <= 0.0:
            return C
        if self.shrink_target == "identity":
            T = torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
        else:  # 'creal'
            T = self.C_real.to(device=C.device, dtype=C.dtype)
        return (1.0 - self.shrink_alpha) * C + self.shrink_alpha * T


    # -------- internal helpers (keep differentiable; no .no_grad, no numpy) --------

    def _reverse_to_u(self, x_proc: torch.Tensor, e_proc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Undo ONLY: Reshape → ScaleEnergy → LogEnergy → StandardizeFromFile → ExclusiveLogitTransform.
        Returns:
          u_raw: (B,45)   # stick-breaking params in raw scale
          e_raw: (B,)     # raw incident energy (GeV)
        """
        layers, energy = x_proc, e_proc
        for fn in (self.inv_fns):
            name = getattr(fn, "__class__", type(fn)).__name__
            #print(f"==== Transformation:  {name} ===")
            layers, energy = fn(layers, energy, rev=True)
        if layers.ndim == 3 and layers.shape[-1] == 1:
            layers = layers.squeeze(-1)
        if energy.ndim > 1:
            energy = energy.view(-1)
        return layers, energy

    @staticmethod
    def _u_to_layerE(u: torch.Tensor, e_inc: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct per-layer energies via stick-breaking (matches NormalizeByElayer.rev):
           total_E = e_inc * u0
           layer[i] = (total_E - sum_{k<i} layer[k]) * u_{i+1} for i=0..L-2
           layer[L-1] = total_E - sum_{k<L-1} layer[k]
        u:     (B,L) in (delta, 1-delta)
        e_inc: (B,)
        Returns: (B,L)
        """
        B, L = u.shape
        #print("shape of B: ",B, e_inc.shape)
        assert e_inc.shape[0] == B
        # Clamp u_{i>0} to [0,1] to mirror reverse logic and avoid numeric overshoot
        u = u.clone()
        if L > 1:
            u[:, 1:] = u[:, 1:].clamp_(0.0, 1.0)
        total_E = e_inc.view(-1) * u[:, 0]
        cum = torch.zeros_like(total_E)
        layers = []
        for i in range(L - 1):
            Ei = (total_E - cum) * u[:, i + 1]
            layers.append(Ei)
            cum = cum + Ei
        layers.append(total_E - cum)
        return torch.stack(layers, dim=1)

    @staticmethod
    def _corr(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Correlation across the batch (features D along dim=1).
        X: (B,D)
        Returns: (D,D)
        """
        Xc  = X - X.mean(dim=0, keepdim=True)
        cov = (Xc.transpose(0, 1) @ Xc) / (Xc.shape[0] - 1 + eps)
        var = cov.diag().clamp_min(eps)
        inv = var.rsqrt()
        return (cov * (inv[:, None] * inv[None, :])).clamp(-1.0, 1.0)

    # -------- public API --------
    def loss(self, x0_hat_proc: torch.Tensor, e_proc: torch.Tensor) -> torch.Tensor:
        """
        x0_hat_proc: (B,45,1) predicted x_0 in processed space
        e_proc:      (B,) or (B,1) processed incident energies
        Returns: scalar torch.Tensor (CFD loss)
        """
        B = x0_hat_proc.size(0)
        if B < 3:
            return x0_hat_proc.new_tensor(0.0)

        dev = x0_hat_proc.device
        x0_hat_proc = x0_hat_proc.to(dev)
        e_proc      = e_proc.to(dev)

        # 1) Invert last transforms
        u_raw, e_raw = self._reverse_to_u(x0_hat_proc, e_proc)  # (B,L), (B,)

        # 2) Per-layer energies
        L = self._u_to_layerE(u_raw, e_raw)                     # (B,L)

        # 3) Batch correlation
        C_pred = self._corr(L)                                  # (L,L)

        # 4) Shrinkage
        C_work = self._shrink_corr(C_pred)

        # helper uses mask on correct device
        def _offdiag_fro(A: torch.Tensor) -> torch.Tensor:
            offmask = self.offdiag.to(device=A.device)
            return A[offmask].pow(2).sum().sqrt()

        C_real = self.C_real.to(device=dev, dtype=C_work.dtype)
        fro_pred   = _offdiag_fro(C_pred - C_real)
        fro_shrunk = _offdiag_fro(C_work - C_real)

        # 5) EMA smoothing
        if self.ema_beta is not None:
            if self.C_ema is None:
                self.C_ema = C_work.detach()
            ema_dist = _offdiag_fro(C_work - self.C_ema)
            C_smooth = (1.0 - self.ema_beta) * C_work + self.ema_beta * self.C_ema
            self.C_ema = C_smooth.detach()
        else:
            ema_dist = torch.tensor(float('nan'), device=dev, dtype=C_work.dtype)
            C_smooth = C_work

        fro_smooth = _offdiag_fro(C_smooth - C_real)

        # 6) Off-diagonal MSE vs target (mask on correct device)
        offmask = self.offdiag.to(device=dev)
        diff = (C_smooth - C_real)
        loss = diff[offmask].pow(2).mean()

        # Debug stash
        self.last_debug = {
            "cfd_fro_pred":   float(fro_pred.detach().item()),
            "cfd_fro_shrunk": float(fro_shrunk.detach().item()),
            "cfd_fro_smooth": float(fro_smooth.detach().item()),
            "cfd_ema_dist":   float(ema_dist.detach().item()),
        }
        return loss

    @torch.no_grad()
    def metrics_from_model(
        self,
        x0_hat_proc: torch.Tensor,
        e_proc: torch.Tensor,
        normalize_by_real: bool = True,
        offdiag_only: bool = True,
        ) -> dict:
        """
        Compute CFD-style metrics for logging, mirroring the loss path:
          - accepts model-space (processed) x̂0 (B,45,1) and processed energies,
          - inverts to raw stick-breaking params & energy,
          - reconstructs per-layer energies, computes batch PCC (45x45),
          - returns Frobenius metrics (normalized and raw).
        """
        B = x0_hat_proc.size(0)
        if B < 3:
            return {
                "CFD_full_fro": float("nan"),
                "CFD_offdiag_fro": float("nan"),
                "fro_raw_full": float("nan"),
                "fro_raw_off": float("nan"),
            }
    
        dev = x0_hat_proc.device
        # 1) Invert last transforms (reuse same path as in loss)
        u_raw, e_raw = self._reverse_to_u(x0_hat_proc.to(dev), e_proc.to(dev))  # (B,L), (B,)
    
        # 2) Per-layer energies and PCC
        L = self._u_to_layerE(u_raw, e_raw)  # (B,L)
        C_pred = self._corr(L)               # (L,L)
    
        # 3) Build difference vs. C_real
        Cg = C_pred
        Cr = self.C_real.to(device=dev, dtype=Cg.dtype)
        D  = Cg - Cr
    
        # Build off-diagonal mask (before norms)
        eye = torch.eye(Cg.shape[0], device=dev, dtype=Cg.dtype)
        off = (1.0 - eye)
        
        # Raw (unnormalized) Frobenius norms for reference/logging
        fro_raw_full = torch.linalg.norm(D,        ord="fro")
        fro_raw_off  = torch.linalg.norm(D * off,  ord="fro")
        
        if normalize_by_real:
            # Normalization denominators from C_real (RAW-space), matching your qual definition
            denom_off  = torch.linalg.norm(Cr * off, ord="fro") + 1e-8
            denom_full = torch.linalg.norm(Cr,       ord="fro") + 1e-8  # optional, for a full-matrix normalized readout
        
            # Normalized CFD metrics
            offdiag_fro = fro_raw_off  / denom_off
            full_fro    = fro_raw_full / denom_full   # optional; delete if you want only off-diag normalized
        else:
            # No normalization: return raw Frobenius norms in the CFD_* slots
            offdiag_fro = fro_raw_off
            full_fro    = fro_raw_full
        
        return {
            "CFD_full_fro":    float(full_fro.detach().cpu().item()),
            "CFD_offdiag_fro": float(offdiag_fro.detach().cpu().item()),
            "fro_raw_full":    float(fro_raw_full.detach().cpu().item()),
            "fro_raw_off":     float(fro_raw_off.detach().cpu().item()),
        }
            
            
        
