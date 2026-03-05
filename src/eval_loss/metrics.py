# eval/metrics.py
from __future__ import annotations
import numpy as np

_EPS = 1e-6  # f32-friendly epsilon

def compute_energy_conservation(U: np.ndarray, E: np.ndarray) -> dict:
    """
    Energy closure & error metrics (float32-safe).
      U : [N, 45] per-layer energies (post-processed/physics units)
      E : [N]      incident energies  (post-processed/physics units)
    """
    U = np.asarray(U, dtype=np.float32)
    E = np.asarray(E, dtype=np.float32)

    sumU = U.sum(axis=1).astype(np.float32)
    denom = (E + _EPS).astype(np.float32)

    ratio = (sumU + _EPS) / denom
    mae   = np.mean(np.abs(sumU - E), dtype=np.float32)
    mre   = np.mean(np.abs(sumU - E) / denom, dtype=np.float32) * np.float32(100.0)

    return {
        "mean_ratio":  float(np.mean(ratio, dtype=np.float32)),
        "std_ratio":   float(np.std(ratio,  dtype=np.float32)),
        "mae_abs_diff": float(mae),
        "mre_percent":  float(mre),
    }

def corr_matrix(X: np.ndarray) -> np.ndarray:
    """
    Pearson correlation (float32) with guards.
      X : [N, D]
      returns C : [D, D] in [-1,1] (float32)
    """
    X = np.asarray(X, dtype=np.float32)
    Xc = X - X.mean(axis=0, keepdims=True, dtype=np.float32)
    std = Xc.std(axis=0, keepdims=True, dtype=np.float32)
    std = np.maximum(std, _EPS).astype(np.float32)

    Z = (Xc / std).astype(np.float32)
    n = max(int(Z.shape[0] - 1), 1)
    C = (Z.T @ Z) / np.float32(n)
    return np.clip(C.astype(np.float32), -1.0, 1.0)
def cfd_offdiag_fro(C_true: np.ndarray, C_gen: np.ndarray) -> float:
    """
    Frobenius norm over off-diagonal entries:
      sqrt( sum_{i≠j} (C_true[i,j] - C_gen[i,j])^2 )
    """
    C_true = np.asarray(C_true, dtype=np.float32)
    C_gen  = np.asarray(C_gen,  dtype=np.float32)
    assert C_true.shape == C_gen.shape and C_true.ndim == 2, "corr shapes must match"
    D = C_true.shape[0]
    off = ~np.eye(D, dtype=bool)
    diff_vec = (C_true - C_gen)[off].astype(np.float32)  # 1D vector of off-diagonal elements
    return float(np.linalg.norm(diff_vec))  # L2 over vector == Frobenius over off-diagonals

