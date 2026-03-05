# eval/plots.py
"""
Matplotlib PDF plots for evaluation.

All functions:
- accept NumPy arrays (will cast to float32)
- are robust to small NaNs/Infs via nan_to_num
- save a single PDF per figure into `outdir`
"""

from __future__ import annotations
import os
import numpy as np

# Headless-safe: don't error if no display
try:
    import matplotlib
    if "backend" not in matplotlib.get_backend().lower():
        matplotlib.use("Agg")  # safe for servers
except Exception:
    pass

import matplotlib.pyplot as plt


def _f32(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_bins(arr: np.ndarray, nbins: int = 60, qlow: float = 0.01, qhigh: float = 0.99):
    arr = _f32(arr).ravel()
    lo, hi = np.quantile(arr, [qlow, qhigh]) if arr.size > 0 else (0.0, 1.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(arr, initial=0.0)), float(np.max(arr, initial=1.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
    return np.linspace(lo, hi, nbins, dtype=np.float32)


def plot_layer_means_pdf(
    lm_true: np.ndarray,
    lm_gen: np.ndarray,
    outdir: str,
    label: str,
    ts: str,
):
    """Line plot of per-layer mean energies (post-processed)."""
    lm_true = _f32(lm_true)
    lm_gen  = _f32(lm_gen)

    plt.figure(figsize=(10, 4))
    plt.plot(lm_true, label="Truth", linewidth=2)
    plt.plot(lm_gen,  label="Generated", linewidth=2)
    plt.xlabel("Layer index (0..44)")
    plt.ylabel("Mean energy")
    plt.title(f"Layer-wise mean energies [{label}]")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, f"layer_means_{ts}.pdf")
    plt.savefig(path)
    plt.close()


def plot_energy_closure_pdf(
    U_true: np.ndarray,
    U_gen: np.ndarray,
    E_true: np.ndarray,
    outdir: str,
    label: str,
    ts: str,
):
    """Histogram of closure ratio sum(E_i)/E_inc for truth vs generated."""
    U_true = _f32(U_true)
    U_gen  = _f32(U_gen)
    E_true = _f32(E_true)

    r_true = (U_true.sum(axis=1) + 1e-6) / (E_true + 1e-6)
    r_gen  = (U_gen.sum(axis=1)  + 1e-6) / (E_true + 1e-6)

    bins = _safe_bins(np.concatenate([r_true, r_gen], axis=0), nbins=60, qlow=0.01, qhigh=0.99)

    plt.figure(figsize=(6, 4))
    plt.hist(r_true, bins=bins, alpha=0.6, density=True, label="Truth")
    plt.hist(r_gen,  bins=bins, alpha=0.6, density=True, label="Generated")
    plt.xlabel("sum(layer energies) / E_inc")
    plt.ylabel("Density")
    plt.title(f"Energy closure ratio [{label}]")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, f"energy_closure_{ts}.pdf")
    plt.savefig(path)
    plt.close()


def plot_u_fractions_pdf(
    U_true: np.ndarray,
    U_gen: np.ndarray,
    outdir: str,
    label: str,
    ts: str,
):
    """Line plot of average per-layer fractions u_i."""
    U_true = _f32(U_true)
    U_gen  = _f32(U_gen)

    denom_t = U_true.sum(axis=1, keepdims=True) + 1e-6
    denom_g = U_gen.sum(axis=1,  keepdims=True) + 1e-6
    ut = (U_true / denom_t).mean(axis=0)
    ug = (U_gen  / denom_g).mean(axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(ut, label="Truth u_i", linewidth=2)
    plt.plot(ug, label="Gen u_i", linewidth=2)
    plt.xlabel("Layer index (0..44)")
    plt.ylabel("Average fraction per layer")
    plt.title(f"Average layer fractions u_i [{label}]")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, f"u_fractions_{ts}.pdf")
    plt.savefig(path)
    plt.close()


def plot_corr_triptych_pdf(
    C_true: np.ndarray,
    C_gen: np.ndarray,
    outdir: str,
    label: str,
    ts: str,
    metrics: dict | None = None,
):
    """Three-panel correlation heatmaps: truth, generated, and difference."""
    C_true = _f32(C_true)
    C_gen  = _f32(C_gen)
    C_diff = _f32(C_gen - C_true)

    vmax = 1.0
    vmin = -1.0

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axs[0].imshow(C_true, vmin=vmin, vmax=vmax)
    axs[0].set_title("Corr (Truth)")
    im1 = axs[1].imshow(C_gen,  vmin=vmin, vmax=vmax)
    axs[1].set_title("Corr (Generated)")
    im2 = axs[2].imshow(C_diff, vmin=-1.0, vmax=1.0)
    axs[2].set_title("Corr diff (Gen - Truth)")

    for ax in axs:
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")

    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    title = f"Layer-wise correlations [{label}]"
    if metrics:
        cf = metrics.get("CFD_full_fro", None)
        cd = metrics.get("CFD_offdiag_fro", None)
        if cf is not None or cd is not None:
            parts = []
            if cf is not None: parts.append(f"CFD_fro={cf:.4f}")
            if cd is not None: parts.append(f"offdiag={cd:.4f}")
            title += "\n" + ", ".join(parts)

    fig.suptitle(title)
    plt.tight_layout()
    path = os.path.join(outdir, f"correlations_{ts}.pdf")
    plt.savefig(path)
    plt.close()


def plot_classifier_logits_pdf(
    logits: np.ndarray,
    labels: np.ndarray,
    outdir: str,
    label: str,
    ts: str,
    auc_value: float,
):
    """Histogram of classifier logits for truth vs generated."""
    logits = _f32(logits)
    labels = np.asarray(labels).astype(np.int32)

    # robust binning on logits
    bins = _safe_bins(logits, nbins=60, qlow=0.01, qhigh=0.99)

    plt.figure(figsize=(6, 4))
    plt.hist(logits[labels == 1], bins=bins, alpha=0.6, density=True, label="Truth logits")
    plt.hist(logits[labels == 0], bins=bins, alpha=0.6, density=True, label="Gen logits")
    plt.xlabel("Classifier logits")
    plt.ylabel("Density")
    plt.title(f"Separation (AUC={auc_value:.3f}) [{label}]")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, f"classifier_logits_{ts}.pdf")
    plt.savefig(path)
    plt.close()
