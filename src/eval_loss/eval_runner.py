# eval/eval_runner.py
from typing import Dict, Tuple
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .postprocess import reverse_transforms_like_ui
from .metrics import compute_energy_conservation, corr_matrix, cfd_offdiag_fro
#from .classifier import quick_linear_auc
from .plots import (
    plot_layer_means_pdf,
    plot_energy_closure_pdf,
    plot_u_fractions_pdf,
    plot_corr_triptych_pdf,
    #plot_classifier_logits_pdf,
)
from .io_utils import save_metrics_json, save_eval_h5, ensure_outdir

from .diagnostics import (
    _report_basic,
    _energy_closure_report,
    _scale_hint,
)

@torch.no_grad()
def run_comprehensive_eval_noangles(
    *,
    model,                       # your ModelBase (has .device, .params, .transforms, .sample_batch)
    dataset,                     # an instance of CaloChallengeDataset
    n_samples: int,
    batch_size: int,
    sampling_type: str = "ddpm",
    postprocess: bool = True,
    skip_normalize_by_elayer: bool = True,
    outdir: str,                 # .../self.doc.basedir/eval_results
    label: str,                  # self.params['LBL'] or similar
    return_numpy: bool = False   # if True -> return numpy arrays, else CPU torch tensors
):
    """
    Orchestrates the full evaluation without angle information:
      - samples from model with energy-only condition
      - reverses transforms to physics space (for metrics/plots/saves)
      - computes energy closure, layer means, correlations, CFD
      - runs a tiny linear classifier AUC test
      - saves PDF plots, metrics.json, and an HDF5 snapshot

    IMPORTANT: This function DOES NOT return metrics.
               It returns the RAW (non-postprocessed) generated samples and their RAW conditions,
               so you can reuse them later.

    Returns:
        U_gen_raw, E_cond_raw
            If return_numpy=False: torch.FloatTensor on CPU
                - U_gen_raw: [N, 45]
                - E_cond_raw: [N, 1]
            If return_numpy=True: numpy arrays with the same shapes
    """
    ensure_outdir(outdir)

    device = model.device
    model.eval()
    if getattr(model, "net", None) is not None:
        model.net.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # To SAVE/EVALUATE (post-processed)
    U_true_list, U_gen_list = [], []  #U values
    E_true_list, E_gen_list = [], [] #incident energies

    # To RETURN (raw, non-postprocessed)
    U_gen_raw_list, E_raw_list = [], []

    collected = 0

    for layers, energy in loader:
        layers = layers.to(device)                 # [B,45] (preprocessed/model space)
        Ein    = energy.to(device).reshape(-1, 1)  # [B,1]   (preprocessed/model space)

        # Generate in model space (RAW)
        u_hat = model.sample_batch(Ein, sampling_type=sampling_type)  # [B,45] or [B,45,1]
        print("shape of u_hat: ",u_hat.shape)
        if u_hat.ndim == 3 and u_hat.shape[-1] == 1:
            u_hat = u_hat.squeeze(-1)  # -> [B,45]

        # Stash RAW outputs to return later
        U_gen_raw_list.append(u_hat.detach().cpu())
        E_raw_list.append(Ein.detach().cpu())

        # Reverse to physics space for metrics/plots/saving
        if postprocess:
            #generated data
            xg, Eg = reverse_transforms_like_ui(
                model, u_hat, Ein, skip_normalize_by_elayer=skip_normalize_by_elayer
            )
            print("shape of xg and eg in eval_runner: ",xg.shape, Eg.shape)
            #reference data
            xt, Et = reverse_transforms_like_ui(
                model, layers, Ein, skip_normalize_by_elayer=skip_normalize_by_elayer
            )
        else:
            xg, Eg = u_hat, Ein.squeeze(-1)
            xt, Et = layers, Ein.squeeze(-1)

        U_true_list.append(xt.detach().cpu())
        U_gen_list.append(xg.detach().cpu())
        E_true_list.append(Et.detach().cpu())
        E_gen_list.append(Eg.detach().cpu())

        collected += layers.size(0)
        if collected >= n_samples:
            break

    # Stack and trim (post-processed, for saving/metrics)
    U_true = torch.vstack(U_true_list)[:n_samples].numpy()
    U_gen  = torch.vstack(U_gen_list) [:n_samples].numpy()
    E_true = torch.hstack(E_true_list)[:n_samples].numpy()
    E_gen  = torch.hstack(E_gen_list) [:n_samples].numpy()

    # Stack and trim (RAW, for return)
    U_gen_raw = torch.vstack(U_gen_raw_list)[:n_samples]
    E_cond_raw = torch.vstack(E_raw_list)[:n_samples]  # keep as [N,1]
    
    #diagonize issues.....
    _report_basic("U_true (post-processed)", U_true)
    _report_basic("U_gen  (post-processed)", U_gen)
    _report_basic("E_true (post-processed)", E_true)
    _report_basic("E_gen  (post-processed)", E_gen)

    # Check equality / collapse
    try:
        same_U = np.allclose(U_true, U_gen, equal_nan=True)
        print(f"\nU_true vs U_gen allclose: {same_U}")
    except Exception as e:
        print(f"\nU_true vs U_gen allclose check failed: {e}")

    # Energy closure on post-processed
    _energy_closure_report(U_true, E_true, tag="TRUE (post-processed)")
    _energy_closure_report(U_gen,  E_gen,  tag="GEN  (post-processed)")

    # RAW tensors (just shape/finite/zero checks)
    _report_basic("U_gen_raw (RAW)", U_gen_raw)
    _report_basic("E_cond_raw (RAW)", E_cond_raw)

    _scale_hint("U_true", U_true, "E_true", E_true)
    _scale_hint("U_gen",  U_gen,  "E_gen",  E_gen)
    
    # ---------------- Metrics (computed but not returned) ----------------
    metrics: Dict[str, float] = {}

    # Energy conservation (sum layer energies vs incident energy)
    ec_true = compute_energy_conservation(U_true, E_true)
    ec_gen  = compute_energy_conservation(U_gen,  E_true)  # compare gen sums to truth E_inc
    metrics.update({f"energy_conservation_true_{k}": v for k, v in ec_true.items()})
    metrics.update({f"energy_conservation_gen_{k}":  v for k, v in ec_gen.items()})

    # Layer-wise means (for both reporting and plots)
    lm_true = U_true.mean(axis=0)
    lm_gen  = U_gen.mean(axis=0)
    metrics["layer_means_L1_abs"] = float(np.mean(np.abs(lm_true - lm_gen)))
    metrics["layer_means_rel_MAE_percent"] = float(
        np.mean(np.abs(lm_true - lm_gen) / (np.abs(lm_true) + 1e-12)) * 100.0
    )

    # Correlations & CFD
    C_true = corr_matrix(U_true)
    C_gen  = corr_matrix(U_gen)
    diff   = C_true - C_gen
    metrics["CFD_full_fro"]    = float(np.linalg.norm(diff, ord="fro"))
    metrics["CFD_offdiag_fro"] = float(cfd_offdiag_fro(C_true, C_gen))

    # # Quick classifier test (tiny linear model)
    # auc, logits, labels = quick_linear_auc(U_true, U_gen, device)
    # metrics["classifier_AUC_true_vs_gen"] = float(auc)

    # ---------------- Plots ----------------
    ts = _timestamp()
    plot_layer_means_pdf(lm_true, lm_gen, outdir, label, ts)
    plot_energy_closure_pdf(U_true, U_gen, E_true, outdir, label, ts)
    plot_u_fractions_pdf(U_true, U_gen, outdir, label, ts)
    plot_corr_triptych_pdf(C_true, C_gen, outdir, label, ts, metrics)
    #plot_classifier_logits_pdf(logits, labels, outdir, label, ts, auc)

    # ---------------- Save artifacts ----------------
    save_metrics_json(metrics, outdir, ts)
    save_eval_h5(
        U_true, U_gen, E_true, E_gen,
        lm_true, lm_gen, C_true, C_gen,
        outdir, ts, sampling_type, model
    )

    # ---------------- Return RAW tensors/arrays ----------------
    if return_numpy:
        return U_gen_raw.numpy(), E_cond_raw.numpy()
    else:
        return U_gen_raw, E_cond_raw


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
