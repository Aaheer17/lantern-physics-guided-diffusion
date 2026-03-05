import numpy as np
import torch

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _finite_mask(arr):
    return np.isfinite(arr)

def _percentiles(arr, qs=(0,1,5,50,95,99,100)):
    arrf = arr[np.isfinite(arr)]
    if arrf.size == 0:
        return {q: None for q in qs}
    return {q: float(np.percentile(arrf, q)) for q in qs}

def _report_basic(name, arr, near_eps=1e-8):
    arr = _to_numpy(arr)
    print(f"\n=== {name} ===")
    print(f"shape={arr.shape}, dtype={arr.dtype}")

    nan_cnt = np.isnan(arr).sum()
    inf_cnt = np.isinf(arr).sum()
    fin_mask = _finite_mask(arr)
    fin_frac = float(fin_mask.mean())
    print(f"NaNs={int(nan_cnt)}, Infs={int(inf_cnt)}, finite_frac={fin_frac:.6f}")

    if fin_mask.any():
        arrf = arr[fin_mask]
        zero_cnt = int((arrf == 0).sum())
        zero_frac = zero_cnt / arrf.size
        # Near-zero threshold is absolute; also report relative near-zero vs max|x|
        abs_thresh = near_eps
        rel_thresh = near_eps * max(float(np.max(np.abs(arrf))), 1.0)
        near_zero_abs = int((np.abs(arrf) < abs_thresh).sum())
        near_zero_rel = int((np.abs(arrf) < rel_thresh).sum())
        print(f"min={arrf.min():.6g}, max={arrf.max():.6g}, mean={arrf.mean():.6g}, std={arrf.std():.6g}")
        print(f"exact_zero: {zero_cnt} ({zero_frac:.6%}) | near_zero_abs<{abs_thresh}: {near_zero_abs} | near_zero_rel<{rel_thresh:.3g}: {near_zero_rel}")
        p = _percentiles(arrf)
        print("percentiles:", ", ".join([f"{k}%={v:.6g}" if v is not None else f"{k}%=None" for k,v in p.items()]))
    else:
        print("All values are non-finite; skipping stats.")

def _energy_closure_report(U, E, tag, eps=1e-6):
    U = _to_numpy(U)
    E = _to_numpy(E).reshape(-1)
    print(f"\n--- Energy closure ({tag}) ---")
    ok = _finite_mask(U).all() and _finite_mask(E).all()
    if not ok:
        print("Non-finite values present in U or E; closure stats may be invalid.")
    if U.ndim != 2:
        print(f"WARNING: expected U to be 2D [N,45], got {U.shape}")
    if E.ndim != 1:
        print(f"WARNING: expected E to be 1D [N], got {E.shape}")

    N = min(U.shape[0], E.shape[0]) if U.ndim >= 1 and E.ndim == 1 else 0
    if N == 0:
        print("Empty or mismatched shapes; cannot compute.")
        return

    sumU = U[:N].sum(axis=1)
    denom = E[:N] + eps
    ratio = (sumU + eps) / denom
    mae = np.mean(np.abs(sumU - E[:N]))
    mre = np.mean(np.abs(sumU - E[:N]) / denom) * 100.0

    def _summ(v):
        vf = v[np.isfinite(v)]
        if vf.size == 0:
            return "all non-finite"
        return f"mean={vf.mean():.6g}, std={vf.std():.6g}, min={vf.min():.6g}, max={vf.max():.6g}"

    print(f"E stats: {_summ(E[:N])}")
    print(f"sum(U) stats: {_summ(sumU)}")
    print(f"ratio=sum(U)/E -> mean={np.mean(ratio):.6g}, std={np.std(ratio):.6g}")
    print(f"MAE |sum(U)-E| = {mae:.6g}")
    print(f"MRE% = {mre:.6g}%")

# ---- Place this right after your stacking lines ----
# U_true, U_gen, E_true, E_gen are numpy arrays per your snippet.
# U_gen_raw, E_cond_raw are torch tensors (RAW).

# _report_basic("U_true (post-processed)", U_true)
# _report_basic("U_gen  (post-processed)", U_gen)
# _report_basic("E_true (post-processed)", E_true)
# _report_basic("E_gen  (post-processed)", E_gen)

# # Check equality / collapse
# try:
#     same_U = np.allclose(U_true, U_gen, equal_nan=True)
#     print(f"\nU_true vs U_gen allclose: {same_U}")
# except Exception as e:
#     print(f"\nU_true vs U_gen allclose check failed: {e}")

# # Energy closure on post-processed
# _energy_closure_report(U_true, E_true, tag="TRUE (post-processed)")
# _energy_closure_report(U_gen,  E_gen,  tag="GEN  (post-processed)")

# # RAW tensors (just shape/finite/zero checks)
# _report_basic("U_gen_raw (RAW)", U_gen_raw)
# _report_basic("E_cond_raw (RAW)", E_cond_raw)

# Optional: flag suspicious scale gaps (e.g., E ~ 1e5 while U ~ 0)
def _scale_hint(name_u, U, name_e, E, eps=1e-12):
    U = _to_numpy(U); E = _to_numpy(E).reshape(-1)
    if U.ndim >= 2 and E.ndim == 1 and min(len(U), len(E)) > 0:
        su = np.abs(U[:len(E)]).mean()
        se = np.abs(E[:len(U)]).mean()
        if np.isfinite(su) and np.isfinite(se):
            ratio = (su + eps) / (se + eps)
            print(f"\n[Scale hint] mean(|{name_u}|) / mean(|{name_e}|) = {ratio:.6g}")
            if ratio < 1e-6:
                print("  -> U is ~zero relative to E (likely inverse-transform/units mismatch).")
            elif ratio > 1e6:
                print("  -> U is huge relative to E (possible scaling error).")

# _scale_hint("U_true", U_true, "E_true", E_true)
# _scale_hint("U_gen",  U_gen,  "E_gen",  E_gen)
