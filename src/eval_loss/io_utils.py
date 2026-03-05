# eval/io_utils.py
from __future__ import annotations
import os, json, time
import h5py
import numpy as np


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_metrics_json(metrics: dict, outdir: str, ts: str) -> str:
    """
    Save metrics as a pretty JSON and return the file path.
    """
    path = os.path.join(outdir, f"metrics_{ts}.json")
    # Convert any np types to native Python
    def _py(v):
        if isinstance(v, (np.floating, np.float32, np.float64)): return float(v)
        if isinstance(v, (np.integer,  np.int32,  np.int64)):    return int(v)
        return v
    metrics_py = {k: _py(v) for k, v in metrics.items()}
    with open(path, "w") as f:
        json.dump(metrics_py, f, indent=2)
    return path


def _to_np32(x):
    arr = np.asarray(x)
    return arr.astype(np.float32, copy=False) if arr.dtype.kind == "f" else arr


def _write_ds(g, name: str, arr: np.ndarray) -> None:
    arr = _to_np32(arr)
    maxshape = (None,) + arr.shape[1:]
    g.create_dataset(
        name,
        data=arr,
        compression="gzip",
        shuffle=True,
        chunks=True,
        maxshape=maxshape,
    )


def save_eval_h5(
    U_true: np.ndarray,
    U_gen:  np.ndarray,
    E_true: np.ndarray,
    E_gen:  np.ndarray,
    lm_true: np.ndarray,
    lm_gen:  np.ndarray,
    C_true:  np.ndarray,
    C_gen:   np.ndarray,
    outdir: str,
    ts: str,
    sampling_type: str,
    model,                    # for attrs (epoch, traintime, params)
) -> str:
    """
    Store post-processed arrays and summaries in one HDF5 for later analysis.
    """
    path = os.path.join(outdir, f"lantern_eval_noangles_{ts}.h5")
    with h5py.File(path, "w") as f:
        g_gen  = f.create_group("gen")
        g_true = f.create_group("true")
        g_cond = f.create_group("conditions")

        _write_ds(g_gen,  "U", U_gen)
        _write_ds(g_true, "U", U_true)
        _write_ds(g_cond, "E_inc_true", E_true)
        _write_ds(g_cond, "E_inc_gen",  E_gen)

        _write_ds(f, "layer_means_true", lm_true[None, :])
        _write_ds(f, "layer_means_gen",  lm_gen[None, :])
        _write_ds(f, "corr_true", C_true[None, :, :])
        _write_ds(f, "corr_gen",  C_gen[None, :, :])

        # Metadata
        f.attrs["note"]                 = "Lantern comprehensive_eval_noangles output"
        f.attrs["created_utc"]          = time.strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["n_samples"]            = int(U_gen.shape[0])
        f.attrs["features_per_sample"]  = int(U_gen.shape[1]) if U_gen.ndim > 1 else 1
        f.attrs["sampling_type"]        = str(sampling_type)
        f.attrs["units_energy"]         = "GeV (assumed)"

        # Optional model attrs
        f.attrs["model_name_or_doc"]    = str(getattr(model, "doc", ""))
        f.attrs["epoch"]                = int(getattr(model, "epoch", -1))
        f.attrs["train_time_hours"]     = float(getattr(model, "traintime", 0.0) or 0.0)

        # Params as JSON if serializable
        try:
            f.attrs["params_json"] = json.dumps(getattr(model, "params", {}), default=str)
        except Exception:
            pass

    return path
