# eval/postprocess.py
from __future__ import annotations
from typing import Tuple
import torch


@torch.no_grad()
def reverse_transforms_like_ui(
    model,
    x: torch.Tensor,                 # [B,45] or [B,45,1] in model/preprocessed space
    E_scalar: torch.Tensor,          # [B,1] or [B] in model/preprocessed space
    skip_normalize_by_elayer: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reverse your preprocessing to physics space, mirroring yesterday's pipeline.
    If `model` exposes `inverse_transform_to_raw(samples, conditions)`, we use that.
    Otherwise, we walk `model.transforms` backward and handle scalar-energy-aware transforms.

    Returns:
        x_pp  : [B,45] physics-space layer energies
        E_out : [B]    physics-space incident energy
    """
    print("******** Am I even here?? ************")
    # 0) Preferred path: let your own implementation handle it (cfd_head.py based).
    # if hasattr(model, "inverse_transform_to_raw") and callable(model.inverse_transform_to_raw):
    #     print("I am here?? really!")
    #     # NOTE: model.inverse_transform_to_raw may expect shapes:
    #     #   samples: [B,45] or [B,45,1]; conditions: [B,1]
    #     samples = x
    #     conds   = E_scalar if E_scalar.ndim == 2 else E_scalar.reshape(-1, 1)
    #     out = model.inverse_transform_to_raw(samples, conds)
    #     # Allow both tuple or single return (x_pp or (x_pp, maybe_extra))
    #     if isinstance(out, tuple):
    #         x_pp = out[0]
    #     else:
    #         x_pp = out
    #     x_pp = _ensure_2d(x_pp)
    #     # For E_out, if your function updated/returned it elsewhere, we still recompute below
    #     # by flowing through transforms (safe). Otherwise, expose a hook:
    #     if hasattr(model, "inverse_energy_to_raw") and callable(model.inverse_energy_to_raw):
    #         E_out = model.inverse_energy_to_raw(conds).squeeze(-1)
    #     else:
    #         # Fallback: flow only energy through scalar-aware transforms
    #         #E_out = _reverse_energy_only(model, conds, skip_normalize_by_elayer).squeeze(-1)
    #         E_out = _reverse_energy_only(model, conds, skip_normalize_by_elayer, x_template=x).squeeze(-1)
    #     return x_pp, E_out

    # 1) Generic backward walk through `model.transforms`
    xg = x.clone()
    cc = E_scalar.reshape(-1, 1).clone()
    print("Before starting post processing: ", xg.shape , cc.shape)
    transforms = getattr(model, "transforms", None)
    if transforms:
        for fn in transforms[::-1]:
            name = fn.__class__.__name__
            if skip_normalize_by_elayer and name == "NormalizeByElayer":
                break

            if _needs_scalar_energy(name):
                cc_in = cc[:, :1]  # strictly scalar E for these transforms
            else:
                cc_in = cc

            # Expected API: fn(x, c, rev=True) -> (x_out, c_out)
            xg, cc_out = fn(xg, cc_in, rev=True)
            print("name: xg and cc_out ",name, xg.shape, cc_out.shape)

            # Keep condition as [B,1] (scalar E) unless your transform returns more
            if isinstance(cc_out, torch.Tensor):
                cc = cc_out.reshape(-1, 1) if cc_out.ndim == 1 else cc_out
            else:
                # if a transform returned None (shouldn't happen), keep previous cc
                pass

    x_pp = _ensure_2d(xg)
    E_out = cc.squeeze(-1)  # [B]
    return x_pp, E_out


@torch.no_grad()
def _reverse_energy_only(
    model,
    E_scalar: torch.Tensor,
    skip_normalize_by_elayer: bool,
    x_template: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Run only the scalar-energy path backward through transforms to recover physics-space E_inc.
    We supply a dummy shower tensor that matches the incoming x shape so reshape-like transforms
    don't crash. We STILL only trust/use the returned 'cc' (energy) from each transform.
    """
    cc = E_scalar.reshape(-1, 1).clone()

    # Use an x-shaped dummy to satisfy transforms that reshape the shower
    if x_template is not None:
        x_dummy = torch.zeros_like(x_template, dtype=cc.dtype, device=cc.device)
    else:
        # Fallback: assume [B,45] if no template is provided
        B = cc.size(0)
        D = getattr(model, "dim_embedding", 45)
        x_dummy = torch.zeros((B, D), dtype=cc.dtype, device=cc.device)

    transforms = getattr(model, "transforms", None)
    if transforms:
        for fn in transforms[::-1]:
            name = fn.__class__.__name__
            if skip_normalize_by_elayer and name == "NormalizeByElayer":
                break
            # We run the transform with the dummy x just to keep shapes consistent.
            # The scalar energy is what we care about.
            cc_in = cc[:, :1]
            x_dummy, cc_out = fn(x_dummy, cc_in, rev=True)
            if isinstance(cc_out, torch.Tensor):
                cc = cc_out.reshape(-1, 1) if cc_out.ndim == 1 else cc_out

    return cc.squeeze(-1)



def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(-1) if x.dim() == 3 and x.shape[-1] == 1 else x


def _needs_scalar_energy(transform_name: str) -> bool:
    """
    Central place to list transforms that require only scalar E as the condition
    when reversing. Extend this as you add more energy-aware transforms.
    """
    return transform_name in {
        "NormalizeByElayer",
        "ScaleTotalEnergy",
        "LogEnergy",
        "ScaleEnergy",
    }
