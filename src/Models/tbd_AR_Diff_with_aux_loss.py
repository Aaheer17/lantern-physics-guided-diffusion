import numpy as np

from scipy.integrate import solve_ivp
import Networks
from Util.util import get
from Models.ModelBase import GenerativeModel

import Models
from einops import rearrange
from .cfd_head import CFDHead, MVNHead
import math
from typing import Type, Callable, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
class TransfusionDDPM(GenerativeModel):

    def __init__(self, params: dict, device, doc):
        super().__init__(params, device, doc)
        self.params = params
        
        # DDPM specific parameters
        self.num_timesteps = get(self.params, "num_timesteps", 1000)
        self.beta_schedule = get(self.params, "beta_schedule", "linear")
        self.cfg= get(self.params, "cfg", False)
        self.drop_prob= get(self.params, "drop_prob", 0)
        
        
            
        self.use_cfd    = bool(params.get("use_cfd", False))
        self.lambda_cfd = float(params.get("lambda_cfd", 1e-4))
        self.cfd_warmup_steps = int(params.get("cfd_warmup_steps", 5000))
     
        pred = str(params.get("prediction_type", "epsilon")).strip().lower()
        alias = {
            "eps": "epsilon", "e": "epsilon",
            "x_0": "x0", "x-start": "x0", "sample": "x0",
            "v-pred": "v", "v_prediction": "v",
        }
        pred = alias.get(pred, pred)
        allowed = {"epsilon", "x0", "v"}
        if pred not in allowed:
            raise ValueError(f"Unknown prediction_type='{pred}'. Allowed: {sorted(allowed)}")
        self.prediction_type = pred
        self.compute_CFD_metric=self.params.get('compute_CFD_metric', True)
        self.use_MSE_loss=self.params.get("eps_freeze_after_epoch",400)
        if self.use_cfd or self.compute_CFD_metric:
            print("initializing use_cfd")
            stats_dir = self.params.get('stats_ds2',None)
            delta     = float(self.params.get("alpha_logit", 1e-6))
            e_min     = float(self.params.get("e_min", 6.907755))
            e_max     = float(self.params.get("e_max", 13.815510))
            base_corr = self.params["base_corr"]   # path to (45,45) .npy

            ema_beta= self.params.get('cfd_ema_beta',None)      # e.g., 0.9 or 0.99; None disables EMA
            shrink_alpha= self.params.get('cfd_shrink_alpha',0.0)  # 0 disables shrinkage; try 0.05–0.15
            shrink_target= self.params.get('cfd_shrink_target','identity')

            print(ema_beta, shrink_alpha,shrink_target)

            self.cfd_head = CFDHead(
                stats_dir=stats_dir, delta=delta, e_min=e_min, e_max=e_max,
                C_real=base_corr, device=self.device, n_layers=45, require_stats=True,ema_beta=ema_beta,
                shrink_alpha=shrink_alpha, shrink_target=shrink_target
            )

        # ---- Energy-space loss knobs (DS2 defaults) ----
        self.use_energy_loss  = getattr(self, "use_energy_loss", False)
        
        # DS2 layer-wise floor from 15.15 keV voxel threshold × 144 vox/layer
        self.energy_floor_E0  = float(getattr(self, "energy_floor_E0", 2.1816e-3))  # GeV
        
        # smooth stabilizer by default; switch to "wls" for explicit inverse-variance
        self.energy_loss_type = str(getattr(self, "energy_loss_type", "smooth"))     # "smooth" | "wls"
        
        # apply loss on per-layer energies (robust, aligns with physics metrics)
        self.energy_resolution = str(getattr(self, "energy_resolution", "layer"))    # "layer" | "voxel" | "total"
        
        # weighting and timestep gating
        self.energy_lambda    = float(getattr(self, "energy_lambda", 0.05))           # try 0.1–1.0 for ablation
        self.energy_late_frac = float(getattr(self, "energy_late_frac", 0.20))       # last 20% of steps
        self.energy_warmup_steps = 5000

        
        # --- MVN/MCLT auxiliary loss (global second-order structure) ---
        self.use_mvn     = bool(self.params.get("use_mvn", False))
        self.lambda_mvn  = float(self.params.get("lambda_mvn", 1e-3))    # strength
        self.mvn_mode    = str(self.params.get("mvn_mode", "mean"))      # "mean" or "sample"
        self.mvn_stats   = self.params.get("mvn_stats_path", None)       # .npz with mu, Sigma
        # optional knobs
        self.mvn_time_weight = bool(self.params.get("mvn_time_weight", False))  # weight late timesteps
        self.mvn_warmup_steps = int(self.params.get("mvn_warmup_steps", 0))     # separate warmup if desired
        
        if self.use_mvn:
            if not self.mvn_stats:
                raise ValueError("use_mvn=True but mvn_stats_path not set.")
            # MVNHead returns a weighted scalar penalty; lambda is passed here
            self.mvn_head = MVNHead(
                stats_path_or_dict=self.mvn_stats,
                device=self.device,
                mode=self.mvn_mode,
                lambda_weight=self.lambda_mvn,
            )
           
        # warmup counter
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.long))

        # Initialize noise schedule
        self.setup_noise_schedule()
        
        self.dim_embedding = params["dim_embedding"]
        
    def _lambda_warm(self) -> float:
        if self.cfd_warmup_steps <= 0: return 1.0
        s = torch.clamp(self._global_step.float() / self.cfd_warmup_steps, 0, 1)
        return float(s)
    
    def setup_noise_schedule(self):
        """
        Setup the noise schedule for DDPM
        """
        if self.beta_schedule == "linear":
            beta_start = get(self.params, "beta_start", 1e-4)
            beta_end = get(self.params, "beta_end", 2e-2)
            self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            self.betas = self.cosine_beta_schedule(self.num_timesteps)
        elif self.beta_schedule == "sigmoid":
            self.betas = self.sigmoid_beta_schedule(self.num_timesteps) 
        else:
            raise NotImplementedError(f"Beta schedule {self.beta_schedule} not implemented")
        
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def sigmoid_beta_schedule(self, timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
        """Sigmoid schedule in https://arxiv.org/abs/2206.00364"""
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        v_start = torch.sigmoid(torch.tensor(start / tau)).item()
        v_end = torch.sigmoid(torch.tensor(end / tau)).item()
        alphas_cumprod = torch.sigmoid((t * (end - start) + start) / tau)
        alphas_cumprod = (v_end - alphas_cumprod) / (v_end - v_start)
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min = clamp_min, max = 0.999)


    def build_net(self):
        """
        Build the network - should output noise prediction
        """
        network = get(self.params, "network", "ARtransformer")
        try:
            return getattr(Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def get_condition_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        condition = input[1]
        weights = None
        return input[0], condition, weights

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(x_start.device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[t]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
        while len(sqrt_one_minus_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return (sqrt_alphas_cumprod_t * x_start + 
                sqrt_one_minus_alphas_cumprod_t * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        """
        eps = (x_t - sqrt(alpha_bar_t)*x0) / sqrt(1 - alpha_bar_t)
        """
        a, b = self._alpha_terms(t, x_t)
        return (x_t - a * x0) / (b + 1e-12)

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(x_t.device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
        while len(sqrt_one_minus_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    def _alpha_terms(self, t, like):
        """
        Returns (a, b) with correct broadcasting:
        a = sqrt(alpha_bar_t), b = sqrt(1 - alpha_bar_t)
        """
        a = self.sqrt_alphas_cumprod.to(like.device)[t]
        b = self.sqrt_one_minus_alphas_cumprod.to(like.device)[t]
        while len(a.shape) < len(like.shape): a = a.unsqueeze(-1)
        while len(b.shape) < len(like.shape): b = b.unsqueeze(-1)
        return a, b

    def v_from_x0_eps(self, x0, eps, t, like):
        """
        v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0
        """
        a, b = self._alpha_terms(t, like)
        return a * eps - b * x0
    
    def x0_eps_from_v(self, x_t, v, t):
        """
        Solve [x_t; v] = [[a, b],[-b, a]] [x0; eps]
        inverse => [x0; eps] = [[a,-b],[b,a]] [x_t; v]
        """
        a, b = self._alpha_terms(t, x_t)
        x0  = a * x_t - b * v
        eps = b * x_t + a * v
        return x0, eps
        
    def compute_parametrization_loss(self, x, c, t, x_t, eps_true):
        """
        Compute training loss depending on self.prediction_type and
        return a unified set of predictions for downstream use (CFD/logging).
    
        Returns:
            loss_param: scalar tensor
            eps_pred:   model's epsilon estimate (always returned)
            x0_pred:    model's x0 estimate (always returned)
            extras:     dict with per-param MSEs for logging
        """
        # Forward pass: one head, interpreted by prediction_type
        y = self.net(c, x_t, t, x)  # shape like x
    
        a, b = self._alpha_terms(t, x_t)
    
        extras = {}
    
        if self.prediction_type == "epsilon":
            eps_pred = y
            x0_pred  = self.predict_start_from_noise(x_t, t, eps_pred)
            loss_param = torch.mean((eps_pred - eps_true) ** 2)
            extras["mse_epsilon"] = loss_param.item()
            # for completeness
            extras["mse_x0"] = torch.mean((x0_pred - x) ** 2).item()
    
        elif self.prediction_type == "x0":
            x0_pred  = y
            eps_pred = self.predict_noise_from_start(x_t, t, x0_pred)
            loss_param = torch.mean((x0_pred - x) ** 2)
            extras["mse_x0"] = loss_param.item()
            extras["mse_epsilon"] = torch.mean((eps_pred - eps_true) ** 2).item()
    
        elif self.prediction_type == "v":
            # target/convert for v-param
            v_target = self.v_from_x0_eps(x, eps_true, t, like=x_t)
            v_pred   = y
            # primary loss on v
            loss_param = torch.mean((v_pred - v_target) ** 2)
            extras["mse_v"] = loss_param.item()
            # derive x0, eps predictions consistently from v
            x0_pred, eps_pred = self.x0_eps_from_v(x_t, v_pred, t)
            extras["mse_x0"] = torch.mean((x0_pred - x) ** 2).item()
            extras["mse_epsilon"] = torch.mean((eps_pred - eps_true) ** 2).item()
    
        else:
            raise RuntimeError("Unexpected prediction_type guard failed")
    
        return loss_param, eps_pred, x0_pred, extras


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1_t = self.posterior_mean_coef1.to(x_t.device)[t]
        posterior_mean_coef2_t = self.posterior_mean_coef2.to(x_t.device)[t]
        posterior_variance_t = self.posterior_variance.to(x_t.device)[t]
        
        # Reshape for broadcasting
        while len(posterior_mean_coef1_t.shape) < len(x_t.shape):
            posterior_mean_coef1_t = posterior_mean_coef1_t.unsqueeze(-1)
        while len(posterior_mean_coef2_t.shape) < len(x_t.shape):
            posterior_mean_coef2_t = posterior_mean_coef2_t.unsqueeze(-1)
        while len(posterior_variance_t.shape) < len(x_t.shape):
            posterior_variance_t = posterior_variance_t.unsqueeze(-1)
        
        posterior_mean = (posterior_mean_coef1_t * x_start + 
                         posterior_mean_coef2_t * x_t)
        
        return posterior_mean, posterior_variance_t
    def inverse_transform_to_raw(self, samples, conditions):
        """
        Apply inverse transforms to get back to raw data space
        Adapted from your plotting code

        Args:
            samples: Reconstructed data in preprocessed space [batch, 45, 1]
            conditions: Energy conditions (may be None if using CFG)

        Returns:
            samples_raw: Data in original raw space
        """
        # Handle condition shape
        if conditions is not None:
            conditions = conditions.squeeze(-1) if conditions.ndim > 2 else conditions

        # Apply inverse transforms in reverse order
        # Assuming self.transforms is your preprocessing pipeline
        for fn in self.transforms[::-1]:
            # if fn.__class__.__name__ == 'NormalizeByElayer':
            #     break  # Stop before NormalizeByElayer as in your plotting code
            samples, conditions = fn(samples, conditions, rev=True)

        return samples
    def _to_layer_vector(self, samples_raw: torch.Tensor) -> torch.Tensor:
        """
        Reduce raw-space samples to a per-sample 45-D vector (layer sums).
        Accepts shapes like [B, 1, 45, H, W] or [B, 45, 1] etc.
        Returns [B, 45], keeping grads.
        """
        X = samples_raw
        # sum all trailing spatial dims until we get [B, 45] or [B, 45, 1]
        while X.ndim > 3:
            X = X.sum(dim=-1)
        if X.ndim == 3:  # [B,45,1] -> [B,45]
            X = X.squeeze(-1)
        # if the 45 dimension isn’t at dim=1 (rare), rearrange here:
        if X.shape[1] != 45 and 45 in X.shape:
            # move the 45-dim to channel slot
            dim_45 = X.shape.index(45)
            X = X.movedim(dim_45, 1)
        return X  # [B,45]

    def _decode_layers_from_x(self, x_like, c_like):
        """
        Returns dict with:
          u:        (B, L)
          e:        (B,)
          E_layer:  (B, L)  per-layer energies in physical units
        Uses the same inverse transforms as your CFD/MVN utilities.
        """
        u, e = self.cfd_head._reverse_to_u(x_like, c_like)   # differentiable w.r.t. x_like
        E_layer = self.cfd_head._u_to_layerE(u, e)           # (B, L)
        return {"u": u, "e": e, "E_layer": E_layer}

   # ---------- Config getters (keeps batch_loss clean) ----------
    def _cfg_bool(self, key, default=False):
        return bool(getattr(self, key, default))
    
    def _cfg_float(self, key, default=0.0):
        return float(getattr(self, key, default))
    
    def _cfg_str(self, key, default=""):
        return str(getattr(self, key, default))


    def _need_decode_pred(self):
        """Whether we need a decode of x0 for any aux terms or metrics."""
        return (
            (self._cfg_bool("use_mvn") and hasattr(self, "mvn_head")) or
            self._cfg_bool("use_energy_loss") or
            self._cfg_bool("compute_CFD_metric")
        ) and (self.cfd_head is not None)

    # ---------- Timestep gating ----------
    def _late_mask(self, t, frac, T):
        """
        Gate to the last 'frac' of timesteps.
        t: (B,) ints in [0, T]
        Returns boolean mask of shape (B,).
        """
        t_cut = int((1.0 - float(frac)) * (T + 1))  # inclusive cut into late zone
        return (t >= t_cut)
    
    # ---------- MVN / MCLT ----------
    def _compute_mvn_losses(self, dec_pred, t):
        """
        dec_pred: dict from _decode_layers_from_x for x0_pred
        Returns (l_mvn, l_trace, mvn_loss_val, mvn_trace_val)
        """
        if not (self._cfg_bool("use_mvn") and hasattr(self, "mvn_head") and (dec_pred is not None)):
            return None, None, None, None, None
    
        vecs_45 = dec_pred["E_layer"]  # (B, L)
    
        l_mvn = self.mvn_head(vecs_45)  # differentiable
        raw_mvn=l_mvn
    
        # Late gate
        T = max(self.num_timesteps - 1, 1)
        q = self._cfg_float("mvn_late_frac", 0.20)
        gate = self._late_mask(t, q, T).float().mean()
    
        # Warm-up
        if self._cfg_float("mvn_warmup_steps", 0) > 0:
            step_counter = float(getattr(self, "_global_step", 0))
            warm = torch.clamp(
                torch.tensor(step_counter, device=l_mvn.device) / float(self.mvn_warmup_steps),
                0.0, 1.0
            )
        else:
            warm = torch.tensor(1.0, device=l_mvn.device)
    
        factor = gate * warm
        l_mvn = l_mvn * factor
    
        # Optional trace add-on (keeps your previous behavior)
        eta_trace = self._cfg_float("mvn_trace_eta", 0.0)
        unit_sum  = bool(getattr(self, "mvn_unit_sum", False))
        l_trace   = self.mvn_head.trace_penalty(vecs_45, eta=eta_trace, unit_sum=unit_sum)
        l_trace   = l_trace * factor
    
        return l_mvn, l_trace, float(l_mvn.item()), float(l_trace.item()), float(raw_mvn.item())
    
    def _compute_energy_loss(self, dec_pred, dec_true, t, c=None, debug=True, use_early_mask=False):
        """
        dec_pred/dec_true: dicts from _decode_layers_from_x (x0_pred and x)
        Returns: scalar tensor or None
        """
        if not (self._cfg_bool("use_energy_loss") and (self.cfd_head is not None)
                and (dec_pred is not None) and (dec_true is not None)):
            return None
    
        T    = max(self.num_timesteps - 1, 1)
        frac = self._cfg_float("energy_late_frac", 0.20)
    
        if use_early_mask:
            # use first `frac` of steps (closer to data)
            t_cut = int(frac * (T + 1)) - 1
            mask_E = (t <= t_cut)
        else:
            # use last `frac` of steps
            t_cut = int((1.0 - frac) * (T + 1))
            mask_E = (t >= t_cut)
    
        if not mask_E.any():
            return None
    
        idx = mask_E.nonzero(as_tuple=False).squeeze(-1)
    
        # (b_sel, L)
        E_pred_layer = dec_pred["E_layer"].index_select(0, idx)
        E_true_layer = dec_true["E_layer"].index_select(0, idx)
    
        # # Clamp tiny numeric negatives; keep differentiable
        E_pred_layer = torch.clamp(E_pred_layer, min=0.0)
        E_true_layer = torch.clamp(E_true_layer, min=0.0)
    
        # Make E0 a tensor on the right device/dtype
        E0_val = self._cfg_float("energy_floor_E0", 2.1816e-3)  # DS2 default in GeV
        E0 = torch.as_tensor(E0_val, device=E_pred_layer.device, dtype=E_pred_layer.dtype)
    
        # Smooth stabilizer: L = mean( || 2*sqrt(E+E0) - 2*sqrt(\hat E+E0) ||^2 )
        y_pred = 2.0 * torch.sqrt(E_pred_layer + E0)
        y_true = 2.0 * torch.sqrt(E_true_layer + E0)
        #loss = torch.mean((y_pred - y_true) ** 2)
        beta = self._cfg_float("energy_huber_beta", 0.5)
        loss = F.smooth_l1_loss(y_pred, y_true, beta=beta)


        if not torch.isfinite(loss):
            # Fallback to WLS if something goes off the rails
            denom = torch.clamp(E_true_layer, min=E0)
            loss = torch.mean(((E_pred_layer - E_true_layer) ** 2) / denom)
    
        if debug:
            # Safe, compact diagnostics
            print(f"[energy] mask true: {int(mask_E.sum().item())}, early={use_early_mask}")
            if c is not None:
                print(f"[energy] c is None? {c is None}")
            print("[energy] E_true min/max/mean:",
                  E_true_layer.min().item(), E_true_layer.max().item(), E_true_layer.mean().item())
            print("[energy] E_pred min/max/mean:",
                  E_pred_layer.min().item(), E_pred_layer.max().item(), E_pred_layer.mean().item())
            print("[energy] allclose(E_true,E_pred)?",
                  torch.allclose(E_true_layer, E_pred_layer, rtol=1e-6, atol=1e-8))
            print("[energy] ||E_pred-E_true||:", torch.norm(E_pred_layer - E_true_layer).item())
    
        return loss
    

    def batch_loss(self, input):
        """
        DDPM training loss
        Returns: loss, comps (dict of step-wise metrics for CSV)
        """
        x, c, _ = self.get_condition_and_input(input)
        if self.latent:
            x = self.ae.encode(x, c)
            if self.ae.kl:
                x = self.ae.reparameterize(x[0], x[1])
            x = self.ae.unflatten_layer_from_batch(x)
    
        # CFG handling -> ensure c has shape [B, 1] when conditioned
        if self.cfg:
            if torch.rand(1).item() < self.drop_prob:
                c = None
            else:
                c = c.unsqueeze(-1)
        else:
            c = c.unsqueeze(-1)
    
        # Timesteps and noise
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        noise = torch.randn_like(x)
        x_t = self.q_sample(x, t, noise=noise)
    
        # Base DDPM param loss
        loss_param, eps_pred, x0_pred, extras = self.compute_parametrization_loss(
            x=x, c=c, t=t, x_t=x_t, eps_true=noise
        )
        # if self.use_MSE_loss < self._global_step:
        #     loss=0
        # else:
        loss = loss_param
    
        # Decode once (pred) and optionally true (for energy)
        dec_pred = self._decode_layers_from_x(x0_pred, c) if self._need_decode_pred() else None
        dec_true = self._decode_layers_from_x(x, c) if self._cfg_bool("use_energy_loss") and (self.cfd_head is not None) else None
    
        # # MVN / trace
        l_mvn, l_trace, mvn_loss_val, mvn_trace_val, raw_mvn = self._compute_mvn_losses(dec_pred, t)
        if l_mvn is not None:
            loss = loss + l_mvn

        if l_trace is not None:
            loss = loss + l_trace
                
    
        # Energy-space loss
        energy_loss = self._compute_energy_loss(dec_pred, dec_true, t)

        # if energy_loss is not None:
        #     base_lambda_E = self._cfg_float("energy_lambda", 0.05)
        
        #     # Warm up energy over first N steps
        #     warm_steps = getattr(self, "energy_warmup_steps", 5000)
        #     if warm_steps > 0:
        #         s = min(1.0, float(self._global_step) / float(warm_steps))
        #         lambda_E = base_lambda_E * s
        #     else:
        #         lambda_E = base_lambda_E
        #     # Clip the per-batch energy loss to avoid catastrophic spikes
        #     clip_max = self._cfg_float("energy_clip_max", 5.0)  # you can tune this
        #     energy_loss_clipped = torch.clamp(energy_loss, max=clip_max)
                
        #     loss = loss + lambda_E * energy_loss_clipped
    
    
        # CFD metric (as before)
        metric_CFD = None
        if self._cfg_bool("compute_CFD_metric") and (self.cfd_head is not None):
            metric_CFD = self.cfd_head.metrics_from_model(x0_pred, c)
    
        # CFD loss (kept with your existing API)
        cfd_raw = None
        w_t_mean = None
        lambda_warm = None
        lambda_cfd_eff = None
        dbg = {}
        if self._cfg_bool("use_cfd") and (self.cfd_head is not None):
            cfd = self.cfd_head.loss(x0_pred, c)
            dbg = getattr(self.cfd_head, "last_debug", {})
            T = max(self.num_timesteps - 1, 1)
            w_t = (1.0 - (t.float() / T)).pow(2).mean()
            w_t_mean = float(w_t.item())
            lambda_warm = float(self._lambda_warm())
            lambda_cfd_eff = float(self.lambda_cfd) * lambda_warm * w_t_mean
            cfd_raw = float(cfd.item())
            loss = loss + lambda_cfd_eff * cfd
    
        # Logging
        self._global_step += 1
        mse_noise = extras.get("mse_epsilon")
        mse_x0    = extras.get("mse_x0")
        mse_v     = extras.get("mse_v")
    
        cfd_metric_val = None
        if isinstance(metric_CFD, dict):
            cfd_metric_val = metric_CFD.get("CFD_offdiag_fro", None)
            if cfd_metric_val is None:
                cfd_metric_val = metric_CFD.get("CFD_full_fro", None)
    
        comps = {
            "loss_param": float(loss_param.item()),
            "loss_total": float(loss.item()),
            "mvn_loss": mvn_loss_val,
            "mvn_trace_val": mvn_trace_val,
            "raw_mvn": raw_mvn,
            "mvn_mode": self.mvn_mode if self._cfg_bool("use_mvn") else None,
            "mse_noise": None if mse_noise is None else float(mse_noise),
            "mse_x0":    None if mse_x0 is None else float(mse_x0),
            "mse_v":     None if mse_v is None else float(mse_v),
            "CFD_metric": None if cfd_metric_val is None else float(cfd_metric_val),
            "cfd_raw": cfd_raw,
            "w_t_mean": w_t_mean,
            "lambda_warm": lambda_warm,
            "lambda_cfd_eff": lambda_cfd_eff,
            "cfd_fro_pred":   dbg.get("cfd_fro_pred"),
            "cfd_fro_shrunk": dbg.get("cfd_fro_shrunk"),
            "cfd_fro_smooth": dbg.get("cfd_fro_smooth"),
            "cfd_ema_dist":   dbg.get("cfd_ema_dist"),
            "energy_loss": None if energy_loss is None else float(energy_loss.item()),
        }
    
        return loss, comps


    @torch.no_grad()
    def predict_x0_and_eps(self, x_t, t, c):
        """
        Get x0 and epsilon predictions from the model during sampling.
        Handles all prediction types: epsilon, x0, v.
        
        Args:
            x_t: noisy sample at timestep t
            t: timestep tensor
            c: conditioning
        
        Returns:
            x0_pred: predicted clean sample
            eps_pred: predicted noise
        """
        # Forward pass - single model output
        y = self.net(c, x_t, t, x=None)  # No x_orig needed during sampling
        
        # Interpret output based on prediction_type
        if self.prediction_type == "epsilon":
            eps_pred = y
            x0_pred = self.predict_start_from_noise(x_t, t, eps_pred)
        
        elif self.prediction_type == "x0":
            x0_pred = y
            eps_pred = self.predict_noise_from_start(x_t, t, x0_pred)
        
        elif self.prediction_type == "v":
            x0_pred, eps_pred = self.x0_eps_from_v(x_t, y, t)
        
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
        
        return x0_pred, eps_pred
    
    
    @torch.no_grad()
    def p_sample(self, x_t, t, c, x_orig=None):
        """
        One reverse DDPM step: q(x_{t-1} | x_t, x0_hat)
        Agnostic to prediction_type ∈ {epsilon, x0, v}.
        
        Args:
            x_t: current noisy sample [B, ...]
            t: timestep tensor [B]
            c: conditioning
            x_orig: unused during standard sampling
        
        Returns:
            x_{t-1}: denoised sample at previous timestep
        """
        # 1) Get x0 prediction (handles all prediction types)
        x0_pred, eps_pred = self.predict_x0_and_eps(x_t, t, c)
        
        # 2) Compute posterior mean and variance: q(x_{t-1} | x_t, x0_hat)
        model_mean, model_var = self.q_posterior_mean_variance(x0_pred, x_t, t)
        
        # 3) Add noise except at t=0
        noise = torch.randn_like(x_t)
        
        # Create broadcastable mask for "t != 0"
        nonzero = (t != 0).float().view(x_t.shape[0], *([1] * (x_t.ndim - 1)))
        
        # Sample: x_{t-1} = mean + sqrt(var) * noise (when t > 0)
        x_prev = model_mean + nonzero * torch.sqrt(model_var.clamp_min(1e-12)) * noise
        
        return x_prev
    # def p_sample(self, x_t, t, c, x_orig=None):
    #     """
    #     Sample x_{t-1} from x_t
    #     """
    #     # Predict noise
    #     predicted_noise = self.net(c, x_t, t, x_orig)
        
    #     # Predict x_0
    #     x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        
    #     # Get posterior mean and variance
    #     posterior_mean, posterior_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        
    #     if t == 0:
    #         return posterior_mean
    #     else:
    #         noise = torch.randn_like(x_t)
    #         return posterior_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, c):
        """Full DDPM sampling loop"""
        device = c.device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, c)
        
        return x
        
    
    
    @torch.inference_mode()
    def sample_batch(self, c,sampling_type='ddpm'):
        """
        Generate samples using DDPM
        """
        #print("Sampling with DDPM: ", c.shape)
        #print("in sample_batch: ",self.params)
        diffusion_params = {
            'sqrt_alphas_cumprod': self.sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod,
            'posterior_mean_coef1': self.posterior_mean_coef1,
            'posterior_mean_coef2': self.posterior_mean_coef2,
            'posterior_variance': self.posterior_variance,
        }
        
        c_unsqueezed = c.unsqueeze(-1)
        
        # Determine shape for sampling
        if hasattr(self, 'sample_shape'):
            shape = self.sample_shape
        else:
            # Infer shape from condition or use default
            shape = list(c.shape) + [self.dim_embedding]  # Adjust as needed
        
        # Run sampling loop
        #sample = self.p_sample_loop(shape, c_unsqueezed)
        sample = self.net(
            c, x_t=None, t=None, x=None, rev=True,
            diffusion_buffers=diffusion_params,
            sampling_type=sampling_type
        )
        
        if self.latent:  # decode the generated sample
            sample, c = self.ae.flatten_layer_to_batch(sample, c)
            sample = self.ae.decode(sample.squeeze(), c)
            
        return sample

    @torch.no_grad()
    def ddim_sample(self, c, ddim_timesteps=50, eta=0.0):
        """DDIM sampling for faster inference"""
        device = c.device
        batch_size = c.shape[0]
        c_unsqueezed = c.unsqueeze(-1)
        
        # Create DDIM timestep schedule
        step = self.num_timesteps // ddim_timesteps
        timesteps = list(reversed(range(0, self.num_timesteps, step)))[:ddim_timesteps]
        
        # Start from pure noise
        if hasattr(self, 'sample_shape'):
            shape = self.sample_shape
        else:
            shape = list(c.shape) + [self.dim_embedding]
        
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get predictions (handles all prediction types)
            x0_pred, eps_pred = self.predict_x0_and_eps(x, t_tensor, c_unsqueezed)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t].to(device)
            alpha_t_prev = (self.alphas_cumprod[timesteps[i+1]].to(device) 
                           if i < len(timesteps) - 1 
                           else torch.tensor(1.0, device=device))
            
            # Reshape for broadcasting
            while len(alpha_t.shape) < len(x.shape):
                alpha_t = alpha_t.unsqueeze(-1)
            while len(alpha_t_prev.shape) < len(x.shape):
                alpha_t_prev = alpha_t_prev.unsqueeze(-1)
            
            # DDIM update
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * 
                                       (1 - alpha_t / alpha_t_prev))
            
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * eps_pred
            noise = sigma_t * torch.randn_like(x) if eta > 0 else 0
            
            x = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt + noise
        
        if self.latent:
            x, c = self.ae.flatten_layer_to_batch(x, c)
            x = self.ae.decode(x.squeeze(), c)
        
        return x


# import torch
# import torch.nn.functional as F

@torch.no_grad()
def evaluate_energy_losses_and_mse(model, dataloader, device,
                                   E0_val=None,
                                   huber_beta=0.5,
                                   max_batches=None):
    """
    Post-hoc evaluation for a trained TransfusionDDPM model.

    For each batch:
      1) Sample timesteps t and forward-noise x -> x_t
      2) Run the model to get eps_pred and x0_pred
      3) Decode per-layer energies from x and x0_pred
      4) Apply the same late-t gate (energy_late_frac) for the energy term
      5) Compute:
           - eps-MSE:   mean( (eps_pred - noise)^2 ) over all voxels
           - L_E^MSE:   mean( (2*sqrt(E+E0) - 2*sqrt(Ehat+E0))^2 ) on gated samples
           - L_E^Huber: smooth L1 loss in the same stabilized space

    Returns:
      avg_L_eps_mse, avg_L_E_mse, avg_L_E_huber
    """

    model.eval()

    # Energy floor: use model setting if not provided
    if E0_val is None:
        E0_val = float(getattr(model, "energy_floor_E0", 2.1816e-3))

    E0 = torch.as_tensor(E0_val, device=device)

    total_eps_mse = 0.0
    total_mse_E   = 0.0
    total_huber_E = 0.0
    total_eps_count = 0
    total_E_count   = 0

    T = max(model.num_timesteps - 1, 1)
    frac = float(getattr(model, "energy_late_frac", 0.20))  # same as training
    t_cut = int((1.0 - frac) * (T + 1))

    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break

        # --- 1. Get x and condition exactly as in batch_loss ---
        x, c, _ = model.get_condition_and_input(batch)
        x = x.to(device)
        c = c.to(device)

        # latent AE handling (copied from batch_loss)
        if getattr(model, "latent", False):
            x_lat = model.ae.encode(x, c)
            if model.ae.kl:
                x_lat = model.ae.reparameterize(x_lat[0], x_lat[1])
            x = model.ae.unflatten_layer_from_batch(x_lat)

        # CFG handling -> keep c as in training (no dropout at eval)
        if getattr(model, "cfg", False):
            c_in = c.unsqueeze(-1)  # conditioned branch only at eval
        else:
            c_in = c.unsqueeze(-1)

        B = x.shape[0]
        if B == 0:
            continue

        # --- 2. Sample timesteps and q(x_t | x_0) ---
        t = torch.randint(0, model.num_timesteps, (B,), device=device).long()
        noise = torch.randn_like(x)
        x_t = model.q_sample(x, t, noise=noise)

        # --- 3. Run model to get eps_pred and x0_pred (reuse training helper) ---
        loss_param, eps_pred, x0_pred, extras = model.compute_parametrization_loss(
            x=x, c=c_in, t=t, x_t=x_t, eps_true=noise
        )

        # ==========================
        # 3a. eps-MSE over all voxels
        # ==========================
        # Shape: (B, ...) for both eps_pred and noise
        eps_mse_batch = torch.mean((eps_pred - noise) ** 2)
        n_eps = x.numel()  # number of scalar entries used for the eps MSE
        total_eps_mse   += eps_mse_batch.item() * n_eps
        total_eps_count += n_eps

        # ==========================
        # 4. Decode per-layer energies (for energy losses)
        # ==========================
        dec_pred = model._decode_layers_from_x(x0_pred, c_in)
        dec_true = model._decode_layers_from_x(x,       c_in)

        E_pred = torch.clamp(dec_pred["E_layer"], min=0.0)  # (B, L)
        E_true = torch.clamp(dec_true["E_layer"], min=0.0)

        # --- 5. Apply the same late-t gate as during training ---
        mask_E = (t >= t_cut)
        if not mask_E.any():
            continue

        idx = mask_E.nonzero(as_tuple=False).squeeze(-1)
        E_pred_sel = E_pred.index_select(0, idx)
        E_true_sel = E_true.index_select(0, idx)

        # --- 6. Compute losses in stabilized space g(E) = 2 sqrt(E + E0) ---
        y_pred = 2.0 * torch.sqrt(E_pred_sel + E0)
        y_true = 2.0 * torch.sqrt(E_true_sel + E0)

        L_mse_E_batch   = torch.mean((y_pred - y_true) ** 2)
        L_huber_E_batch = F.smooth_l1_loss(y_pred, y_true, beta=huber_beta)

        n_E = E_true_sel.shape[0]
        total_mse_E   += L_mse_E_batch.item()   * n_E
        total_huber_E += L_huber_E_batch.item() * n_E
        total_E_count += n_E

    if total_eps_count == 0 or total_E_count == 0:
        return float("nan"), float("nan"), float("nan")

    avg_L_eps_mse = total_eps_mse / total_eps_count
    avg_L_E_mse   = total_mse_E   / total_E_count
    avg_L_E_huber = total_huber_E / total_E_count

    return avg_L_eps_mse, avg_L_E_mse, avg_L_E_huber
