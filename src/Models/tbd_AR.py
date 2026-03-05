import numpy as np
import torch
from scipy.integrate import solve_ivp
import Networks
from Util.util import get
from Models.ModelBase import GenerativeModel
import Networks
import Models
from einops import rearrange
from .cfd_head import CFDHead, MVNHead
import math
from typing import Type, Callable, Union, Optional
import torch
import torch.nn as nn
from torchdiffeq import odeint
from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import apply_gradient_vector, get_gradient_vector


class TransfusionAR(GenerativeModel):

    def __init__(self, params: dict, device, doc):
        super().__init__(params, device, doc)
        self.params = params
        trajectory = get(self.params, "trajectory", "linear_trajectory")
        print(f"Trajectory is set to: {trajectory}")
        try:
            self.trajectory = getattr(Models.tbd, trajectory)
        except AttributeError:
            raise NotImplementedError(f"build_model: Trajectory type {trajectory} not implemented")

        self.dim_embedding = params["dim_embedding"]

        self.t_min = get(self.params, "t_min", 0)
        self.t_max = get(self.params, "t_max", 1)
        distribution = get(self.params, "distribution", "uniform")
        self.n_time_steps = params.get("n_time_steps", 20)  # default = 20
        if distribution == "uniform":
            self.distribution = torch.distributions.uniform.Uniform(low=self.t_min, high=self.t_max)
        elif distribution == "beta":
            self.distribution = torch.distributions.beta.Beta(1.5, 1.5)
        else:
            raise NotImplementedError(f"build_model: Distribution type {distribution} not implemented")

    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "ARtransformer")
        #print("Did I come here?")
        try:
            return getattr(Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def get_condition_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        # x = input[0].clone()
        condition = input[1]
        weights = None
        # return x, condition, weights
        return input[0], condition, weights
        
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
        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
            kl_scale: factor in front of KL loss term, default 0
        Returns:
            loss: batch loss
            loss_terms: dictionary with loss contributions
        """
        # print("batch loss in tbd_AR")
        # print("did I come here batch loss of transfusionAR")
        x, c, _ = self.get_condition_and_input(input)
        # print("inside batch loss: ",x.shape,c.shape)
        if self.latent: # encode x into autoencoder latent space
            # print("inside self.latent")
            x = self.ae.encode(x, c)
            if self.ae.kl:
                x = self.ae.reparameterize(x[0], x[1])
            x = self.ae.unflatten_layer_from_batch(x)
        # else:
        #     print(x.shape)
        #     x = x.movedim(1,2)
        #     print(x.shape)
            
        # add phantom layer dim to condition
        c = c.unsqueeze(-1)
        # print("after unsqueexzing: ",c.shape)
        # Sample time steps
        t = self.distribution.sample(
            list(x.shape[:2]) + [1]*(x.ndim-2)).to(dtype=x.dtype, device=x.device)

        # Sample noise variables
        x_0 = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        # Calculate point and derivative on trajectory
        # print("before trajectory: ", x_0.shape, t.shape,c.shape)
        x_t, x_t_dot = self.trajectory(x_0, x, t)
        v_pred = self.net(c,x_t,t,x)
        # print("what is the net? ")
        # Mask out masses if not needed
        loss_FM = ((v_pred - x_t_dot) ** 2).mean()   #loss from CFM
        loss=loss_FM
        

        #Let me generate the estimation of the data distribution from v_pred and x_t. Found this from : https://github.com/tum-pbs/PBFM/blob/main/darcy_flow/flow_matching.py
        t_1 = t.clone()
        dt  = (1 - t) / n_time_steps
        x_1_pred = x_t + dt * v_pred
        for _ in range(1, n_time_steps):
            t_1   = t_1 + dt
            v_t_1 = self.net(c, x_1_pred, t_1, x)   # or without x if you drop that arg
            x_1_pred = x_1_pred + dt * v_t_1

        
        dec_true = self._decode_layers_from_x(x, c)      # dict with "E_layer"
        dec_pred = self._decode_layers_from_x(x_1_pred, c)

        energy_loss = self._compute_energy_loss(dec_pred, dec_true, t)


        # Energy-space loss
        energy_loss = self._compute_energy_loss(dec_pred, dec_true, t)

        if energy_loss is not None:
            base_lambda_E = self._cfg_float("energy_lambda", 0.05)
        
            # Warm up energy over first N steps
            warm_steps = getattr(self, "energy_warmup_steps", 5000)
            if warm_steps > 0:
                s = min(1.0, float(self._global_step) / float(warm_steps))
                lambda_E = base_lambda_E * s
            else:
                lambda_E = base_lambda_E
            # Clip the per-batch energy loss to avoid catastrophic spikes
            clip_max = self._cfg_float("energy_clip_max", 5.0)  # you can tune this
            energy_loss_clipped = torch.clamp(energy_loss, max=clip_max)
                
            #loss = loss + lambda_E * energy_loss_clipped
            loss_phys = lambda_E * energy_loss_clipped

        

        return loss_FM, loss_phys

    @torch.inference_mode()
    def sample_batch(self,c, sampling_type=None):
        print("Am I here in sample_batch: ", c.shape)
        sample = self.net(c.unsqueeze(-1), rev=True)
        if self.latent: # decode the generated sample
            sample, c = self.ae.flatten_layer_to_batch(sample, c)
            sample = self.ae.decode(sample.squeeze(), c)
        return sample

def linear_trajectory(x_0, x_1, t):
    # print("Weirdo")
    x_t = (1 - t) * x_0 + t * x_1
    x_t_dot = x_1 - x_0
    return x_t, x_t_dot