import numpy as np
import torch
from scipy.integrate import solve_ivp
import Networks
from Util.util import get
from Models.ModelBase import GenerativeModel
import Models
from torchdiffeq import odeint
import math
import sys
import os
from torch import linalg as LA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
from transforms import (
    Reshape,
    AddFeaturesToCond,
    ScaleEnergy,
    LogEnergy,
    StandardizeFromFile,
    ExclusiveLogitTransform,
    SelectiveUniformNoise,
    CutValues,
    NormalizeByElayer,
    ScaleVoxels
)

# ============================================
# Auxiliary losses integration
# ============================================
from .auxiliary_losses_shape import AuxiliaryLossManager
from .multi_objective_optimizers import MultiObjectiveOptimizer

torch.cuda.empty_cache()


class TBD_DIFF_ENHANCED(GenerativeModel):
    """
    Class for Trajectory Based Diffusion - converted to proper diffusion model
    Inheriting from the GenerativeModel BaseClass
    
    Supports 4 multi-objective optimization methods (configurable):
    1. weighted_sum - Simple weighted combination (default)
    2. config - Uses existing ConFIG in ModelBase.py
    3. uncertainty - Automatic uncertainty weighting
    4. aug_lag - Augmented Lagrangian constraint optimization
    """

    def __init__(self, params, device, doc):
        super().__init__(params, device, doc)
        print("Initializing Diffusion Model")
        
        # Diffusion parameters
        self.timesteps = get(self.params, "num_timesteps", 1000)
        self.beta_start = get(self.params, "beta_start", 0.0001)
        self.beta_end = get(self.params, "beta_end", 0.02)
        self.beta_schedule = get(self.params, "beta_schedule", "linear")
        print("self.beta_schedule: ",self.beta_schedule)
        self.cfg= get(self.params, "cfg", False)
        self.drop_prob= get(self.params, "drop_prob", 0)
        self.cfg_scale= get(self.params, "cfg_scale", 0)


        self.stats_dir = self.doc.basedir 
        self.xml_file = get(self.params, 'xml_filename', 
                           '/project/biocomplexity/fa7sa/calo_dreamer/src/challenge_files/binning_dataset_2.xml')
        
        self.batch_counter=0
        # ============================================
        # Auxiliary Loss Configuration
        # ============================================
        aux_config = {
            'use_energy_loss': get(self.params, 'use_energy_loss', False),
            'use_mvn_loss': get(self.params, 'use_mvn_loss', False),
            'use_cfd_loss': get(self.params, 'use_cfd_loss', False),
            'num_layers': self.shape[1] if len(self.shape) > 1 else 45,
            
            # Energy loss config
            'energy_loss_type': get(self.params, 'energy_loss_type', 'huber'),
            'huber_delta': get(self.params, 'huber_delta', 1.0),
            'energy_use_transform': get(self.params, 'energy_use_transform', False),
            'energy_E_0': get(self.params, 'energy_E_0', 1e-3),
            
            # MVN loss config
            'mvn_lambda_mean': get(self.params, 'mvn_lambda_mean', 1.0),
            'mvn_lambda_cov': get(self.params, 'mvn_lambda_cov', 0.5),
            'mvn_shrinkage': get(self.params, 'mvn_shrinkage', 0.1),
            'mvn_ema_rho': get(self.params, 'mvn_ema_rho', 0.99),
            
            # CFD loss config
            'cfd_shrinkage': get(self.params, 'cfd_shrinkage', 0.1),
            'cfd_ema_rho': get(self.params, 'cfd_ema_rho', 0.99),
            'cfd_warmup_steps': get(self.params, 'cfd_warmup_steps', 2000),
            'lambda_cfd': get(self.params, 'lambda_cfd', 0.1),
            'cfd_off_diagonal_only': get(self.params, 'cfd_off_diagonal_only', True)
        }
        
        # Initialize auxiliary loss manager
        self.aux_loss_manager = AuxiliaryLossManager(aux_config).to(device)
        
        # Check if any auxiliary losses are enabled
        self.use_aux_losses = (aux_config['use_energy_loss'] or 
                               aux_config['use_mvn_loss'] or 
                               aux_config['use_cfd_loss'])
        
        # ============================================
        # Multi-Objective Optimization Method Selection
        # ============================================
        # Options: 'weighted_sum', 'config', 'uncertainty', 'aug_lag'
        self.mo_method = get(self.params, 'mo_method', 'weighted_sum')
        
        if self.use_aux_losses:
            print(f"Auxiliary losses enabled with method: {self.mo_method}")
            
            if self.mo_method == 'weighted_sum':
                # Simple weighted sum - manual weight tuning
                self.lambda_energy = get(self.params, 'lambda_energy', 0.02)
                self.lambda_mvn = get(self.params, 'lambda_mvn', 0.05)
                self.lambda_cfd = get(self.params, 'lambda_cfd', 0.1)
                print(f"  Weighted sum - Energy: {self.lambda_energy}, MVN: {self.lambda_mvn}, CFD: {self.lambda_cfd}")
            
            elif self.mo_method == 'config':
                # Use existing ConFIG in ModelBase.py
                # ModelBase will handle gradient combination
                self.lambda_energy = get(self.params, 'lambda_energy', 0.02)
                self.lambda_mvn = get(self.params, 'lambda_mvn', 0.05)
                self.lambda_cfd = get(self.params, 'lambda_cfd', 0.1)
                print(f"  ConFIG mode - ModelBase will handle gradient combination")
                print(f"  Note: Set use_config=True in config for ConFIG to work")
            
            elif self.mo_method == 'uncertainty':
                # Automatic uncertainty weighting
                task_names = ['diffusion']
                if aux_config['use_energy_loss']:
                    task_names.append('energy')
                if aux_config['use_mvn_loss']:
                    task_names.append('mvn')
                if aux_config['use_cfd_loss']:
                    task_names.append('cfd')
                
                self.mo_optimizer = MultiObjectiveOptimizer(
                    method='uncertainty',
                    task_names=task_names,
                    init_log_var=get(self.params, 'uncertainty_init_log_var', -1.0)
                )
                print(f"  Uncertainty weighting for tasks: {task_names}")
            
            elif self.mo_method == 'aug_lag':
                # Augmented Lagrangian
                self.mo_optimizer = MultiObjectiveOptimizer(
                    method='aug_lag',
                    init_mu=get(self.params, 'al_init_mu', 1.0),
                    beta=get(self.params, 'al_beta', 1.5),
                    tau=get(self.params, 'al_tau', 0.8)
                )

                # The internal nn.Module needs to be on CUDA
                if hasattr(self.mo_optimizer, 'optimizer'):
                    self.mo_optimizer.optimizer = self.mo_optimizer.optimizer.to(device)
                elif hasattr(self.mo_optimizer, 'to'):
                    self.mo_optimizer = self.mo_optimizer.to(device)
                print(f"  Augmented Lagrangian - μ={get(self.params, 'al_init_mu', 1.0)}")
            
            else:
                raise ValueError(f"Unknown mo_method: {self.mo_method}. "
                               f"Choose from: weighted_sum, config, uncertainty, aug_lag")
        else:
            print("No auxiliary losses enabled")
        
        # Initialize noise schedule
        self.setup_noise_schedule()
        
        # Other parameters
        self.C = get(self.params, "C", 1)
        if self.C != 1:
            print(f"C is {self.C}")

        self.bayesian = get(self.params, "bayesian", 0)
        self.add_noise = get(self.params, "add_noise", False)
        self.gamma = get(self.params, "gamma", 1.e-4)
        
        # Prediction type: 'noise' or 'x0'
        self.prediction_type = get(self.params, "prediction_type", "noise")

        # ============================================================
        # Load energy-binned MVN targets (ONE TIME)
        # ============================================================
        
        mvn_path = "/project/biocomplexity/fa7sa/Modified_Training_Objective/LANTERN_new/src/Models/mvn_shape_targets_ds2_binned_corr_full.pt"
        
        self.mvn_binned = torch.load(mvn_path, map_location="cpu")
        
        # Sanity check
        assert "bins_gev" in self.mvn_binned
        assert "bins" in self.mvn_binned
        
        # Build a clean list of usable bins:
        # (lo, hi, key) where key matches entries in self.mvn_binned["bins"]
        self.mvn_bin_list = []
        
        for (lo, hi) in self.mvn_binned["bins_gev"]:
            key = f"{lo:g}_{hi:g}_GeV"
        
            # Only keep bins that actually have statistics
            bin_data = self.mvn_binned["bins"].get(key, None)
            if bin_data is None:
                continue
            if "mu" not in bin_data or "chol" not in bin_data:
                continue
            if bin_data.get("count", 0) < 2:
                continue
        
            self.mvn_bin_list.append((float(lo), float(hi), key))
        
        # Optional but very useful: sort by energy (safety)
        self.mvn_bin_list.sort(key=lambda x: x[0])

        print(f"[MVN] Loaded {len(self.mvn_bin_list)} energy bins from {mvn_path}")

        
    def setup_noise_schedule(self):
        """Setup the noise schedule for diffusion"""
        if self.beta_schedule == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        elif self.beta_schedule == "cosine":
            betas = self.cosine_beta_schedule(self.timesteps)
        elif self.beta_schedule == "sigmoid":
            betas = self.sigmoid_beta_schedule(self.timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        self.betas = betas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        
        # Log calculation clipped because the posterior variance is 0 at the beginning
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        )
        
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min = 0, max = 0.999)
    
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

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Sample from q(x_t | x_0) using the reparameterization trick
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

    def build_net(self):
        """Build the network"""
        network = get(self.params, "network", "Resnet")
        try:
            return getattr(Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def get_condition_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        condition = input[1] if isinstance(input, (list, tuple)) and len(input) > 1 else None
        weights = None
        return input[0] if isinstance(input, (list, tuple)) else input, condition, weights

    def apply_inverse_transforms(self, x, condition=None,return_layer_energies=False, return_summed_energies=False):
        """
        Apply inverse preprocessing transforms to get layer energies
        
        Args:
            x: (B, C, L, H, W) - preprocessed data from model (e.g., B, 1, 45, 16, 9)
            condition: (B, 46) - [E_inc, u_1, u_2, ..., u_45] where:
                - condition[:, 0] is incident energy (after ScaleEnergy and LogEnergy)
                - condition[:, 1:46] are the 45 u features (after StandardizeFromFile)
        
        Returns:
            layer_energies: (B, 45) - energy deposited in each of 45 layers
        
        This applies reverse transforms up to NormalizeByElayer which directly
        computes layer energies from the u features.
        """
        
        if condition is None:
            raise ValueError("Condition is required for inverse transforms!")
        
        shower = x
        energy = condition
        
        # ========================================
        # REVERSE TRANSFORMATION PIPELINE
        # ========================================
        
        # 10. Reshape (reverse): (B, 1, 45, 16, 9) → (B, 6480)
        transform = Reshape(shape=[1, 45, 16, 9])
        shower, energy = transform(shower, energy, rev=True)
        # shower: (B, 6480), energy: (B, 46)
        
        # 9. AddFeaturesToCond (reverse): Add u features back to shower
        transform = AddFeaturesToCond(split_index=6480)
        print("Before calling the AddFeaturesToCond: ",shower.shape, energy.shape)
        shower, energy = transform(shower, energy, rev=True)
        # shower: (B, 6525) = (B, 6480 + 45), energy: (B, 1)
        
        # 8. ScaleEnergy (reverse): Unnormalize incident energy
        transform = ScaleEnergy(e_min=6.907755, e_max=13.815510)
        shower, energy = transform(shower, energy, rev=True)
        # energy is now in log-space
        
        # 7. LogEnergy (reverse): exp(energy) to get linear energy
        transform = LogEnergy()
        shower, energy = transform(shower, energy, rev=True)
        # energy is now in linear space (MeV)
        
        # 6. StandardizeFromFile (reverse): Unstandardize last 45 features
        # Note: You need to provide the model_dir where stats are saved
        
        transform = StandardizeFromFile(model_dir=self.stats_dir)
        shower, energy = transform(shower, energy, rev=True)
        # shower[:, -45:] are now unstandardized u features
        
        # 5. ExclusiveLogitTransform (reverse): Apply sigmoid with rescaling
        transform = ExclusiveLogitTransform(delta=1.0e-6, rescale=True)
        shower, energy = transform(shower, energy, rev=True)
        # u features are now in [0, 1] range
        
        # 4. SelectiveUniformNoise (reverse): Remove noise (cut applied)
        transform = SelectiveUniformNoise(
            noise_width=0.0e-6,
            cut=True,
            exclusions=[-45, -44, -43, -42, -41, -40, -39, -38, -37,
                        -36, -35, -34, -33, -32, -31, -30, -29, -28,
                        -27, -26, -25, -24, -23, -22, -21, -20, -19,
                        -18, -17, -16, -15, -14, -13, -12, -11, -10,
                        -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1]
        )
        shower, energy = transform(shower, energy, rev=True)
        
        # 3. CutValues (reverse): Restore cut values
        transform = CutValues(cut=1.0e-7)
        shower, energy = transform(shower, energy, rev=True)
        
        # 2. NormalizeByElayer (reverse): COMPUTE LAYER ENERGIES
        # This is the KEY step - it reconstructs layer energies from u features
        xml_file = getattr(self, 'xml_file', 
                          '/project/biocomplexity/fa7sa/calo_dreamer/src/challenge_files/binning_dataset_2.xml')
        transform = NormalizeByElayer(
            ptype=self.xml_file,
            xml_file= 'electron',
            return_layer_energies=return_layer_energies,
            return_summed_energies=return_summed_energies,
            layer_energy_source="voxels",   # NEW: key change
            layer_weight_temp=1.0,          # try 1.0, 0.5, 2.0
        )
        layer_energies, energy = transform(shower, energy, rev=True)
        
        
        return layer_energies,energy/1000
    def diagnose_energy_loss(self, E_true_physical, x_0_hat_physical, aux_losses_dict):
        """
        Print diagnostic information for energy loss debugging
        
        Args:
            E_true_physical: Ground truth layer energies (B, 45)
            x_0_hat_physical: Model predicted layer energies (B, 45)
            aux_losses_dict: Dictionary containing auxiliary losses
        """
        # Only print every 100 batches or at the first batch
        if self.batch_counter == 0 or self.batch_counter % 100 == 0:
            print(f"\n{'='*70}")
            print(f"ENERGY LOSS DIAGNOSTICS - Batch {self.batch_counter}, Epoch {self.epoch}")
            print(f"{'='*70}")
            
            # Ground truth diagnostics
            print(f"E_true_physical (ground truth):")
            print(f"  Shape: {E_true_physical.shape}")
            print(f"  Range: [{E_true_physical.min():.4f}, {E_true_physical.max():.4f}]")
            print(f"  Mean: {E_true_physical.mean():.4f}")
            print(f"  Std: {E_true_physical.std():.4f}")
            print(f"  Requires grad: {E_true_physical.requires_grad}")
            
            # Model prediction diagnostics
            print(f"\nx_0_hat_physical (model prediction):")
            print(f"  Shape: {x_0_hat_physical.shape}")
            print(f"  Range: [{x_0_hat_physical.min():.4f}, {x_0_hat_physical.max():.4f}]")
            print(f"  Mean: {x_0_hat_physical.mean():.4f}")
            print(f"  Std: {x_0_hat_physical.std():.4f}")
            print(f"  Requires grad: {x_0_hat_physical.requires_grad}")
            
            # Energy loss diagnostics
            if 'energy' in aux_losses_dict:
                print(f"\nEnergy Loss:")
                print(f"  Value: {aux_losses_dict['energy'].item():.6f}")
                print(f"  Requires grad: {aux_losses_dict['energy'].requires_grad}")
                
                # Compute residual manually
                residual = (x_0_hat_physical - E_true_physical).abs().mean()
                print(f"  Mean absolute residual: {residual.item():.4f} MeV")
                
                # Track history
                if not hasattr(self, 'energy_loss_history'):
                    self.energy_loss_history = []
                self.energy_loss_history.append(aux_losses_dict['energy'].item())
                
                # Show trend
                if len(self.energy_loss_history) > 1:
                    recent_losses = self.energy_loss_history[-min(10, len(self.energy_loss_history)):]
                    trend = "↓ Decreasing" if recent_losses[-1] < recent_losses[0] else "↑ Increasing"
                    print(f"  Trend (last {len(recent_losses)} checks): {trend}")
                    print(f"  Recent std: {np.std(recent_losses):.6f}")
            else:
                print(f"\n⚠️ WARNING: No 'energy' key found in aux_losses_dict!")
            
            print(f"{'='*70}\n")
        
        # Increment batch counter
        self.batch_counter += 1
    


    def energy_residual_diagnostics(
        self,
        E_true: torch.Tensor,   # (B,45) target layer energies (MeV)
        E_pred: torch.Tensor,   # (B,45) predicted layer energies (MeV)
        use_transform: bool = True,
        E_0: float = 1e-3,
        huber_delta: float = 1.0,
    ) -> dict:
        """
        Compute transformed (or raw) residual diagnostics + Huber/MSE losses between
        predicted and target per-layer energies.
    
        Returns scalars suitable for CSV logging.
        """
        assert E_true.ndim == 2 and E_pred.ndim == 2, f"Expected (B,L) tensors, got {E_true.shape}, {E_pred.shape}"
        assert E_true.shape == E_pred.shape, f"Shape mismatch: {E_true.shape} vs {E_pred.shape}"
    
        eps = 1e-12
    
        if use_transform:
            # y = 2 * sqrt(E + E_0)
            E_true_safe = torch.clamp(E_true, min=0.0)
            E_pred_safe = torch.clamp(E_pred, min=0.0)
    
            y_true = 2.0 * torch.sqrt(E_true_safe + E_0)
            y_pred = 2.0 * torch.sqrt(E_pred_safe + E_0)
    
            residual = y_pred - y_true  # (B,L)
    
            mse = (residual ** 2).mean()
            huber = F.huber_loss(y_true, y_pred, reduction="mean", delta=huber_delta)
    
            # some extra stats for sanity
            mae = residual.abs().mean()
            rel_l1 = residual.abs().sum() / (y_true.abs().sum() + eps)
    
            return {
                "Ediag_mode": "sqrt",
                "Ediag_mse": float(mse.item()),
                "Ediag_huber": float(huber.item()),
                "Ediag_mae": float(mae.item()),
                "Ediag_rel_l1": float(rel_l1.item()),
            }
    
        else:
            residual = E_pred - E_true  # (B,L)
    
            mse = (residual ** 2).mean()
            huber = F.huber_loss(E_true, E_pred, reduction="mean", delta=huber_delta)
    
            mae = residual.abs().mean()
            rel_l1 = residual.abs().sum() / (E_true.abs().sum() + eps)
    
            return {
                "Ediag_mode": "raw",
                "Ediag_mse": float(mse.item()),
                "Ediag_huber": float(huber.item()),
                "Ediag_mae": float(mae.item()),
                "Ediag_rel_l1": float(rel_l1.item()),
            }

    def batch_loss(self, x):
        """
        Calculate batch loss for diffusion model
        
        Supports 4 multi-objective optimization methods:
        1. weighted_sum: Simple weighted combination
        2. config: ModelBase ConFIG (existing implementation)
        3. uncertainty: Automatic uncertainty weighting
        4. aug_lag: Augmented Lagrangian
        
        Returns (as expected by ModelBase.py):
            total_loss: Combined loss (for backward or ConFIG)
            loss_gen: Diffusion loss (for logging and ConFIG)
            loss_aux: Auxiliary losses (for logging and ConFIG)
            comps: Dictionary of component losses for CSV logging
        """

        if self.batch_counter == 0:
            torch.autograd.set_detect_anomaly(True)
        # Get input and conditions
        x, condition, weights = self.get_condition_and_input(x)
        
        if self.latent:
            x = self.ae.encode(x, condition)
            if self.ae.kl:
                x = self.ae.reparameterize(x[0], x[1])
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
        t = t.unsqueeze(1)
        
        # CFG dropout
        if self.cfg:
            if torch.rand(1).item() < self.drop_prob:
                condition = None
        
        # Add noise to input
        if self.add_noise:
            x = x + self.gamma * torch.randn_like(x, device=x.device, dtype=x.dtype)
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Get noisy samples
        x_noisy = self.q_sample(x, t, noise=noise)
        
        # Reset KL divergence
        self.net.kl = 0
        
        # ===== GENERATIVE LOSS (DIFFUSION) =====
        if self.prediction_type == "noise":
            predicted_noise = self.net(x_noisy, t, condition)
            target = noise
            model_output = predicted_noise
        elif self.prediction_type == "x0":
            predicted_x0 = self.net(x_noisy, t, condition)
            target = x
            model_output = predicted_x0
        elif self.prediction_type == "v":
            alpha_bar = self.alphas_cumprod.to(t.device)[t].view(-1, *[1] * (x_noisy.ndim - 1))
            sqrt_alpha = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha = torch.sqrt(1. - alpha_bar)
            v_target = sqrt_alpha * noise + sqrt_one_minus_alpha * x
            v_pred = self.net(x_noisy, t, condition)
            target = v_target
            model_output = v_pred
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Base diffusion/generative loss
        loss_gen = torch.mean((model_output - target) ** 2)
        
        # Initialize component dictionary for CSV logging
        comps = {
            'CFD_metric': 0.0,
            'mvn_loss': 0.0,
            'energy_loss': 0.0,
            'mse_noise': 0.0,
            'mse_x0': 0.0,
            'mse_v': 0.0,
            'loss_param': 0.0,
            'cfd_raw': 0.0,
            'w_t_mean': 0.0,
            'lambda_warm': 0.0,
            'lambda_cfd_eff': 0.0,
            'r_eff': 0.0,
            'cfd_fro_pred': 0.0,
            'cfd_fro_shrunk': 0.0,
            'cfd_fro_smooth': 0.0,
            'cfd_ema_dist': 0.0,
            'mvn_mode': 0.0,
            'mvn_trace_val': 0.0,
            'raw_mvn': 0.0,
            'Ediag_mode': "",
            'Ediag_mse': 0.0,
            'Ediag_huber': 0.0,
            'Ediag_mae': 0.0,
            'Ediag_rel_l1': 0.0,
        }
        
        # Store MSE for appropriate prediction type
        if self.prediction_type == "noise":
            comps['mse_noise'] = loss_gen.item()
        elif self.prediction_type == "x0":
            comps['mse_x0'] = loss_gen.item()
        elif self.prediction_type == "v":
            comps['mse_v'] = loss_gen.item()
        
        # ===== AUXILIARY LOSSES =====
        loss_aux = torch.tensor(0.0, device=x.device)
        
        if self.use_aux_losses:
            aux_losses_dict={}
            with torch.no_grad():
                E_true_physical, Einc_true = self.apply_inverse_transforms(x.clone(), condition.clone(),return_layer_energies=True, return_summed_energies=False)
            

           
            # Get predicted clean sample (mean estimation)
            
            if self.prediction_type == "noise":
                x_0_hat = self.predict_start_from_noise(x_noisy, t, predicted_noise)
            elif self.prediction_type == "x0":
                x_0_hat = predicted_x0
            elif self.prediction_type == "v":
                alpha_bar = self.alphas_cumprod.to(t.device)[t].view(-1, *[1] * (x_noisy.ndim - 1))
                sqrt_alpha = torch.sqrt(alpha_bar)
                sqrt_one_minus_alpha = torch.sqrt(1. - alpha_bar)
                x_0_hat = sqrt_alpha * x_noisy - sqrt_one_minus_alpha * v_pred
            
            # ===== CRITICAL: APPLY INVERSE TRANSFORMS =====
            x_0_hat_physical,Einc_pred = self.apply_inverse_transforms(x_0_hat.clone(), condition.clone(),return_layer_energies=False, return_summed_energies=True)
            #x_0_hat_physical.requires_grad_(True)
            
            Einc_gev = Einc_true.squeeze(1).to(device=x_0_hat_physical.device)
            # Extract E_pred from condition
            # MODIFY THIS based on your condition structure!
            E_pred = E_true_physical  # (B, 45) 
            if self.aux_loss_manager.use_energy:
                
                aux_losses_dict['energy'] = self.aux_loss_manager.compute_losses(
                x_0_hat_physical,
                E_pred=E_pred
            )
            loss_aux=aux_losses_dict['energy']
            
            #mvn_loss=self.mvn_loss_binned(x_0_hat_physical,Einc_gev)
            mvn_loss=self.mvn_loss_binned_corr(x_0_hat_physical,Einc_gev)
            print("mvn_loss.requires_grad:", mvn_loss.requires_grad)
            print("mvn_loss.grad_fn:", mvn_loss.grad_fn)

            aux_losses_dict['mvn']=mvn_loss
            # Compute all auxiliary losses
            # aux_losses_dict = self.aux_loss_manager.compute_losses(
            #     x_0_hat_physical,
            #     E_pred=E_pred
            # )
            #self.diagnose_energy_loss(E_true_physical, x_0_hat_physical, aux_losses_dict)
            # Update logging components
            # diagnostics for CSV
            diag = self.energy_residual_diagnostics(
                E_pred, x_0_hat_physical,
                use_transform=True,     # or False
                E_0=1e-3,
                huber_delta=1.0
            )
            # ===== log layer-energy diagnostics =====
            if diag is not None:
                comps['Ediag_mode']   = diag.get('Ediag_mode', "")
                comps['Ediag_mse']    = diag.get('Ediag_mse', 0.0)
                comps['Ediag_huber']  = diag.get('Ediag_huber', 0.0)
                comps['Ediag_mae']    = diag.get('Ediag_mae', 0.0)
                comps['Ediag_rel_l1'] = diag.get('Ediag_rel_l1', 0.0)
            if 'energy' in aux_losses_dict:
                comps['energy_loss'] = aux_losses_dict['energy'].item()
            if 'mvn' in aux_losses_dict:
                comps['mvn_loss'] = aux_losses_dict['mvn'].item()
                comps['raw_mvn'] = aux_losses_dict['mvn'].item()
            if 'cfd' in aux_losses_dict:
                comps['CFD_metric'] = aux_losses_dict['cfd'].item()
                comps['cfd_raw'] = aux_losses_dict['cfd'].item()
            
        #     # ===== MULTI-OBJECTIVE OPTIMIZATION =====
        #     if self.mo_method == 'weighted_sum':
        #         # Simple weighted combination
        #         loss_aux = torch.tensor(0.0, device=x.device)
        #         if 'energy' in aux_losses_dict:
        #             loss_aux = loss_aux + self.lambda_energy * aux_losses_dict['energy']
        #         if 'mvn' in aux_losses_dict:
        #             loss_aux = loss_aux + self.lambda_mvn * aux_losses_dict['mvn']
        #         if 'cfd' in aux_losses_dict:
        #             loss_aux = loss_aux + self.lambda_cfd * aux_losses_dict['cfd']
                
        #         total_loss = loss_gen + loss_aux
            
        #     elif self.mo_method == 'config':
        #         # Let ModelBase handle ConFIG (your existing implementation)
        #         # Just return weighted sum for logging
        #         loss_aux = torch.tensor(0.0, device=x.device)
        #         if 'energy' in aux_losses_dict:
        #             loss_aux = loss_aux + aux_losses_dict['energy']
        #         if 'mvn' in aux_losses_dict:
        #             loss_aux = loss_aux +  aux_losses_dict['mvn']
        #         if 'cfd' in aux_losses_dict:
        #             loss_aux = loss_aux + self.lambda_cfd * aux_losses_dict['cfd']
                
        #         total_loss = loss_gen + loss_aux
            
        #     elif self.mo_method == 'uncertainty':
        #         # Automatic uncertainty weighting
        #         all_losses = {'diffusion': loss_gen}
        #         all_losses.update(aux_losses_dict)
                
        #         total_loss = self.mo_optimizer(all_losses)
        #         loss_aux = sum(aux_losses_dict.values())
            
        #     elif self.mo_method == 'aug_lag':
        #         # Augmented Lagrangian
        #         constraints = self.aux_loss_manager.get_constraint_violations(
        #             x_0_hat_physical,
        #             E_pred=E_pred
        #         )
                
        #         all_losses = {'diffusion': loss_gen}
        #         all_losses.update(aux_losses_dict)
                
        #         al_term = self.mo_optimizer(
        #             all_losses,
        #             constraint_violations=constraints
        #         )
                
        #         total_loss = loss_gen + al_term
        #         loss_aux = sum(aux_losses_dict.values())
            
        #     else:
        #         # Fallback: just diffusion loss
        #         total_loss = loss_gen
        
        # else:
        #     # No auxiliary losses
        #     total_loss = loss_gen
        
        # Return 4 values as ModelBase expects:
        # - total_loss: for standard backward (or ConFIG input)
        # - loss_gen: for ConFIG gradient 1 (if use_config=True)
        # - loss_aux: for ConFIG gradient 2 (if use_config=True)
        # - comps: for CSV logging
        print("loss_gen.requires_grad:", loss_gen.requires_grad)
        print("loss_aux.requires_grad:", loss_aux.requires_grad)
        
        print("loss_gen.grad_fn:", loss_gen.grad_fn)
        print("loss_aux.grad_fn:", loss_aux.grad_fn)

        return loss_gen, loss_aux, comps

    def mvn_loss_binned_corr(self, x_layer: torch.Tensor, Einc_gev: torch.Tensor, eps_std: float = 1e-6) -> torch.Tensor:
        """
        Correlation-Mahalanobis (per-bin):
          z = (x - mu) / std
          d^2 = || chol_R^{-1} z ||^2
    
        x_layer: (B,45) predicted layer energies in SAME units as targets (your targets are MeV layer sums)
        Einc_gev: (B,) incident energy in GeV for binning
        """
        assert x_layer.ndim == 2 and x_layer.shape[1] == 45, f"x_layer must be (B,45), got {tuple(x_layer.shape)}"
        assert Einc_gev.ndim == 1, f"Einc_gev must be (B,), got {tuple(Einc_gev.shape)}"
    
        device = x_layer.device
        dtype = x_layer.dtype
        Einc_gev = Einc_gev.to(device=device)
    
        total_sum = torch.zeros((), device=device, dtype=dtype)
        total_count = 0
    
        for lo, hi, key in self.mvn_bin_list:
            mask = (Einc_gev >= lo) & (Einc_gev < hi)
            n = int(torch.count_nonzero(mask).item())
            if n == 0:
                continue
    
            bin_data = self.mvn_binned["bins"].get(key, None)
            if (bin_data is None) or ("mu" not in bin_data) or ("std" not in bin_data) or ("chol_R" not in bin_data):
                continue
    
            mu = bin_data["mu"].to(device=device, dtype=dtype)              # (45,)
            std = bin_data["std"].to(device=device, dtype=dtype)            # (45,)
            chol_R = bin_data["chol_R"].to(device=device, dtype=dtype)      # (45,45)
    
            std = torch.clamp(std, min=eps_std)
    
            xb = x_layer[mask]                                              # (n,45)
            z = (xb - mu.unsqueeze(0)) / std.unsqueeze(0)                   # (n,45) standardized
            y = torch.linalg.solve_triangular(chol_R, z.T, upper=False)     # (45,n)
            d2 = (y * y).sum(dim=0)                                         # (n,)
    
            total_sum = total_sum + d2.sum()
            total_count += n
    
        if total_count == 0:
            return torch.zeros((), device=device, dtype=dtype)
    
        return total_sum / total_count

    def mvn_loss_binned(self, x_layer: torch.Tensor, Einc_gev: torch.Tensor) -> torch.Tensor:
        assert x_layer.ndim == 2 and x_layer.shape[1] == 45, f"x_layer must be (B,45), got {tuple(x_layer.shape)}"
        assert Einc_gev.ndim == 1, f"Einc_gev must be (B,), got {tuple(Einc_gev.shape)}"
    
        device = x_layer.device
        dtype = x_layer.dtype
        Einc_gev = Einc_gev.to(device=device)
    
        total_sum = torch.zeros((), device=device, dtype=dtype)
        total_count = 0
        
        for lo, hi, key in self.mvn_bin_list:
            mask = (Einc_gev >= lo) & (Einc_gev < hi)
            n = int(torch.count_nonzero(mask).item())
            if n == 0:
                continue
            
            bin_data = self.mvn_binned["bins"].get(key, None)
            if (bin_data is None) or ("mu" not in bin_data) or ("chol" not in bin_data):
                continue
    
            mu = bin_data["mu"].to(device=device, dtype=dtype)        # (45,)
            chol = bin_data["chol"].to(device=device, dtype=dtype)    # (45,45)
    
            xb = x_layer[mask]                                        # (n,45)
            diff = xb - mu.unsqueeze(0)                               # (n,45)
            y = torch.linalg.solve_triangular(chol, diff.T, upper=False)  # (45,n)
            d2 = (y * y).sum(dim=0)                                   # (n,)
    
            total_sum = total_sum + d2.sum()
            total_count += n
    
        if total_count == 0:
            return torch.zeros((), device=device, dtype=dtype)
        
        return total_sum / total_count


    def update_augmented_lagrangian(self, x):
        """
        Update Lagrange multipliers for Augmented Lagrangian method
        Call this AFTER optimizer.step() in training loop (if using aug_lag)
        """
        if self.mo_method != 'aug_lag' or not self.use_aux_losses:
            return
        
        # Re-compute constraint violations (no gradients)
        with torch.no_grad():
            x, condition, _ = self.get_condition_and_input(x)
            
            if self.latent:
                x = self.ae.encode(x, condition)
                if self.ae.kl:
                    x = self.ae.reparameterize(x[0], x[1])
            
            t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
            t = t.unsqueeze(1)
            
            noise = torch.randn_like(x)
            x_noisy = self.q_sample(x, t, noise=noise)
            
            # Get x_0_hat
            if self.prediction_type == "noise":
                predicted_noise = self.net(x_noisy, t, condition)
                x_0_hat = self.predict_start_from_noise(x_noisy, t, predicted_noise)
            else:
                x_0_hat = self.net(x_noisy, t, condition)
            
            # Apply inverse transforms
            x_0_hat_physical = self.apply_inverse_transforms(x_0_hat, condition)
            
            # Extract E_pred
            E_pred = condition[:, :45] if condition is not None else None
            
            # Get constraint violations
            constraints = self.aux_loss_manager.get_constraint_violations(
                x_0_hat_physical,
                E_pred=E_pred
            )
            
            # Update multipliers
            self.mo_optimizer.update_multipliers(constraints)

    def setup_auxiliary_targets(self, train_loader):
        """
        Pre-compute target statistics from training data
        Call this BEFORE starting training!
        """
        if not self.use_aux_losses:
            print("No auxiliary losses enabled, skipping target setup")
            return
        
        print("=" * 70)
        print("Setting up auxiliary loss targets from training data...")
        print("=" * 70)
        
        self.aux_loss_manager.set_targets_from_data(train_loader, device=self.device)
        
        print("Auxiliary loss targets ready!")
        print("=" * 70)

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(x_t.device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t]
        
        while len(sqrt_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
        while len(sqrt_one_minus_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def p_mean_variance(self, x, t, condition=None):
        """Compute the mean and variance for posterior q(x_{t-1} | x_t, x_0)"""
        if self.prediction_type == "noise":
            predicted_noise = self.net(x, t, condition)
            x_recon = self.predict_start_from_noise(x, t, predicted_noise)
        elif self.prediction_type == "x0":
            x_recon = self.net(x, t, condition)
        elif self.prediction_type == "v":
            alpha_bar = self.alphas_cumprod.to(t.device)[t].view(-1, *[1] * (x.ndim - 1))
            sqrt_alpha = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha = torch.sqrt(1. - alpha_bar)
            v_pred = self.net(x, t, condition)
            x_recon = sqrt_alpha * x - sqrt_one_minus_alpha * v_pred
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        posterior_mean_coef1_t = self.posterior_mean_coef1.to(x.device)[t]
        posterior_mean_coef2_t = self.posterior_mean_coef2.to(x.device)[t]
        posterior_variance_t = self.posterior_variance.to(x.device)[t]
        
        while len(posterior_mean_coef1_t.shape) < len(x.shape):
            posterior_mean_coef1_t = posterior_mean_coef1_t.unsqueeze(-1)
        while len(posterior_mean_coef2_t.shape) < len(x.shape):
            posterior_mean_coef2_t = posterior_mean_coef2_t.unsqueeze(-1)
        while len(posterior_variance_t.shape) < len(x.shape):
            posterior_variance_t = posterior_variance_t.unsqueeze(-1)
        
        model_mean = posterior_mean_coef1_t * x_recon + posterior_mean_coef2_t * x
        model_variance = posterior_variance_t
        
        return model_mean, model_variance, x_recon

    def p_sample(self, x, t, condition=None):
        """Sample x_{t-1} from p(x_{t-1} | x_t)"""
        model_mean, model_variance, _ = self.p_mean_variance(x, t, condition)
        
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float()
        while len(nonzero_mask.shape) < len(x.shape):
            nonzero_mask = nonzero_mask.unsqueeze(-1)
        
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise

    def sample_batch(self, batch):
        """Generate samples for a batch"""
        dtype = batch.dtype
        device = batch.device
        
        x = torch.randn((batch.shape[0], *self.shape), dtype=dtype, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch.shape[0],), i, device=device, dtype=torch.long)
            t = t.unsqueeze(1)
            x = self.p_sample(x, t, batch if isinstance(batch, torch.Tensor) else None)
        
        if self.latent:
            x = self.ae.decode(x, batch)
            
        return x

    def sample_batch_causal(self, condition):
        """Generate samples layer-by-layer with causal masking"""
        dtype = condition.dtype
        device = condition.device
        B = condition.shape[0]

        if hasattr(self, 'shape') and self.shape is not None:
            C, L, H, W = self.shape
        else:
            C, L, H, W = 1, 45, 16, 9

        x = torch.zeros((B, C, L, H, W), dtype=dtype, device=device)

        for l in range(L):
            x[:, :, l:l+1, :, :] = torch.randn((B, C, 1, H, W), dtype=dtype, device=device)

            for i in reversed(range(self.timesteps)):
                t = torch.full((B,), i, device=device, dtype=torch.long)
                t = t.unsqueeze(1)

                model_mean, model_var, _ = self.p_mean_variance(x, t, condition)
                mean_l = model_mean[:, :, l:l+1, :, :]
                var_l = model_var[:, :, l:l+1, :, :]
                
                noise = torch.randn_like(mean_l)
                nonzero_mask = (t != 0).float().view(B, 1, 1, 1, 1)
                x[:, :, l:l+1, :, :] = mean_l + nonzero_mask * torch.sqrt(var_l) * noise

        if self.latent:
            x = self.ae.decode(x, condition)

        return x

    def ddim_sample(self, batch, eta=0.0, ddim_steps=50):
        """DDIM sampling for faster generation"""
        dtype = batch.dtype
        device = batch.device
        
        c = self.timesteps // ddim_steps
        ddim_timesteps = torch.arange(0, self.timesteps, c, device=device)
        ddim_timesteps = ddim_timesteps + 1
        
        x = torch.randn((batch.shape[0], *self.shape), dtype=dtype, device=device)
        
        for i in reversed(range(len(ddim_timesteps))):
            t = ddim_timesteps[i]
            prev_t = ddim_timesteps[i - 1] if i > 0 else 0
            
            t_batch = torch.full((batch.shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = self.net(x, t_batch, batch if isinstance(batch, torch.Tensor) else None)
            
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[prev_t] if prev_t > 0 else torch.tensor(1.0)
            
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            dir_xt = torch.sqrt(1 - alpha_prev - eta**2 * (1 - alpha_t) / (1 - alpha_prev)) * predicted_noise
            
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t) * noise
        
        if self.latent:
            x = self.ae.decode(x, batch)
            
        return x

    def sample_n_evolution(self, n_samples, use_ddim=False, ddim_steps=50):
        """Generate n samples"""
        if self.net.bayesian:
            self.net.map = get(self.params, "fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        
        self.eval()
        batch_size = get(self.params, "batch_size", 64)
        
        samples = []
        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                current_batch_size = min(batch_size, n_samples - i)
                
                if use_ddim:
                    dummy_batch = torch.zeros(current_batch_size, device=self.device)
                    batch_samples = self.ddim_sample(dummy_batch, ddim_steps=ddim_steps)
                else:
                    dummy_batch = torch.zeros(current_batch_size, device=self.device)
                    batch_samples = self.sample_batch(dummy_batch)
                
                samples.append(batch_samples.detach().cpu())
        
        return torch.cat(samples, dim=0)[:n_samples]

    def invert_n(self, samples):
        """Invert samples through the forward process"""
        if self.net.bayesian:
            self.net.map = get(self.params, "fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        
        self.eval()
        batch_size = get(self.params, "batch_size", 64)
        n_samples = samples.shape[0]
        
        inverted_samples = []
        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                current_batch_size = min(batch_size, n_samples - i)
                batch_samples = samples[i:i+current_batch_size].to(self.device)
                
                t = torch.randint(0, self.timesteps, (current_batch_size,), device=self.device)
                noise = torch.randn_like(batch_samples)
                noisy_samples = self.q_sample(batch_samples, t, noise)
                
                inverted_samples.append(noisy_samples.detach().cpu())
        
        return torch.cat(inverted_samples, dim=0)[:n_samples]