import numpy as np
import torch
from scipy.integrate import solve_ivp
import Networks
from Util.util import get
from Models.ModelBase import GenerativeModel
import Networks
import Models
from torchdiffeq import odeint
import math
from torch import linalg as LA
torch.cuda.empty_cache()
# ---- Loss helpers (adapted from your toy code) ----
def mmd_loss(x, y, sigma=1.0):
    """Maximum Mean Discrepancy (RBF kernel)."""
    dist_xx = torch.cdist(x, x)
    dist_yy = torch.cdist(y, y)
    dist_xy = torch.cdist(x, y)
    K_xx = torch.exp(-dist_xx**2 / (2 * sigma**2)).mean()
    K_yy = torch.exp(-dist_yy**2 / (2 * sigma**2)).mean()
    K_xy = torch.exp(-dist_xy**2 / (2 * sigma**2)).mean()
    return K_xx + K_yy - 2 * K_xy

def mmd_poly2(x, y, c=0.0):
    """Polynomial kernel MMD^2 loss."""
    xx = (x @ x.T + c) ** 2
    yy = (y @ y.T + c) ** 2
    xy = (x @ y.T + c) ** 2
    Nx, Ny = x.size(0), y.size(0)
    return (xx.sum() - xx.diag().sum()) / (Nx*(Nx-1) + 1e-6) \
         + (yy.sum() - yy.diag().sum()) / (Ny*(Ny-1) + 1e-6) \
         - 2.0 * xy.mean()
def correlation_matrix(x):
    """Compute correlation matrix across features (batch dimension = samples)."""
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.T @ x / (x.size(0) - 1)
    d = torch.sqrt(torch.diag(cov) + 1e-8)
    corr = cov / torch.ger(d, d)
    return corr

def correlation_matrix_loss(x, ref_corr):
    corr = correlation_matrix(x)
    num = torch.sum((corr - ref_corr) ** 2).sqrt()   # Frobenius norm
    denom = torch.sum(ref_corr ** 2).sqrt() + 1e-8
    return num / denom


class TBD_DIFF(GenerativeModel):
    """
    Class for Trajectory Based Diffusion - converted to proper diffusion model
    Inheriting from the GenerativeModel BaseClass
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
        
        
        # ------------------ New: Regularization Loss Params ------------------
        # MMD losses
        self.use_mmd    = get(self.params, "use_mmd", False)
        self.mmd_weight = get(self.params, "mmd_weight", 0.05)   # strength of RBF MMD
        self.use_mmd2   = get(self.params, "use_mmd2", False)
        self.mmd2_weight = get(self.params, "mmd2_weight", 0.05) # strength of poly MMD^2

        # Correlation loss
        self.use_corr    = get(self.params, "use_corr", False)
        self.corr_weight = get(self.params, "corr_weight", 0.1) # weight on correlation matrix loss
        # Initialize noise schedule
        self.setup_noise_schedule()
        
        # Other parameters : what is this?
        self.C = get(self.params, "C", 1)
        if self.C != 1:
            print(f"C is {self.C}")

        self.bayesian = get(self.params, "bayesian", 0)
        self.add_noise = get(self.params, "add_noise", False)
        self.gamma = get(self.params, "gamma", 1.e-4)
        
        # Prediction type: 'noise' or 'x0'
        self.prediction_type = get(self.params, "prediction_type", "noise")
        
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
        """
        Build the network
        """
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

    def batch_loss(self, x):
        """
        Calculate batch loss for diffusion model
        """
        #print("batch_loss in diffusion model")
        # get input and conditions
        x, condition, weights = self.get_condition_and_input(x)
        
        # if condition is not None:
        #     print("condition shape: ", condition.shape)
        
        if self.latent:
            # encode x into autoencoder latent space
            #print("inside the latent")
            x = self.ae.encode(x, condition)
            if self.ae.kl:
                x = self.ae.reparameterize(x[0], x[1])
        assert torch.all(torch.isfinite(x)), "NaN or Inf detected in input x"
        #print("shape of tbd_diff's x in batch_loss: ",x.shape)
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
        t = t.unsqueeze(1) 
        #print("inside batch loss shape of t:: ",t.shape)
        
        if self.cfg:
            if torch.rand(1).item() < self.drop_prob:
                print("unconditioned")
                condition = None
        
        
        # Add noise to input
        if self.add_noise:
            x = x + self.gamma * torch.randn_like(x, device=x.device, dtype=x.dtype)
        assert torch.all(torch.isfinite(x)), "NaN or Inf after noise injection"
            
        # Sample noise
        noise = torch.randn_like(x)
        assert torch.all(torch.isfinite(noise)), "NaN or Inf in noise"
        
        # Get noisy samples
        x_noisy = self.q_sample(x, t, noise=noise)
        assert torch.all(torch.isfinite(x_noisy)), "NaN or Inf in x_noisy"

        # Reset KL divergence
        self.net.kl = 0
        
        #print("right before calling the net in diffusion model: ",x_noisy.shape, t.shape, condition.shape)
        # Predict noise or x0
        if self.prediction_type == "noise":
            # Predict the noise
            predicted_noise = self.net(x_noisy, t, condition) #this pattern is for vit.py
            assert torch.all(torch.isfinite(predicted_noise)), "NaN or Inf in predicted_noise"
            # attn_maps = self.net.get_all_attention_maps()
            # visualize_attention(attn_maps)
            #predicted_noise = self.net(condition,x_noisy, t,x) # this pattern is for transformer_voxel.py
            target = noise
        elif self.prediction_type == "x0":
            # Predict the original image
            predicted_x0 = self.net(x_noisy, t, condition)
            assert torch.all(torch.isfinite(predicted_x0)), "NaN or Inf in predicted_x0"
            target = x
        elif self.prediction_type == "v":
            # Predict velocity: v = sqrt(ᾱ) * ε + sqrt(1 - ᾱ) * x₀
            alpha_bar = self.alphas_cumprod.to(t.device)[t].view(-1, *[1] * (x_noisy.ndim - 1))
            sqrt_alpha = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha = torch.sqrt(1. - alpha_bar)
            
            assert torch.all(torch.isfinite(sqrt_alpha)), "NaN or Inf in sqrt_alpha"
            assert torch.all(torch.isfinite(sqrt_one_minus_alpha)), "NaN or Inf in sqrt_one_minus_alpha"

            v_target = sqrt_alpha * noise + sqrt_one_minus_alpha * x
            assert torch.all(torch.isfinite(v_target)), "NaN or Inf in v_target"
            
            v_pred = self.net(x_noisy, t, condition)
            assert torch.all(torch.isfinite(v_pred)), "NaN or Inf in v_pred"
            target = v_target
        
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Calculate loss
        if self.prediction_type == "noise":
            loss = torch.mean((predicted_noise - target) ** 2)
        elif self.prediction_type=='v':
            loss= torch.mean((v_pred - target) ** 2)
        else:
            loss = torch.mean((predicted_x0 - target) ** 2)
            
        #self.regular_loss.append(loss.detach().cpu().numpy())
        
        # Add KL loss if using Bayesian network
        # if self.C != 0:
        #     kl_loss = self.C * self.net.kl / self.n_traindata
        #     self.kl_loss.append(kl_loss.cpu().numpy())
        #     loss = loss + kl_loss
        
        # if not torch.isfinite(loss):
        #     print("LOSS IS NOT FINITE!")
        #     print("loss:", loss.item())
        #     print("x:", x.min().item(), x.max().item())
        #     print("x_noisy:", x_noisy.min().item(), x_noisy.max().item())
        #     print("target:", target.min().item(), target.max().item())
        #     if self.prediction_type == "v":
        #         print("v_pred:", v_pred.min().item(), v_pred.max().item())
                
        # 1. MMD Loss (RBF)
        if self.use_mmd:
            # Flatten spatial dims: (B, 45*16*9) for voxel data or (B, 45) for layer sums
            x_ref = x.view(x.size(0), -1)
            x_hat = target.view(target.size(0), -1)
            # Median heuristic for sigma
            with torch.no_grad():
                d2 = torch.cdist(x_hat, x_ref).pow(2)
                sigma = torch.sqrt(d2.median() / 2 + 1e-8).item()
            mmd_val = mmd_loss(x_hat, x_ref, sigma=sigma)
            loss = loss + self.mmd_weight * mmd_val

        # 2. MMD^2 (poly2)
        if self.use_mmd2:
            x_ref = x.view(x.size(0), -1)
            x_hat = target.view(target.size(0), -1)
            mmd2_val = mmd_poly2(x_hat, x_ref)
            loss = loss + self.mmd2_weight * mmd2_val

        # 3. Correlation Loss (layer-level, like CFD)
        if self.use_corr:
            # Sum voxels into layer energy: (B, 45)
            x_ref_layer = x.view(x.size(0), x.size(1), -1).sum(-1).squeeze(1)
            x_hat_layer = target.view(target.size(0), target.size(1), -1).sum(-1).squeeze(1)
            corr_loss_val = correlation_matrix_loss(x_hat_layer, self.ref_corr.to(x.device))
            loss = loss + self.corr_weight * corr_loss_val

        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from noise prediction
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(x_t.device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
        while len(sqrt_one_minus_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1 = self.posterior_mean_coef1.to(x_t.device)[t]
        posterior_mean_coef2 = self.posterior_mean_coef2.to(x_t.device)[t]
        posterior_variance = self.posterior_variance.to(x_t.device)[t]
        
        # Reshape for broadcasting
        while len(posterior_mean_coef1.shape) < len(x_t.shape):
            posterior_mean_coef1 = posterior_mean_coef1.unsqueeze(-1)
        while len(posterior_mean_coef2.shape) < len(x_t.shape):
            posterior_mean_coef2 = posterior_mean_coef2.unsqueeze(-1)
        while len(posterior_variance.shape) < len(x_t.shape):
            posterior_variance = posterior_variance.unsqueeze(-1)
        
        posterior_mean = (
            posterior_mean_coef1 * x_start + 
            posterior_mean_coef2 * x_t
        )
        #added to get rid of autoregressive error
        posterior_variance = posterior_variance.expand_as(x_t)
        
        return posterior_mean, posterior_variance

    def p_mean_variance(self, x_t, t, condition=None):
        """
        Apply the model to get p(x_{t-1} | x_t)
        """
        #model_output = self.net(condition, x_t, t,x=None, rev=False,  )
        
        if getattr(self, 'cfg', False):  # only use CFG if explicitly enabled
            eps_cond = self.net(x_t, t, condition)
            eps_uncond = self.net(x_t, t, None)
            guidance_scale = getattr(self, 'cfg_scale', 1.0)
            model_output = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            model_output = self.net(x_t, t, condition)
            
        if self.prediction_type == "noise":
            # Model predicts noise
            x_start = self.predict_start_from_noise(x_t, t, model_output)
        elif self.prediction_type == "x0":
            # Model predicts x_0
            x_start = model_output
        elif self.prediction_type == "v":
            self.alphas_cumprod = self.alphas_cumprod.to(self.device)
            alpha_t = self.alphas_cumprod[t].view(-1, *[1] * (x_t.ndim - 1))
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha = torch.sqrt(1. - alpha_t)
            x_start = sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        model_mean, model_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        
        return model_mean, model_variance, x_start

    def p_sample(self, x_t, t, condition=None):
        """
        Sample x_{t-1} from the model
        """
        model_mean, model_variance, x_start = self.p_mean_variance(x_t, t, condition)
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float()
        while len(nonzero_mask.shape) < len(x_t.shape):
            nonzero_mask = nonzero_mask.unsqueeze(-1)
        
        sample = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        return sample

    @torch.inference_mode()
    def sample_batch(self, batch):
        """
        Generate samples using the reverse diffusion process
        """
        print("AM I here??")
        dtype = batch.dtype
        device = batch.device
        
        # Start from pure noise
        x = torch.randn((batch.shape[0], *self.shape), dtype=dtype, device=device)
        print("shape of x: ",x.shape)
        
        # Reverse diffusion process
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch.shape[0],), i, device=device, dtype=torch.long)
            t=t.unsqueeze(1)
            x = self.p_sample(x, t, batch if isinstance(batch, torch.Tensor) else None)
        
        if self.latent:
            # decode the generated sample
            x = self.ae.decode(x, batch)
            
        return x
    
#     @torch.inference_mode()
#     def sample_batch(self,batch):
#         diffusion_params = {
#             'sqrt_alphas_cumprod': self.sqrt_alphas_cumprod,
#             'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod,
#             'posterior_mean_coef1': self.posterior_mean_coef1,
#             'posterior_mean_coef2': self.posterior_mean_coef2,
#             'posterior_variance': self.posterior_variance,
#         }
#         dtype = batch.dtype
#         device = batch.device
        
#         sample = self.net(
#             batch, x_t=None, t=None, x=None, rev=True,
#             diffusion_buffers=diffusion_params,
#             sampling_type='ddpm'
#         )
#         return sample
        
        
    
    
    @torch.inference_mode()
    def autoregressive_sample_batch(self, condition):
        """
        Generate samples autoregressively over layers using the reverse diffusion process.
        Assumes the model uses causal attention.
        """
        dtype = condition.dtype
        device = condition.device
        B = condition.shape[0]

        # Get the shape from the condition or define it based on your model
        # You need to set this correctly based on your model architecture
        if hasattr(self, 'shape') and self.shape is not None:
            C, L, H, W = self.shape
        else:
            # Fallback: infer from condition or set manually
            # This is a fallback - you should set self.shape properly in your model
            C, L, H, W = 1, 45, 16, 9  # MODIFY THESE VALUES TO MATCH YOUR MODEL
            print(f"Warning: using fallback shape: {C, L, H, W}")

        # Initialize with zeros (we'll fill each layer as we generate it)
        x = torch.zeros((B, C, L, H, W), dtype=dtype, device=device)

        # Generate each layer sequentially
        for l in range(L):
            # Initialize current layer with noise
            #print("----currently dealing with layer number: ", l)
            x[:, :, l:l+1, :, :] = torch.randn((B, C, 1, H, W), dtype=dtype, device=device)
            #print("shape of x before the time loop: ",x.shape)

            # Run reverse diffusion for the current layer
            for i in reversed(range(self.timesteps)):
                #print("current timestep: ",i)
                t = torch.full((B,), i, device=device, dtype=torch.long)
                t = t.unsqueeze(1)  # Add the missing dimension

                # Use the current state of x for prediction
                # This naturally implements causal masking since:
                # - Previous layers (0 to l-1) are fully generated
                # - Current layer (l) is being sampled  
                # - Future layers (l+1 to L-1) are zero

                # Get model prediction
                model_mean, model_var, _ = self.p_mean_variance(x, t, condition)
                #print("shape of model_mean and model_var: ", model_mean.shape, model_var.shape)

                # Extract prediction for current layer only
                mean_l = model_mean[:, :, l:l+1, :, :]
                var_l = model_var[:, :, l:l+1, :, :]
                #print("shape of mean_l and var_l: ", mean_l.shape, var_l.shape)
                # Sample using reparameterization trick
                noise = torch.randn_like(mean_l)
                #print("shape of noise: ",noise.shape)
                nonzero_mask = (t != 0).float().view(B, 1, 1, 1, 1)
                #print("shape of nonzero_mask: ", nonzero_mask.shape)
                # Update current layer only
                x[:, :, l:l+1, :, :] = mean_l + nonzero_mask * torch.sqrt(var_l) * noise

        # Decode if using latent space
        if self.latent:
            x = self.ae.decode(x, condition)

        return x

    def ddim_sample(self, batch, eta=0.0, ddim_steps=50):
        """
        DDIM sampling for faster generation
        """
        dtype = batch.dtype
        device = batch.device
        
        # Create DDIM timestep schedule
        c = self.timesteps // ddim_steps
        ddim_timesteps = torch.arange(0, self.timesteps, c, device=device)
        ddim_timesteps = ddim_timesteps + 1  # Start from 1
        
        # Start from pure noise
        x = torch.randn((batch.shape[0], *self.shape), dtype=dtype, device=device)
        
        # Reverse DDIM process
        for i in reversed(range(len(ddim_timesteps))):
            t = ddim_timesteps[i]
            prev_t = ddim_timesteps[i - 1] if i > 0 else 0
            
            # Predict noise
            t_batch = torch.full((batch.shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = self.net(x, t_batch, batch if isinstance(batch, torch.Tensor) else None)
            
            # Predict x_0
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[prev_t] if prev_t > 0 else torch.tensor(1.0)
            
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Compute direction to x_t
            dir_xt = torch.sqrt(1 - alpha_prev - eta**2 * (1 - alpha_t) / (1 - alpha_prev)) * predicted_noise
            
            # Compute x_{t-1}
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t) * noise
        
        if self.latent:
            # decode the generated sample
            x = self.ae.decode(x, batch)
            
        return x

    def sample_n_evolution(self, n_samples, use_ddim=False, ddim_steps=50):
        """
        Generate n samples
        """
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
                    # Use DDIM for faster sampling
                    dummy_batch = torch.zeros(current_batch_size, device=self.device)
                    batch_samples = self.ddim_sample(dummy_batch, ddim_steps=ddim_steps)
                else:
                    # Use standard DDPM sampling
                    dummy_batch = torch.zeros(current_batch_size, device=self.device)
                    batch_samples = self.sample_batch(dummy_batch)
                
                samples.append(batch_samples.detach().cpu())
        
        return torch.cat(samples, dim=0)[:n_samples]

    def invert_n(self, samples):
        """
        Invert samples through the forward process (for analysis)
        """
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
                
                # Add noise gradually (forward process)
                t = torch.randint(0, self.timesteps, (current_batch_size,), device=self.device)
                noise = torch.randn_like(batch_samples)
                noisy_samples = self.q_sample(batch_samples, t, noise)
                
                inverted_samples.append(noisy_samples.detach().cpu())
        
        return torch.cat(inverted_samples, dim=0)[:n_samples]
    
# def compute_attention_rollout(attn_matrices):
#     """
#     attn_matrices: list of [B, heads, N, N] tensors from all layers
#     Returns: [B, N, N] cumulative attention map
#     """
#     rollout = None
#     for attn in attn_matrices:
#         attn_avg = attn.mean(dim=1)  # average over heads → [B, N, N]
#         attn_with_residual = attn_avg + torch.eye(attn_avg.size(-1), device=attn_avg.device)
#         attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)  # normalize
#         rollout = attn_with_residual if rollout is None else torch.bmm(rollout, attn_with_residual)
#     return rollout  # [B, N, N]


# def visualize_attention(attn_map, save_path='attention_map.png', title='Attention Map'):
#     plt.figure(figsize=(6, 5))
#     plt.imshow(attn_map, cmap='viridis')
#     plt.xlabel("Key Token")
#     plt.ylabel("Query Token")
#     plt.title(title)
#     plt.colorbar()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()