import torch
import numpy as np
import os

from challenge_files import *
from challenge_files import XMLHandler
import torch.nn.functional as F
import os
import pwd

def get_username():
    return pwd.getpwuid(os.getuid())[0]

if get_username()=='zm8bh':
    from more_itertools import pairwise
else:
    from itertools import pairwise

def logit(array, alpha=1.e-6, inv=False):
    if inv:
        z = torch.sigmoid(array)
        z = (z-alpha)/(1-2*alpha)
    else:
        z = array*(1-2*alpha) + alpha
        z = torch.logit(z)
    return z

class Standardize(object):
    """
    Standardize features 
        mean: vector of means 
        std: vector of stds
    """
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = shower*self.stds + self.means
        else:
            transformed = (shower - self.means)/self.stds
        return transformed, energy

# class StandardizeFromFile(object):
#     """
#     Standardize features 
#         mean_path: path to `.npy` file containing means of the features 
#         std_path: path to `.npy` file containing standard deviations of the features
#         create: whether or not to calculate and save mean/std based on first call
#     """

#     def __init__(self, model_dir):

#         self.model_dir = model_dir
#         #print('In StandardizeFromFile: ',model_dir)
#         self.mean_path = os.path.join(model_dir, 'means.npy')
#         self.std_path = os.path.join(model_dir, 'stds.npy')
#         self.dtype = torch.get_default_dtype()
#         try:
#             # load from file
#             self.mean = torch.from_numpy(np.load(self.mean_path)).to(self.dtype)
#             self.std = torch.from_numpy(np.load(self.std_path)).to(self.dtype)
#             self.written = True
#         except FileNotFoundError:
#             self.written = False

#     def write(self, shower, energy):
#         self.mean = shower.mean(axis=0)
#         self.std = shower.std(axis=0)
#         np.save(self.mean_path, self.mean.detach().cpu().numpy())
#         np.save(self.std_path, self.std.detach().cpu().numpy())
#         self.written = True

#     def __call__(self, shower, energy, rev=False):
#         if rev:
#             print("shape of std and mean: ",self.std.shape, self.mean.shape)
#             transformed = shower*self.std.to(shower.device) + self.mean.to(shower.device)
#         else:
#             if not self.written:
#                 self.write(shower, energy)
#             transformed = (shower - self.mean.to(shower.device))/self.std.to(shower.device)
#         if shower is not None and energy is not None:
#             print("standard: ", transformed.shape, transformed.min(), transformed.max(), energy.shape,energy.min(), energy.max())
#         return transformed, energy
#### Author's code    
# class StandardizeFromFile(object):
#     """
#     Standardize features 
#         mean_path: path to `.npy` file containing means of the features 
#         std_path: path to `.npy` file containing standard deviations of the features
#         create: whether or not to calculate and save mean/std based on first call
#     """

#     def __init__(self, model_dir):

#         self.model_dir = model_dir
#         self.mean_path = os.path.join(model_dir, 'means.npy')
#         self.std_path = os.path.join(model_dir, 'stds.npy')
#         self.dtype = torch.get_default_dtype()
#         try:
#             # load from file
#             self.mean = torch.from_numpy(np.load(self.mean_path)).to(self.dtype)
#             self.std = torch.from_numpy(np.load(self.std_path)).to(self.dtype)
#             self.written = True
#         except FileNotFoundError:
#             self.written = False

#     def write(self, shower, energy):
#         self.mean = shower.mean(axis=0)
#         self.std = shower.std(axis=0)
#         np.save(self.mean_path, self.mean.detach().cpu().numpy())
#         np.save(self.std_path, self.std.detach().cpu().numpy())
#         self.written = True
        
#     def __call__(self, shower, energy, rev=False):
#         if rev:
#             # Only unstandardize the last 45 dimensions
#             shower[:, -45:] = shower[:, -45:] * self.std.to(shower.device) + self.mean.to(shower.device)
#             transformed = shower
#         else:
#             if not self.written:
#                 # Only compute stats on the last 45 dimensions
#                 self.mean = shower[:, -45:].mean(axis=0)
#                 self.std = shower[:, -45:].std(axis=0)
#                 np.save(self.mean_path, self.mean.detach().cpu().numpy())
#                 np.save(self.std_path, self.std.detach().cpu().numpy())
#                 self.written = True
#             # Only standardize the last 45 dimensions
#             shower[:, -45:] = (shower[:, -45:] - self.mean.to(shower.device)) / self.std.to(shower.device)
#             transformed = shower
#         return transformed, energy
    # def __call__(self, shower, energy, rev=False):
    #     if rev:
    #         transformed = shower*self.std.to(shower.device) + self.mean.to(shower.device)
    #     else:
    #         if not self.written:
    #             self.write(shower, energy)
    #         transformed = (shower - self.mean.to(shower.device))/self.std.to(shower.device)
    #     return transformed, energy
###Farzana's code
# class StandardizeFromFile(object):
#     """
#     Standardize 45-D features with saved mean/std.
#     - Works with inputs shaped (B,45) or (B,45,1).
#     - Always stores mean/std with shape (45,).
#     """

#     def __init__(self, model_dir):
#         self.model_dir = model_dir or "./stats"
#         os.makedirs(self.model_dir, exist_ok=True)
#         self.mean_path = os.path.join(self.model_dir, "means.npy")
#         self.std_path  = os.path.join(self.model_dir, "stds.npy")
#         self.dtype = torch.get_default_dtype()

#         try:
#             self.mean = torch.from_numpy(np.load(self.mean_path)).to(self.dtype)
#             self.std  = torch.from_numpy(np.load(self.std_path)).to(self.dtype)
#             self.written = True
#         except FileNotFoundError:
#             self.mean = None
#             self.std  = None
#             self.written = False

#     def _as_45(self, shower: torch.Tensor) -> torch.Tensor:
#         """Return a (B,45) view for statistics computation."""
#         if shower.ndim == 3 and shower.shape[-1] == 1:  # (B,45,1)
#             return shower.squeeze(-1)                   # -> (B,45)
#         if shower.ndim == 2 and shower.shape[1] == 45:  # (B,45)
#             return shower
#         raise ValueError(f"Unexpected input shape {tuple(shower.shape)}; expected (B,45) or (B,45,1).")

#     def _broadcast_stats(self, shower: torch.Tensor):
#         """Return mean/std tensors broadcastable to shower."""
#         if self.mean is None or self.std is None:
#             raise RuntimeError("Mean/std not initialized.")
#         mean = self.mean.to(shower.device)
#         std  = self.std.to(shower.device).clamp_min(1e-8)
#         if shower.ndim == 3 and shower.shape[-1] == 1:   # (B,45,1)
#             return mean.view(1, -1, 1), std.view(1, -1, 1)
#         if shower.ndim == 2 and shower.shape[1] == 45:   # (B,45)
#             return mean.view(1, -1), std.view(1, -1)
#         # Fallback to strict error
#         raise ValueError(f"Unexpected input shape {tuple(shower.shape)}; expected (B,45) or (B,45,1).")

#     def write(self, shower, energy):
#         """Compute stats from the current batch (shape-robust) and save as (45,)."""
#         X = self._as_45(shower).detach()
#         self.mean = X.mean(dim=0).to(self.dtype)                   # (45,)
#         self.std  = X.std(dim=0, unbiased=True).to(self.dtype)     # (45,)
#         np.save(self.mean_path, self.mean.cpu().numpy())
#         np.save(self.std_path,  self.std.cpu().numpy())
#         self.written = True

#     def __call__(self, shower, energy, rev: bool = False):
#         # If there is no feature tensor (edge case), just pass through
#         if shower is None:
#             return shower, energy

#         if rev:
#             mean, std = self._broadcast_stats(shower)
#             transformed = shower * std + mean
#         else:
#             if not self.written:
#                 # Compute & save stats from the FIRST (train) batch
#                 self.write(shower, energy)
#             mean, std = self._broadcast_stats(shower)
#             transformed = (shower - mean) / std

#         return transformed, energy


class StandardizeFromFile(object):
    """
    Standardize last 45 features using mean/std loaded from .npy files.

    Updates:
    - avoids in-place slice assignment (graph-safe)
    - device/dtype-safe mean/std each call
    - epsilon in denominator for stability
    """

    def __init__(self, model_dir, n_features=45, eps=1e-12):
        self.model_dir = model_dir
        self.mean_path = os.path.join(model_dir, "means.npy")
        self.std_path = os.path.join(model_dir, "stds.npy")
        self.dtype = torch.get_default_dtype()
        self.n_features = int(n_features)
        self.eps = float(eps)

        try:
            self.mean = torch.from_numpy(np.load(self.mean_path)).to(self.dtype)
            self.std = torch.from_numpy(np.load(self.std_path)).to(self.dtype)
            self.written = True
        except FileNotFoundError:
            self.mean = None
            self.std = None
            self.written = False

    def write(self, shower, energy):
        # If you ever call this, ensure it computes over the last n_features
        tail = shower[:, -self.n_features:]
        mean = tail.mean(dim=0)
        std = tail.std(dim=0)

        np.save(self.mean_path, mean.detach().cpu().numpy())
        np.save(self.std_path, std.detach().cpu().numpy())

        self.mean = mean.detach().cpu().to(self.dtype)
        self.std = std.detach().cpu().to(self.dtype)
        self.written = True

    def __call__(self, shower, energy, rev=False):
        # Compute stats once if missing (non-rev path)
        if (not rev) and (not self.written):
            tail = shower[:, -self.n_features:]
            mean = tail.mean(dim=0)
            std = tail.std(dim=0)

            np.save(self.mean_path, mean.detach().cpu().numpy())
            np.save(self.std_path, std.detach().cpu().numpy())

            # store CPU tensors (like your original behavior)
            self.mean = mean.detach().cpu().to(self.dtype)
            self.std = std.detach().cpu().to(self.dtype)
            self.written = True

        # If still not written, something is wrong
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardizeFromFile: mean/std not initialized and not written.")

        # Move mean/std to the same device/dtype as shower (local tensors)
        mean = self.mean.to(device=shower.device, dtype=shower.dtype)
        std = self.std.to(device=shower.device, dtype=shower.dtype)

        head = shower[:, :-self.n_features]
        tail = shower[:, -self.n_features:]

        if rev:
            # Unstandardize: tail * std + mean
            tail_out = tail * std + mean
        else:
            # Standardize: (tail - mean) / (std + eps)
            tail_out = (tail - mean) / (std + self.eps)

        transformed = torch.cat([head, tail_out], dim=1)
        return transformed, energy

class SelectDims(object):
    """
    Selects a subset of the features 
        start: start of range of indices to keep
        end:   end of range of indices to keep (exclusive)
    """

    def __init__(self, start, end):
        #print("inside selectDims")
        self.indices = torch.arange(start, end)
    def __call__(self, shower, energy, rev=False):
        # if shower is not None and energy is not None:
        #     #print("Inside SelectDims: ",shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max() )
        if rev:
            return shower, energy
        transformed = shower[..., self.indices]
        return transformed, energy
#         if energy is not None and transformed is not None:
            
#             print("transformed: ",transformed.shape, energy.shape)
#             print("inside selectDims, transformed and energy: ",transformed.shape, transformed.min(), transformed.max(), energy.shape,energy.min(), energy.max())
        

class AddFeaturesToCond(object):
    """
    Transfers a subset of the input features to the condition
        split_index: Index at which to split input. Features past the index will be moved
    """

    def __init__(self, split_index):
        self.split_index = split_index
    
    def __call__(self, x, c, rev=False):
        
        if rev:
            c_, split = c[:, :1], c[:, 1:]
            x_ = torch.cat([x, split], dim=1)
        else:
            x_, split = x[:, :self.split_index], x[:, self.split_index:]
            c_ = torch.cat([c, split], dim=1)
        return x_, c_
    
class LogEnergy(object):
    """
    Log transform incident energies
        alpha: Optional regularization for the log
    """            
    def __init__(self, alpha=0.):
        self.alpha = alpha
        self.cond_transform = True
        
    def __call__(self, shower, energy, rev=False):
        # if shower is not None and energy is not None:
        #     print("Inside LogEnergy: ",shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max() )
        
        if rev:
            transformed = torch.exp(energy) - self.alpha
        else:
            #print("alpha: ",self.alpha, energy.shape)
            transformed = torch.log(energy + self.alpha)
            # if shower is not None and transformed is not None:
            #     print("Inside LogEnergy: ",shower.shape, shower.min(), shower.max(), transformed.shape, transformed.min(), transformed.max())
        #print("Log energy shower and transformed: ",shower.shape,transformed.shape)
        return shower, transformed              

class ScaleVoxels(object):
    """
    Apply a multiplicative factor to the voxels.
        factor: Number to multiply voxels
    """
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, shower, energy, rev=False):
        #print("inside Scale voxels: ",shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max())
        if rev:
            transformed = shower/self.factor
        else:
            transformed = shower*self.factor
        return transformed, energy

class ScaleTotalEnergy(object):
    """
    Scale only E_tot/E_inc by a factor f.
    The effect is the same of ScaleVoxels but
    it is applied in a different position in the
    preprocessing chain.
    """
    def __init__(self, factor, n_layers=45):
        self.factor = factor
        self.n_layers = n_layers

    def __call__(self, shower, energy, rev=False):
        
        if rev:
            shower[..., -self.n_layers] /= self.factor
        else:
            shower[..., -self.n_layers] *= self.factor
        # if shower is not None and energy is not None:
        #     print("inside Scale Total Energy: ",shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max())
        return shower, energy

class ScaleEnergy(object):
    """
    Scale incident energies to lie in the range [0, 1]
        e_min: Expected minimum value of the energy
        e_max: Expected maximum value of the energy
    """
    def __init__(self, e_min, e_max):
        self.e_min = e_min
        self.e_max = e_max
        self.cond_transform = True

    def __call__(self, shower, energy, rev=False):
        # if shower is not None and energy is not None:
        #     print("Inside ScaleEnergy: ",shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max() )
        if rev:
            transformed = energy * (self.e_max - self.e_min)
            transformed += self.e_min
        else:
            transformed = energy - self.e_min
            transformed /= (self.e_max - self.e_min)
        # if shower is not None and transformed is not None:
        #     print("Inside ScaleEnergy: ",shower.shape, shower.min(), shower.max(), transformed.shape, transformed.min(), transformed.max())
        return shower, transformed

class LogTransform(object):
    """
    Take log of input data
        alpha: regularization
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = torch.exp(shower) - self.alpha
        else:
            transformed = torch.log(shower + self.alpha)
        # if shower is not None and transformed is not None:
        #     print("Inside LogTransform: ",energy.shape, transformed.shape)
        return transformed, energy

class SelectiveLogTransform(object):
    """
    Take log of input data
        alpha: regularization
        exclusions: list of indices for features that should not be transformed
    """
    def __init__(self, alpha, exclusions=None):
        self.alpha = alpha
        self.exclusions = exclusions

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = torch.exp(shower) - self.alpha
        else:
            transformed = torch.log(shower + self.alpha)
        if self.exclusions is not None:
            transformed[..., self.exclusions] = shower[..., self.exclusions]
        return transformed, energy

class ExclusiveLogTransform(object):
    """
    Take log of input data
        delta: regularization
        exclusions: list of indices for features that should not be transformed
    """

    def __init__(self, delta, exclusions=None):
        self.delta = delta
        self.exclusions = exclusions

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = torch.exp(shower) - self.delta
        else:
            transformed = torch.log(shower + self.delta)
        if self.exclusions is not None:
            transformed[..., self.exclusions] = shower[..., self.exclusions] 
        return transformed, energy
 
class ExclusiveLogitTransform(object):
    """
    Take log of input data
        delta: regularization
        exclusions: list of indices for features that should not be transformed
    """

    def __init__(self, delta, exclusions=None, rescale=False):
        self.delta = delta
        self.exclusions = exclusions
        self.rescale = rescale

    def __call__(self, shower, energy, rev=False):
        # if shower is not None and energy is not None:
        #     print("inside exclusive, shower and energy: ",shower.shape, shower.min(), shower.max(), energy.shape,energy.min(), energy.max())
        if rev:
            if self.rescale:
                transformed = logit(shower, alpha=self.delta, inv=True)
            else:
                transformed = torch.special.expit(shower)
        else:
            if self.rescale:
                transformed = logit(shower, alpha=self.delta)
            else:
                transformed = torch.logit(shower, eps=self.delta)

        if self.exclusions is not None:
            transformed[..., self.exclusions] = shower[..., self.exclusions]
#         if energy is not None and transformed is not None:
           
#             print("inside exclusive, transformed and energy: ",transformed.shape, transformed.min(), transformed.max(), energy.shape,energy.min(), energy.max())
        return transformed, energy
    

class AddNoise(object):
    """
    Add noise to input data
        func: torch distribution used to sample from
        width_noise: noise rescaling
    """
    def __init__(self, noise_width, cut=False):
        #self.func = func
        self.func = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0))
        self.noise_width = noise_width
        self.cut = cut # apply cut if True

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.noise_width)
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        else:
            noise = self.func.sample(shower.shape)*self.noise_width
            transformed = shower + noise.reshape(shower.shape).to(shower.device)
        return transformed, energy


class SmoothUPeaks(object):
    """
    Smooth voxels equal to 0 or 1 using uniform noise
        w0: noise width for zeros
        w1: noise width for ones
        eps: threshold below which values are considered zero
    """

    def __init__(self, w0, w1, eps=1.e-10):
        self.func = torch.distributions.Uniform(
            torch.tensor(0.0), torch.tensor(1.0))
        self.w0 = w0
        self.w1 = w1
        self.scale = 1 + w0 + w1
        self.eps = eps

    def __call__(self, u, energy, rev=False):
        if rev:
            # undo scaling
            transformed = u*self.scale - self.w0
            # clip to [0, 1]
            transformed = torch.clip(transformed, min=0., max=1.)
            # restore u0
            transformed[:, 0] = u[:, 0]
        else:
            # sample noise values
            n0 = self.w0*self.func.sample(u.shape).to(u.device)
            n1 = self.w1*self.func.sample(u.shape).to(u.device)
            # add noise to us
            transformed = u - n0*(u<=self.eps) + n1*(u>=1-self.eps)
            # scale to [0,1] in preparation for logit
            transformed = (transformed + self.w0)/self.scale
            # restore u0
            transformed[:, 0] = u[:, 0]
            
        return transformed, energy

# class SelectiveUniformNoise(object):
#     """
#     Add noise to input data with the option to exlude some features
#         func: torch distribution used to sample from
#         width_noise: noise rescaling
#         exclusions: list of indices for features that should not be transformed
#     """
#     def __init__(self, noise_width, exclusions = None, cut=False):
#         #self.func = func
#         self.func = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0))
#         self.noise_width = noise_width
#         self.exclusions = exclusions
#         self.cut = cut # apply cut if True

#     def __call__(self, shower, energy, rev=False):
#         if rev:
#             mask = (shower < self.noise_width)
#             if self.exclusions:
#                 mask[:, self.exclusions] = False
#             transformed = shower
#             if self.cut:
#                 transformed[mask] = 0.0 
#         else:
#             noise = self.func.sample(shower.shape)*self.noise_width
#             if self.exclusions:
#                 noise[:, self.exclusions] = 0.0
#             transformed = shower + noise.reshape(shower.shape).to(shower.device)
#         return transformed, energy        


class SelectiveUniformNoise(object):
    """
    Add uniform noise to input data with the option to exclude some features.

    - forward (rev=False): adds Uniform(0, noise_width) noise to non-excluded features
    - reverse (rev=True): optionally applies a *soft* cut to values below noise_width
                         (instead of hard masking) to keep gradients alive and avoid
                         in-place assignment.
    """
    def __init__(self, noise_width, exclusions=None, cut=False, temperature=0.01):
        """
        Args:
            noise_width: scalar threshold/scale
            exclusions: list of feature indices that should not be changed
            cut: if True, apply cut in rev=True branch
            temperature: softness for the cut gate (smaller = sharper)
        """
        self.noise_width = float(noise_width)
        self.exclusions = exclusions
        self.cut = bool(cut)
        self.temperature = float(temperature)

    def __call__(self, shower, energy, rev=False):
        if rev:
            # Reverse: previously did hard mask -> set to 0.
            # Now: do a differentiable smooth gate (only if cut=True).
            if not self.cut or self.noise_width <= 0.0:
                return shower, energy

            # Build a gate that is ~0 when shower < noise_width and ~1 otherwise
            tau = max(self.temperature, 1e-12)
            gate = torch.sigmoid((shower - self.noise_width) / tau)  # same shape as shower

            # Exclusions: force gate=1 so those features are untouched
            if self.exclusions:
                # Create a slice gate for excluded features without in-place ops
                gate_excl = torch.ones_like(gate[:, self.exclusions])
                gate = gate.clone()  # safe since this is small compared to shower
                gate[:, self.exclusions] = gate_excl

            transformed = shower * gate
            return transformed, energy

        else:
            # Forward: add noise to non-excluded features (graph-safe; noise has no grad)
            if self.noise_width <= 0.0:
                return shower, energy

            # Sample noise on the same device/dtype as shower
            noise = torch.rand_like(shower) * self.noise_width  # Uniform(0, noise_width)

            if self.exclusions:
                # Zero out excluded noise (avoid in-place on original noise if you prefer)
                noise = noise.clone()
                noise[:, self.exclusions] = 0.0

            transformed = shower + noise
            return transformed, energy

class SetToVal(object):
    """
    Masks voxels to zero in the reverse transformation
        cut: threshold value for the mask
    """
    def __init__(self, val=0.):
        self.val = val

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.val)
            transformed = shower
            if self.val:
                transformed[mask] = self.val
        else:
            mask = (shower < self.val)
            transformed = shower
            if self.val:
                transformed[mask] = self.val
        return transformed, energy        

# class CutValues(object):
#     """
#     Cut in Normalized space
#         cut: threshold value for the cut
#         n_layers: number of layers to avoid cutting on the us
#     """
#     def __init__(self, cut=0., n_layers=45):
#         self.cut = cut
#         self.n_layers = n_layers

#     def __call__(self, shower, energy, rev=False):
#         if rev:
#             mask = (shower <= self.cut)
#             mask[:, -self.n_layers:] = False
#             transformed = shower
#             if self.cut:
#                 transformed[mask] = 0.0 
#         else:
#             transformed = shower
#         return transformed, energy        
#### Newer Version of CutValues to keep it differentiable cut

class CutValues(object):
    """
    Differentiable cut that mimics: if x <= cut -> 0, else keep x
    Uses a smooth sigmoid gate; protects last n_layers features.
    """
    def __init__(self, cut=0.0, n_layers=45, temperature=0.01):
        self.cut = float(cut)
        self.n_layers = int(n_layers)
        self.temperature = float(temperature)

    def __call__(self, shower, energy, rev=False):
        if not rev or self.cut == 0.0:
            return shower, energy

        voxels = shower[:, :-self.n_layers] if self.n_layers > 0 else shower
        u_features = shower[:, -self.n_layers:] if self.n_layers > 0 else None

        tau = max(self.temperature, 1e-12)
        gate = torch.sigmoid((voxels - self.cut) / tau)
        voxels_cut = voxels * gate  # smooth suppression toward 0

        transformed = (
            torch.cat([voxels_cut, u_features], dim=1)
            if u_features is not None else voxels_cut
        )
        return transformed, energy

class CutBothValues(object):
    def __init__(self, cut=0.):
        self.cut = cut

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower <= self.cut)
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        else:
            mask = (shower <= self.cut)
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        return transformed, energy        

class ZeroMask(object):
    """
    Masks voxels to zero in the reverse transformation
        cut: threshold value for the mask
    """
    def __init__(self, cut=0.):
        self.cut = cut

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.cut)
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        else:
            transformed = shower
        return transformed, energy        

class AddPowerlawNoise(object):
    """
    Add noise to input data following a power law distribution:
        eps ~ k x^(k-1)
        k   -- The power parameter of the distribution
        cut -- The value below which voxels will be masked to zero in the reverse transformation
    """
    def __init__(self, k, cut=None):
        self.k = k
        self.cut = cut

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.cut)
            transformed = shower
            if self.cut is not None:
                transformed[mask] = 0.0
        else:
            noise = torch.from_numpy(np.random.power(self.k, shower.shape)).to(shower.dtype)
            transformed = shower + noise.reshape(shower.shape).to(shower.device)
        return transformed, energy

class NormalizeByEinc(object):
    """
    Normalize each shower by the incident energy

    """
    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = shower*energy
        else:
            transformed = shower/energy
        return transformed, energy

class Reshape(object):
    """
    Reshape the shower as specified. Flattens batch in the reverse transformation.
        shape -- Tuple representing the desired shape of a single example
    """

    def __init__(self, shape):
        self.shape = torch.Size(shape)

    def __call__(self, shower, energy, rev=False):
        # if shower is not None and energy is not None:
        #     print("Before reshaping: shower and energy: ",shower.shape, shower.min(), shower.max(),energy.shape,energy.min(),energy.max())
        if rev:
            shower = shower.reshape(-1, self.shape.numel())
        else:
            print("Inside the reshape function and shape of shower: ", shower.shape)
            shower = shower.reshape(-1, *self.shape)
#         if shower is not None and energy is not None:
            
#             print("after reshaping: shower and energy: ",shower.shape, shower.min(), shower.max(),energy.shape,energy.min(),energy.max())
        return shower, energy
 
class Reweight(object):
    """
    Reweight voxels
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, shower, energy, rev=False):
        if rev:
            shower = shower**(1/self.factor)
        else:
            shower = shower**(self.factor)
        return shower, energy

# class NormalizeByElayer(object):
#     """
#     Normalize each shower by the layer energy
#     This will change the shower shape to N_voxels+N_layers
#        layer_boundaries: ''
#        eps: numerical epsilon
#     """
#     def __init__(self, ptype, xml_file, cut=0.0, eps=1.e-10, return_layer_energies=False, return_summed_energies=False):
#         self.eps = eps
#         #print("what is transforms: ",xml_file)
#         self.xml = XMLHandler.XMLHandler(xml_file, ptype)
#         self.layer_boundaries = np.unique(self.xml.GetBinEdges())
#         self.n_layers = len(self.layer_boundaries) - 1
#         self.cut = cut
#         self.return_layer_energies = return_layer_energies  # NEW PARAMETER
#         self.return_summed_energies = return_summed_energies

#     def __call__(self, shower, energy, rev=False):
#         # print("Inside normalizeByELayer: Before", shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max())
#         if rev:

#             # select u features
#             us = shower[:, -self.n_layers:]
            
#             # clip u_{i>0} into [0,1] Commented due to gradient issue in loss calculation
#             # us[:, (-self.n_layers+1):] = torch.clip(
#             #     us[:, (-self.n_layers+1):],
#             #     min=torch.tensor(0., device=shower.device),
#             #     max=torch.tensor(1., device=shower.device)
#             # ) 
#             clipped_u = torch.clip(
#             us[:, (-self.n_layers+1):],
#                 min=0.0,
#                 max=1.0
#             )
#             us = torch.cat([us[:, :(-self.n_layers+1)], clipped_u], dim=1)
#             # calculate unnormalised energies from the u's
#             layer_Es = []
#             total_E = torch.multiply(energy.flatten(), us[:,0]) # Einc * u_0
#             cum_sum = torch.zeros_like(total_E)
#             for i in range(us.shape[-1]-1):
#                 layer_E = (total_E - cum_sum) * us[:,i+1]
#                 layer_Es.append(layer_E)
#                 cum_sum += layer_E
#             layer_Es.append(total_E - cum_sum)
#             layer_Es = torch.vstack(layer_Es).T
#             #next two lines are done by Farzana
#             print("shape of layer_Es: ", layer_Es.shape)
#             #transformed=layer_Es  #needed only for energy model sample generation
#             # select voxels
#             # ------ commented by Farzana------ to get the layer energy only.
#             # shower = shower[:, :-self.n_layers]

#             # # Normalize each layer and multiply it with its original energy
#             # transformed = torch.zeros_like(shower)
#             # for l, (start, end) in enumerate(pairwise(self.layer_boundaries)):
#             #     layer = shower[:, start:end] # select layer
#             #     layer /= layer.sum(-1, keepdims=True) + self.eps # normalize to unity
#             #     mask = (layer <= self.cut)
#             #     layer[mask] = 0.0 # apply normalized cut
#             #     transformed[:, start:end] = layer * layer_Es[:,[l]] # scale to layer energy

#             # ========== NEW: OPTION TO RETURN LAYER ENERGIES ONLY ==========
#             if self.return_layer_energies:
#                 # For energy loss computation - return layer energies directly
#                 transformed = layer_Es  # (B, 45)
#                 print("Returning from self.return_layer_energies!")
#                 return transformed, energy 
#             elif self.return_summed_energies:
#                 # Reconstruct physical voxels from predicted distribution
#                 voxels = shower[:, :-self.n_layers]  # (B, 6480) - predicted voxel distribution
                
#                 # Unnormalize and scale by layer energies
#                 physical_voxels = torch.zeros_like(voxels)
#                 summed_layer_Es = []

#                 for l, (start, end) in enumerate(pairwise(self.layer_boundaries)):
#                     layer = voxels[:, start:end]  # Predicted voxel distribution for this layer
#                     layer = layer / (layer.sum(-1, keepdims=True) + self.eps)  # Normalize to unity
#                     layer = layer.clone()
#                     mask = (layer <= self.cut)
#                     layer[mask] = 0.0
#                     physical_layer = layer * layer_Es[:,[l]]  # Scale to physical energy
#                     physical_voxels[:, start:end] = physical_layer
                    
#                     # Sum this layer's voxels to get predicted layer energy
#                     summed_layer_E = torch.sum(physical_layer, dim=1)
#                     summed_layer_Es.append(summed_layer_E)
                
#                 summed_layer_Es = torch.stack(summed_layer_Es, dim=1)  # (B, 45)
#                 print(f"[NormalizeByElayer] Returning summed_layer_Es: {summed_layer_Es.shape}")
#                 return summed_layer_Es, energy
                
#             else:
#                 # For full reconstruction - reconstruct voxel structure
#                 shower = shower[:, :-self.n_layers]
                
#                 # Normalize each layer and multiply it with its original energy
#                 transformed = torch.zeros_like(shower)
#                 for l, (start, end) in enumerate(pairwise(self.layer_boundaries)):
#                     layer = shower[:, start:end] # select layer
#                     layer /= layer.sum(-1, keepdims=True) + self.eps # normalize to unity
#                     mask = (layer <= self.cut)
#                     layer[mask] = 0.0 # apply normalized cut
#                     transformed[:, start:end] = layer * layer_Es[:,[l]] # scale to layer energy

#         else:
#             # compute layer energies
#             layer_Es = []
#             for start, end in pairwise(self.layer_boundaries):
#                 layer_E = torch.sum(shower[:, start:end], dim=1, keepdims=True)
#                 shower[:, start:end] /= layer_E + self.eps # normalize to unity
#                 layer_Es.append(layer_E) # store layer energy
#             layer_Es = torch.cat(layer_Es, dim=1).to(shower.device)
#             print("in in ",layer_Es.shape)

#             # compute generalized extra dimensions
#             extra_dims = [torch.sum(layer_Es, dim=1, keepdim=True) / energy]
#             for l in range(layer_Es.shape[1]-1):
#                 remaining_E = torch.sum(layer_Es[:, l:], dim=1, keepdim=True)
#                 extra_dim = layer_Es[:, [l]] / (remaining_E+ self.eps)
#                 extra_dims.append(extra_dim)
#             extra_dims = torch.cat(extra_dims, dim=1)
#             #print("extra_dims concatenated: ",extra_dims.shape)
#             transformed = torch.cat((shower, extra_dims), dim=1)
# #         if energy is not None and transformed is not None:
            
#         print("inside Normalizeby Elayer Transform and energy: ",transformed.shape, transformed.min(),transformed.max(), energy.shape,energy.min(), energy.max())
#         return transformed, energy


# class NormalizeByElayer(object):
#     def __init__(
#         self,
#         ptype,
#         xml_file,
#         cut=0.0,
#         eps=1.0e-10,
#         return_layer_energies=False,
#         return_summed_energies=False,
#         soft_clip_temp=0.1,
#         soft_threshold_temp=0.01,
#     ):
#         self.eps = float(eps)
#         self.cut = float(cut)
#         self.return_layer_energies = bool(return_layer_energies)
#         self.return_summed_energies = bool(return_summed_energies)

#         self.soft_clip_temp = float(soft_clip_temp)
#         self.soft_threshold_temp = float(soft_threshold_temp)

#         self.xml = XMLHandler.XMLHandler(xml_file, ptype)
#         self.layer_boundaries = np.unique(self.xml.GetBinEdges())
#         self.n_layers = len(self.layer_boundaries) - 1

#     def _sigmoid_squash_01(self, x: torch.Tensor, temp: float) -> torch.Tensor:
#         """
#         Differentiable mapping to (0,1).
#         Smaller temp -> sharper; too small can saturate gradients.
#         """
#         t = max(float(temp), 1.0e-12)
#         return torch.sigmoid(x / t)

#     def _soft_gate_cut_to_zero(self, x: torch.Tensor, cut: float, temp: float) -> torch.Tensor:
#         """
#         Differentiable approximation of:
#             if x <= cut: 0 else: x
#         Implemented as:
#             x * sigmoid((x - cut)/temp)
#         """
#         t = max(float(temp), 1.0e-12)
#         gate = torch.sigmoid((x - cut) / t)
#         return x * gate

#     def __call__(self, shower, energy, rev=False):
#         if rev:
#             # 1) Split u-features from the end of shower
#             us = shower[:, -self.n_layers:]  # (B, n_layers)
#             voxels = shower[:, :-self.n_layers]  # (B, n_vox)

#             # 2) Softly constrain u1..uL to (0,1), keep u0 unchanged
#             # Your old code: clipped_u = soft_clip(us[:,1:], 0,1); us = cat([u0, clipped_u])
#             u0 = us[:, :1]
#             u_rest_raw = us[:, 1:]
#             u_rest = self._sigmoid_squash_01(u_rest_raw, self.soft_clip_temp)
#             us = torch.cat([u0, u_rest], dim=1)

#             # 3) Compute per-layer energies from u features (graph-safe torch ops)
#             layer_Es = []
#             total_E = energy.flatten() * us[:, 0]  # (B,)
#             cum_sum = torch.zeros_like(total_E)

#             for i in range(us.shape[1] - 1):
#                 layer_E = (total_E - cum_sum) * us[:, i + 1]
#                 layer_Es.append(layer_E)
#                 cum_sum = cum_sum + layer_E

#             layer_Es.append(total_E - cum_sum)
#             layer_Es = torch.stack(layer_Es, dim=1)  # (B, L)

#             if self.return_layer_energies:
#                 return layer_Es, energy

#             # Helper: build reconstructed layers
#             if self.return_summed_energies:
#                 summed_layer_Es = []
#                 physical_voxels = torch.zeros_like(voxels)

#                 for l, (start, end) in enumerate(pairwise(self.layer_boundaries)):
#                     layer = voxels[:, start:end]

#                     # Normalize to fractions summing to 1 (your code did this)
#                     layer = layer / (layer.sum(dim=1, keepdim=True) + self.eps)

#                     # Soft "cut to zero" on fractions, then renormalize to preserve layer energy
#                     if self.cut > 0.0:
#                         layer = self._soft_gate_cut_to_zero(layer, self.cut, self.soft_threshold_temp)
#                         layer = layer / (layer.sum(dim=1, keepdim=True) + self.eps)

#                     physical_layer = layer * layer_Es[:, [l]]
#                     physical_voxels[:, start:end] = physical_layer

#                     # This now matches layer_Es[:, l] up to numerical eps
#                     summed_layer_Es.append(physical_layer.sum(dim=1))

#                 summed_layer_Es = torch.stack(summed_layer_Es, dim=1)  # (B, L)
#                 return summed_layer_Es, energy

#             else:
#                 transformed = torch.zeros_like(voxels)

#                 for l, (start, end) in enumerate(pairwise(self.layer_boundaries)):
#                     layer = voxels[:, start:end]
#                     layer = layer / (layer.sum(dim=1, keepdim=True) + self.eps)

#                     if self.cut > 0.0:
#                         layer = self._soft_gate_cut_to_zero(layer, self.cut, self.soft_threshold_temp)
#                         layer = layer / (layer.sum(dim=1, keepdim=True) + self.eps)

#                     transformed[:, start:end] = layer * layer_Es[:, [l]]

#                 return transformed, energy

#         else:
#             # Forward pass: compute normalized per-layer fractions and append u-features.
#             # Key change: avoid in-place writes to `shower`.

#             layer_Es = []
#             normalized_chunks = []

#             for start, end in pairwise(self.layer_boundaries):
#                 layer = shower[:, start:end]
#                 layer_E = layer.sum(dim=1, keepdim=True)  # (B,1)
#                 layer_Es.append(layer_E)

#                 # Out-of-place normalization (replaces in-place /=)
#                 normalized_layer = layer / (layer_E + self.eps)
#                 normalized_chunks.append(normalized_layer)

#             shower_norm = torch.cat(normalized_chunks, dim=1)  # (B, n_vox)
#             layer_Es = torch.cat(layer_Es, dim=1).to(shower.device)  # (B, L)

#             # Compute u-features (same logic, out-of-place)
#             extra_dims = [layer_Es.sum(dim=1, keepdim=True) / (energy + self.eps)]  # u0

#             for l in range(layer_Es.shape[1] - 1):
#                 remaining_E = layer_Es[:, l:].sum(dim=1, keepdim=True)
#                 extra_dim = layer_Es[:, [l]] / (remaining_E + self.eps)
#                 extra_dims.append(extra_dim)

#             extra_dims = torch.cat(extra_dims, dim=1)  # (B, L)
#             transformed = torch.cat((shower_norm, extra_dims), dim=1)
#             return transformed, energy

class NormalizeByElayer(object):
    def __init__(
        self,
        ptype,
        xml_file,
        cut=0.0,
        eps=1.0e-10,
        return_layer_energies=False,
        return_summed_energies=False,
        soft_clip_temp=0.1,
        soft_threshold_temp=0.01,
        # NEW:
        layer_energy_source="u",       # "u" (old) or "voxels" (new)
        layer_weight_temp=1.0,         # temperature for voxel->weight mapping
    ):
        self.eps = float(eps)
        self.cut = float(cut)
        self.return_layer_energies = bool(return_layer_energies)
        self.return_summed_energies = bool(return_summed_energies)

        self.soft_clip_temp = float(soft_clip_temp)
        self.soft_threshold_temp = float(soft_threshold_temp)

        self.layer_energy_source = str(layer_energy_source)
        self.layer_weight_temp = float(layer_weight_temp)

        self.xml = XMLHandler.XMLHandler(xml_file, ptype)
        self.layer_boundaries = np.unique(self.xml.GetBinEdges())
        self.n_layers = len(self.layer_boundaries) - 1

    def _sigmoid_squash_01(self, x: torch.Tensor, temp: float) -> torch.Tensor:
        t = max(float(temp), 1.0e-12)
        return torch.sigmoid(x / t)

    def _soft_gate_cut_to_zero(self, x: torch.Tensor, cut: float, temp: float) -> torch.Tensor:
        t = max(float(temp), 1.0e-12)
        gate = torch.sigmoid((x - cut) / t)
        return x * gate

    def __call__(self, shower, energy, rev=False):
        if rev:
            # shower: (B, n_vox + L) due to AddFeaturesToCond(rev=True)
            us = shower[:, -self.n_layers:]       # (B, L)
            voxels = shower[:, :-self.n_layers]   # (B, n_vox)

            # total incident energy in MeV (after ScaleEnergy+LogEnergy rev)
            total_E = energy.flatten()            # (B,)

            # --------- choose layer energies ----------
            if self.layer_energy_source == "u":
                # Old behavior: layer energies come from u
                u0 = us[:, :1]
                u_rest_raw = us[:, 1:]
                u_rest = self._sigmoid_squash_01(u_rest_raw, self.soft_clip_temp)
                us_used = torch.cat([u0, u_rest], dim=1)

                layer_Es = []
                total_E_eff = total_E * us_used[:, 0]  # (B,)
                cum_sum = torch.zeros_like(total_E_eff)

                for i in range(us_used.shape[1] - 1):
                    layer_E = (total_E_eff - cum_sum) * us_used[:, i + 1]
                    layer_Es.append(layer_E)
                    cum_sum = cum_sum + layer_E

                layer_Es.append(total_E_eff - cum_sum)
                layer_Es = torch.stack(layer_Es, dim=1)  # (B, L)

            elif self.layer_energy_source == "voxels":
                # NEW behavior: layer energies come from the predicted voxels themselves
                layer_raw = []
                for (start, end) in pairwise(self.layer_boundaries):
                    v = voxels[:, start:end]  # (B, n_bin_layer)
                    # raw per-layer mass (can be noisy/negative depending on your space)
                    s = v.sum(dim=1)          # (B,)
                    layer_raw.append(s)
                layer_raw = torch.stack(layer_raw, dim=1)  # (B, L)

                # make positive & smooth
                temp = max(self.layer_weight_temp, 1.0e-12)
                w = F.softplus(layer_raw / temp)  # (B, L), positive
                w = w / (w.sum(dim=1, keepdim=True) + self.eps)  # sum to 1

                layer_Es = total_E.unsqueeze(1) * w  # (B, L)

            else:
                raise ValueError(f"Unknown layer_energy_source={self.layer_energy_source}")

            if self.return_layer_energies:
                return layer_Es, energy

            # --------- reconstruct physical voxels ----------
            if self.return_summed_energies:
                summed_layer_Es = []
                physical_voxels = torch.zeros_like(voxels)

                for l, (start, end) in enumerate(pairwise(self.layer_boundaries)):
                    layer = voxels[:, start:end]

                    # normalize within-layer to fractions
                    layer = layer / (layer.sum(dim=1, keepdim=True) + self.eps)

                    if self.cut > 0.0:
                        layer = self._soft_gate_cut_to_zero(layer, self.cut, self.soft_threshold_temp)
                        layer = layer / (layer.sum(dim=1, keepdim=True) + self.eps)

                    physical_layer = layer * layer_Es[:, [l]]
                    physical_voxels[:, start:end] = physical_layer
                    summed_layer_Es.append(physical_layer.sum(dim=1))

                summed_layer_Es = torch.stack(summed_layer_Es, dim=1)  # (B, L)
                return summed_layer_Es, energy

            else:
                transformed = torch.zeros_like(voxels)
                for l, (start, end) in enumerate(pairwise(self.layer_boundaries)):
                    layer = voxels[:, start:end]
                    layer = layer / (layer.sum(dim=1, keepdim=True) + self.eps)

                    if self.cut > 0.0:
                        layer = self._soft_gate_cut_to_zero(layer, self.cut, self.soft_threshold_temp)
                        layer = layer / (layer.sum(dim=1, keepdim=True) + self.eps)

                    transformed[:, start:end] = layer * layer_Es[:, [l]]

                return transformed, energy
        else:
            # Forward pass: compute normalized per-layer fractions and append u-features.
            # Key change: avoid in-place writes to `shower`.

            layer_Es = []
            normalized_chunks = []

            for start, end in pairwise(self.layer_boundaries):
                layer = shower[:, start:end]
                layer_E = layer.sum(dim=1, keepdim=True)  # (B,1)
                layer_Es.append(layer_E)

                # Out-of-place normalization (replaces in-place /=)
                normalized_layer = layer / (layer_E + self.eps)
                normalized_chunks.append(normalized_layer)

            shower_norm = torch.cat(normalized_chunks, dim=1)  # (B, n_vox)
            layer_Es = torch.cat(layer_Es, dim=1).to(shower.device)  # (B, L)
            if energy.ndim > 2:
                energy = energy.squeeze(1)
                print("shape of energy: ",energy.shape)
            print("in preprocess steps: ",shower_norm.shape)
            print("shape of layer_Es: ",layer_Es.shape)
            # Compute u-features (same logic, out-of-place)
            extra_dims = [layer_Es.sum(dim=1, keepdim=True) / (energy + self.eps)]  # u0
            print("shape of energy: ",energy.shape,extra_dims[0].shape)

            for l in range(layer_Es.shape[1] - 1):
                remaining_E = layer_Es[:, l:].sum(dim=1, keepdim=True)
                extra_dim = layer_Es[:, [l]] / (remaining_E + self.eps)
                extra_dims.append(extra_dim)
            print("in preprocess steps shape of extra_dims before concat: ",len(extra_dims))
            extra_dims = torch.cat(extra_dims, dim=1)  # (B, L)
            transformed = torch.cat((shower_norm, extra_dims), dim=1)
            return transformed, energy

        

class AddCoordChannels(object):
    """
    Add channel to image containing the coordinate value along particular
    dimension. This breaks the translation symmetry of the convoluitons,
    as discussed in arXiv:2308.03876

        dims -- List of dimensions for which should have a coordinate channel
                should be created.
    """

    def __init__(self, dims):
        self.dims = dims

    def __call__(self, shower, energy, rev=False):

        if rev:
            transformed = shower # generated shower already only has 1 channel
        else:
            coords = []
            for d in self.dims:
                bcst_shp = [1] * shower.ndim
                bcst_shp[d] = -1
                size = shower.size(d)
                coords.append(torch.ones_like(shower) / size *
                              torch.arange(size).view(bcst_shp))
            transformed = torch.cat([shower] + coords, dim=1)
        return transformed, energy
