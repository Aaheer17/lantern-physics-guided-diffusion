# standard python libraries
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages
from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import apply_gradient_vector, get_gradient_vector
from conflictfree.length_model import TrackSpecific
from conflictfree.grad_operator import PCGradOperator, IMTLGOperator
import sys
import h5py
# Other functions of project
from Util.util import *
from datasets import *
from documenter import Documenter
from plotting_util import *
from transforms import *
from challenge_files import *
from challenge_files import evaluate # avoid NameError: 'evaluate' is not defined
from challenge_files.compute_metrics_helper import calculate_correlation_voxel

import Models
from Models import *
import time
from itertools import islice

import pandas as pd

from Models.multi_objective import create_mo_optimizer

class GenerativeModel(nn.Module):
    """
    Base Class for Generative Models to inherit from.
    Children classes should overwrite the individual methods as needed.
    Every child class MUST overwrite the methods:

    def build_net(self): should register some NN architecture as self.net
    def batch_loss(self, x): takes a batch of samples as input and returns the loss
    def sample_n_parallel(self, n_samples): generates and returns n_samples new samples

    See tbd.py for an example of child class

    Structure:

    __init__(params)      : Read in parameters and register the important ones
    build_net()           : Create the NN and register it as self.net
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    prepare_training()    : Read in the appropriate parameters and prepare the model for training
                            Currently this is called from run_training(), so it should not be called on its own
    run_training()        : Run the actual training.
                            Necessary parameters are read in and the training is performed.
                            This calls on the methods train_one_epoch() and validate_one_epoch()
    train_one_epoch()     : Performs one epoch of model training.
                            This calls on the method batch_loss(x)
    validate_one_epoch()  : Performs one epoch of validation.
                            This calls on the method batch_loss(x)
    batch_loss(x)         : Takes one batch of samples as input and returns the loss.
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_n(n_samples)   : Generates and returns n_samples new samples as a numpy array
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_and_plot       : Generates n_samples and makes plots with them.
                            This is meant to be used during training if intermediate plots are wanted

    """
    def __init__(self, params, device, doc):
        """
        :param params: file with all relevant model parameters
        """
        super().__init__()
        self.doc = doc
        self.params = params
        #print("inside the modelbase: ",self.params)
        self.device = device
        self.shape = self.params['shape']#get(self.params,'shape')
        self.conditional = get(self.params,'conditional',False)
        self.single_energy = get(self.params, 'single_energy', None) # Train on a single energy
        self.eval_mode = get(self.params, 'eval_mode', 'all')

        self.batch_size = self.params["batch_size"]
        self.batch_size_sample = get(self.params, "batch_size_sample", 10_000)
        print("in modelbase")
        self.net = self.build_net()
        param_count = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f'init model: Model has {param_count} parameters')
        self.params['parameter_count'] = param_count

        self.epoch = get(self.params, "total_epochs", 0)
        self.iterations = get(self.params,"iterations", 1)
        self.regular_loss = []
        self.kl_loss = []
        self.global_step = 0
        self.runs = get(self.params, "runs", 0)
        self.iterate_periodically = get(self.params, "iterate_periodically", False)
        self.validate_every = get(self.params, "validate_every", 50)

        # augment data
        self.aug_transforms = get(self.params, "augment_batch", False)

        # load autoencoder for latent modelling
        self.ae_dir = get(self.params, "autoencoder", None)
        if self.ae_dir is None:
            self.transforms = get_transformations(
                params.get('transforms', None), doc=self.doc
            )
            self.latent = False
        else:
            self.ae = self.load_other(self.ae_dir)# model_class='AE'
            self.transforms = self.ae.transforms
            self.latent = True


        # ========================================
        # Voxel-wise CFD validation config
        # ========================================
        self.validate_voxel_CFD = get(self.params, "validate_voxel_CFD", False)
        self.validate_voxel_CFD_every = get(self.params, "validate_voxel_CFD_every", 0)

        # ── ADD ONLY THESE TWO LINES ──────────────────────────────────────────────
        self.use_laplacian_loss = get(self.params, 'use_laplacian_loss', False)
        self.lambda_laplacian   = get(self.params, 'lambda_laplacian', 0.01)

        self.val_voxel_cfd_epoch = np.array([], dtype=np.float64)
        # Only read CFD-specific paths if enabled
        if self.validate_voxel_CFD:
            
            self.val_CFD_hdf5_file = get(
                self.params,
                "val_CFD_hdf5_file",
                None
            )
            self.voxel_CFD_ref_file = get(
                self.params,
                "voxel_CFD_ref",
                None
            )
        
            if self.val_CFD_hdf5_file is None or self.voxel_CFD_ref_file is None:
                raise ValueError(
                    "validate_voxel_CFD=True but CFD paths not fully provided."
                )

            print(
                f"[Voxel CFD] Enabled | every {self.validate_voxel_CFD_every} epochs\n"
                f"[Voxel CFD] val_CFD_hdf5_file = {self.val_CFD_hdf5_file}\n"
                f"[Voxel CFD] voxel_CFD_ref    = {self.voxel_CFD_ref_file}"
            )

            # ---- load voxel CFD reference ONCE ----
            with h5py.File(self.voxel_CFD_ref_file, "r") as f:
                self.C_ref_voxel = f["C_ref_voxel"][:]          # (44,16,9)
                fro = f["fro_norm_ref"][:]
        
            # fro_norm_ref may be stored as scalar or [1]
            self.fro_norm_ref = float(fro[0] if fro.ndim > 0 else fro)
            # build and cache the conditional loader once
            self._build_voxel_cfd_cond_loader()
        else:
            # Keep attributes defined to avoid hasattr checks later
            
            self.val_CFD_hdf5_file = None
            self.voxel_CFD_ref_file = None
            self.C_ref_voxel = None
            self.fro_norm_ref = None
        
       
        # Multi-objective optimization setup
        self.mo_method = get(self.params, 'mo_method', 'weighted_sum')
        self.use_energy_loss = get(self.params, 'use_energy_loss', False)
        self.use_moment_matching = get(self.params, 'use_moment_matching', False)
        self.use_sparsity = get(self.params, 'use_sparsity', False)
        self.use_voxel_energy_loss=get(self.params, 'use_voxel_energy_loss', False)
        # Weights for weighted_sum method
        self.lambda_energy = get(self.params, 'lambda_energy', 0.1)
        self.lambda_moment = get(self.params, 'lambda_moment', 0.0)
        self.lambda_sparsity = get(self.params, 'lambda_sparsity', 1e-4)
        self.lambda_voxel_energy=get(self.params, 'lambda_voxel_energy',1e-4)
        # Initialize MO optimizer (after self.net and self.optimizer are created)
        self.mo_optimizer = None  # Will be initialized in build_model()

        self.use_voxel_shape_loss = get(self.params, 'use_voxel_shape_loss', False)
        self.lambda_voxel_shape  = get(self.params, 'lambda_voxel_shape', 1e-4)
        # ── GradNorm ──────────────────────────────────────────────────────────
        # mo_method: 'gradnorm'
        # Learns w_i(t) for each objective by minimising a gradient norm loss.
        # phi_t modulation for laplacian is applied inside batch_loss before
        # loss_tensors is returned — GradNorm sees phi_t * L_lap as one task.
        self.gradnorm_alpha    = get(self.params, 'gradnorm_alpha', 1.5)
        self.gradnorm_lr       = get(self.params, 'gradnorm_lr',   1e-3)
        self.gradnorm_balancer = None  # initialised in prepare_training()
        # ── Gradient Blending (professor's method) ──────────────────────────
        # mo_method: 'grad_blend'
        # g_new = β ( ĝ1 + α·ĝ2 )  where β preserves ||g_old|| = ||g1 + g2||
        # alpha = 1.0  →  bisector (equal weight, same direction as ConFIG)
        # alpha < 1.0  →  bias toward diffusion (primary objective)
        # alpha = 0.0  →  pure diffusion
        self.blend_alpha = get(self.params, 'blend_alpha', 1.0)

    def build_net(self):
        pass

    def prepare_training(self):
        
        print("train_model: Preparing model training")

        self.train_loader, self.val_loader, self.bounds = get_loaders(
            self.params.get('hdf5_file'),
            self.params.get('particle_type'),
            self.params.get('xml_filename'),
            self.params.get('val_frac'),
            self.params.get('batch_size'),
            self.transforms,
            self.params.get('eps', 1.e-10),
            device=self.device,
            shuffle=True,
            width_noise=self.params.get('width_noise', 1.e-6),
            single_energy=self.params.get('single_energy', None),
            aug_transforms=self.aug_transforms
        )

        self.use_scheduler = get(self.params, "use_scheduler", False)

        self.n_trainbatches = len(self.train_loader)
        self.n_traindata = self.n_trainbatches*self.batch_size
        self.set_optimizer(steps_per_epoch=self.n_trainbatches)
        # ========================================
        # Multi-Objective Optimizer
        # ========================================
        has_aux_losses = (
            self.use_energy_loss or 
            self.use_moment_matching or 
            self.use_sparsity or 
            self.use_voxel_energy_loss or
            self.use_voxel_shape_loss or
            self.use_laplacian_loss
        )
    
        if not has_aux_losses:
            # No auxiliary losses - skip MO optimizer entirely
            print("✓ No auxiliary losses enabled - using simple diffusion loss only")
            self.mo_optimizer = None
            self.mo_method = 'none'  # Mark as no MO
        elif self.mo_method not in ('config', 'grad_blend', 'pcgrad', 'imtlg', 'gradnorm'):
            mo_kwargs = {}
            
            if self.mo_method == 'weighted_sum':
                mo_kwargs['weights'] = {
                    'diffusion_loss': 1.0,
                    'energy_loss': self.lambda_energy,
                    'moment_loss': self.lambda_moment,
                    'sparsity_loss': self.lambda_sparsity,
                    'voxel_energy_loss': self.lambda_voxel_energy,
                    'voxel_shape_loss':  self.lambda_voxel_shape,
                    'laplacian_loss': self.lambda_laplacian,

                }
            elif self.mo_method == 'dwa':
                mo_kwargs['temperature'] = get(self.params, 'dwa_temperature', 2.0)
            elif self.mo_method == 'grad_norm':
                mo_kwargs['target_loss'] = 'diffusion_loss'
            
            self.mo_optimizer = create_mo_optimizer(
                self.mo_method,
                self.net,
                self.optimizer,
                **mo_kwargs
            )
            print(f"✓ Multi-objective optimizer: {self.mo_method}")
        else:
            if self.mo_method == 'config':
                print(f"✓ Multi-objective optimizer: ConFIG (gradient-based)")
            elif self.mo_method == 'grad_blend':
                print(f"✓ Multi-objective optimizer: Gradient Blending "
                      f"(professor's method, alpha={self.blend_alpha})")
            elif self.mo_method == 'pcgrad':
                print(f"✓ Multi-objective optimizer: PCGrad (gradient surgery)")
            elif self.mo_method == 'imtlg':
                print(f"✓ Multi-objective optimizer: IMTL-G (impartial multi-task learning)")
            elif self.mo_method == 'gradnorm':
                from Models.gradnorm import GradNormBalancer   # see Change 4
                objectives = [k for k in [
                    'diffusion_loss', 'voxel_energy_loss', 'laplacian_loss',
                    'energy_loss', 'moment_loss', 'sparsity_loss'
                ] if get(self.params, f'use_{k.replace("_loss","")}', k == 'diffusion_loss')]
            
                self.gradnorm_balancer = GradNormBalancer(
                    num_tasks  = len(objectives),
                    alpha      = self.gradnorm_alpha,
                    lr         = self.gradnorm_lr,
                    device     = self.device,
                )
                self.gradnorm_objectives = objectives 
                print(f"✓ Multi-objective optimizer: GradNorm "
                      f"(alpha={self.gradnorm_alpha}, tasks={objectives})")
        

        
        if hasattr(self, 'setup_ema'):
            print("Setting up EMA model...")
            self.setup_ema()

        self.sample_periodically = get(self.params, "sample_periodically", False)
        if self.sample_periodically:
            self.sample_every = get(self.params, "sample_every", 1)
            self.sample_every_n_samples = get(self.params, "sample_every_n_samples", 100000)
            print(f'train_model: sample_periodically set to True. Sampling {self.sample_every_n_samples} every'
                  f' {self.sample_every} epochs. This may significantly slow down training!')

        self.log = get(self.params, "log", True)
        if self.log:
            log_dir = self.doc.basedir
            self.logger = SummaryWriter(log_dir)
            print(f"train_model: Logging to log_dir {log_dir}")
        else:
            print("train_model: log set to False. No logs will be written")

    def set_optimizer(self, steps_per_epoch=1, params=None):
        """ Initialize optimizer and learning rate scheduling """
        if params is None:
            params = self.params

        self.optimizer = torch.optim.AdamW(
                self.net.parameters(),
                lr = params.get("lr", 0.0002),
                betas = params.get("betas", [0.9, 0.999]),
                eps = params.get("optimizer_eps", 1e-6),
                weight_decay = params.get("weight_decay", 0.)
                )
        self.scheduler = set_scheduler(self.optimizer, params, steps_per_epoch, last_epoch=-1)
    def _build_voxel_cfd_cond_loader(self):
        """
        Build a fixed conditional DataLoader for voxel-CFD evaluation.
    
        Uses teacher-forced (truth) u's from the CFD validation HDF5 file.
        Stores the DataLoader as self.voxel_cfd_cond.
        """
        batch_size_sample = get(self.params, "batch_size_sample", 100)
    
        # This dataset must match the same transforms used in training
        ds = CaloChallengeDataset(
            self.val_CFD_hdf5_file,                 # <-- IMPORTANT: your 2k CFD file
            self.params.get('particle_type'),
            self.params.get('xml_filename'),
            transform=self.transforms,
            device=self.device,
            single_energy=self.single_energy
        )
    
        # In your existing pipeline, ds.energy is what you used as "truth u's"
        transformed_cond = ds.energy  # expected to be (N,46) for shape model on DS2
    
        # Cache a fixed loader (no shuffle)
        self.voxel_cfd_cond = DataLoader(
            dataset=transformed_cond,
            batch_size=batch_size_sample,
            shuffle=False
        )
    
        # Optional sanity prints (safe)
        try:
            n = len(transformed_cond)
            cdim = transformed_cond.shape[1] if hasattr(transformed_cond, "shape") and transformed_cond.dim() == 2 else None
            print(f"[Voxel CFD] Condition tensor: N={n}, cond_dim={cdim}, batch_size_sample={batch_size_sample}")
            print(f"[Voxel CFD] Using CFD HDF5: {self.val_CFD_hdf5_file}")
        except Exception:
            print(f"[Voxel CFD] Built condition loader from: {self.val_CFD_hdf5_file}")
    def _compute_voxel_cfd_scalar(self, x_gen_all, cond_all):
        """
        Compute voxel-wise CFD scalar from generated samples.
    
        Args:
            x_gen_all: torch.Tensor of generated samples in model space, shape (N, 6480) or (N, *self.shape)
            cond_all:  torch.Tensor of conditions used for generation, shape (N, 46)
    
        Uses:
            self.transforms (reverse loop)
            self.C_ref_voxel  (np.ndarray, shape (44,16,9))
            self.fro_norm_ref (float)
            calculate_correlation_voxel(data_np) -> (44,16,9)
    
        Returns:
            voxel_cfd (float)
        """
        
    
        if not isinstance(x_gen_all, torch.Tensor):
            raise TypeError(f"x_gen_all must be a torch.Tensor, got {type(x_gen_all)}")
        if not isinstance(cond_all, torch.Tensor):
            raise TypeError(f"cond_all must be a torch.Tensor, got {type(cond_all)}")
    
        x = x_gen_all.to(self.device)
        c = cond_all.to(self.device)
        if not torch.is_floating_point(c):
            c = c.float()
    
        # Reverse transforms to physical space
        for fn in self.transforms[::-1]:
            x, c = fn(x, c, rev=True)
    
        # Expect physical showers as (N, 6480)
        if x.dim() != 2 or x.shape[1] != 45 * 16 * 9:
            raise ValueError(f"Expected physical showers shape (N,6480), got {tuple(x.shape)}")
    
        vox = x.detach().cpu().numpy().reshape(-1, 45, 16, 9)  # (N,45,16,9)
    
        # Compute generated voxel correlation
        C_gen = calculate_correlation_voxel(vox)  # (44,16,9)
    
        # Normalized Frobenius distance
        diff = C_gen - self.C_ref_voxel
        fro_diff = float(np.sqrt(np.sum(diff * diff)))
        voxel_cfd = fro_diff / (float(self.fro_norm_ref) + 1e-12)
    
        return voxel_cfd

    def run_training(self):

        self.prepare_training()
        
        
        samples = []
        n_epochs = get(self.params, "n_epochs", 100)
        
        past_epochs = get(self.params, "total_epochs", 0)
        if past_epochs != 0:
            self.load(epoch=past_epochs)
            self.scheduler = set_scheduler(self.optimizer, self.params, self.n_trainbatches, last_epoch=self.params.get("total_epochs", -1)*self.n_trainbatches)
        print(f"train_model: Model has been trained for {past_epochs} epochs before.")
        print(f"train_model: Beginning training. n_epochs set to {n_epochs}")
        t_0 = time.time()
        for e in range(n_epochs):
            t0 = time.time()

            self.epoch = past_epochs + e
            #print("Before net.train")
            self.net.train()
            #print("Before train one epoch")
            one_s=time.time()
            self.train_one_epoch()
            print("One epoch time: ",time.time()-one_s)
            if (self.epoch + 1) % self.validate_every == 0:
                self.eval()
                self.validate_one_epoch()

            if self.sample_periodically:
                if (self.epoch + 1) % self.sample_every == 0:
                    self.eval()

                    # # if true then i * bayesian samples will be drawn, else just 1
                    # iterations = self.iterations if self.iterate_periodically else 1
                    # bay_samples = []
                    # for i in range(0, iterations):
                    #     sample, c = self.sample_n()
                    #     bay_samples.append(sample)
                    # samples = np.concatenate(bay_samples)
                    if get(self.params, "reconstruct", False):
                        samples, c = self.reconstruct_n()
                    else:
                        samples, c = self.sample_n()
                    self.plot_samples(samples=samples, conditions=c, name=self.epoch, energy=self.single_energy)

                    
            voxel_cfd_value = np.nan  # default

            if self.validate_voxel_CFD and self.validate_voxel_CFD_every > 0:
                if (self.epoch + 1) % self.validate_voxel_CFD_every == 0:
                    self.eval()
                    self.net.eval()
            
                    ddim_steps = get(self.params, "voxel_CFD_ddim_steps", 400)
                    eta = get(self.params, "voxel_CFD_ddim_eta", 0.0)
            
                    gen_chunks = []
                    cond_chunks = []
            
                    with torch.inference_mode():
                        for cond_batch in self.voxel_cfd_cond:
                            if not isinstance(cond_batch, torch.Tensor):
                                raise TypeError(f"Expected Tensor from voxel_cfd_cond loader, got {type(cond_batch)}")
            
                            cond_batch = cond_batch.to(self.device)
                            if not torch.is_floating_point(cond_batch):
                                cond_batch = cond_batch.float()
            
                            x_gen = self.ddim_sample(cond_batch, eta=eta, ddim_steps=ddim_steps)
            
                            gen_chunks.append(x_gen.detach().cpu())
                            cond_chunks.append(cond_batch.detach().cpu())
            
                    x_gen_all = torch.cat(gen_chunks, dim=0)
                    cond_all  = torch.cat(cond_chunks, dim=0)
            
                    voxel_cfd_value = self._compute_voxel_cfd_scalar(
                        x_gen_all=x_gen_all,
                        cond_all=cond_all
                    )
            
                    print(f"[Voxel CFD] Epoch {self.epoch+1}: voxel_CFD = {voxel_cfd_value:.6f}")
            
            # Append once per epoch (but only safe if val_voxel_cfd_epoch is always an array)
            self.val_voxel_cfd_epoch = np.append(self.val_voxel_cfd_epoch, voxel_cfd_value)

            
            # save model periodically, useful when trying to understand how weights are learned over iterations
            if get(self.params,"save_periodically",False):
                if (self.epoch + 1) % get(self.params,"save_every",10) == 0 or self.epoch==0:
                    self.save(epoch=f"{self.epoch}")
            print(f"Epoch {self.epoch}: LR = {self.scheduler.get_last_lr()[0]}")
            # estimate training time
            if e==0:
                t1 = time.time()
                dtEst= (t1-t0) * n_epochs
                print(f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h")
            sys.stdout.flush()
        t_1 = time.time()
        traintime = t_1 - t_0
        self.params['train_time'] = traintime
        print(
            f"train_model: Finished training {n_epochs} epochs after {traintime:.2f} s = {traintime / 60:.2f} min = {traintime / 60 ** 2:.2f} h.", flush=True)
        
        #save final model
        print("train_model: Saving diagnostics", flush=True)
        self.save_all_metrics()
        

        self.save()
        # generate and plot samples at the end
        if get(self.params, "sample", True):
            #print("generate_samples: Start generating samples", flush=True)
            if get(self.params, "reconstruct", False):
                samples, c = self.reconstruct_n()
            else:
                samples, c = self.sample_n()
            self.plot_samples(samples=samples, conditions=c, energy=self.single_energy)
        #np.save(self.doc.get_file('train_losses_epoch.npy'), self.train_losses_epoch)
    def combine_losses(self, loss_tensors):
        """
        Combine losses using the selected multi-objective method
        
        Args:
            loss_tensors: dict of individual loss tensors with gradients
        
        Returns:
            total_loss: combined loss for backward()
            mo_info: dict with MO-specific info for logging
        """
        # ========================================
        # CASE 1: No auxiliary losses - just return diffusion loss
        # ========================================
        if self.mo_method == 'none' or self.mo_optimizer is None:
            return loss_tensors['diffusion_loss'], {}
        
        # ========================================
        # CASE 2: ConFIG or grad_blend
        # both handle the gradient step separately in train_one_epoch
        # ========================================
        if self.mo_method in ('config', 'grad_blend', 'pcgrad', 'imtlg'):
            return loss_tensors['diffusion_loss'], {}
        
        # ========================================
        # CASE 3: Multi-objective methods
        # ========================================
        # Build use flags
        use_flags = {
            'diffusion_loss': True,  # Always use
            'energy_loss': self.use_energy_loss,
            'moment_loss': self.use_moment_matching,
            'sparsity_loss': self.use_sparsity,
            'voxel_energy_loss':self.use_voxel_energy_loss,
            'voxel_shape_loss': self.use_voxel_shape_loss,
            'laplacian_loss':    self.use_laplacian_loss, 
        }
        
        # Use the MO optimizer
        total_loss, mo_info = self.mo_optimizer.combine(loss_tensors, use_flags)
        
        return total_loss, mo_info
    def _get_config_objectives(self, loss_tensors: dict) -> list[str]:
        """
        Decide which objectives participate in ConFIG.
        Priority:
          1) params['config_objectives'] if provided
          2) sensible default: diffusion_loss + any aux/residual losses present
        """
        cfg = getattr(self, "params", {}) or {}
        requested = cfg.get("config_objectives", None)
    
        if requested is not None:
            # keep only those present
            return [k for k in requested if k in loss_tensors]
    
        # default behavior
        default_order = [
            "diffusion_loss",
            "residual_loss",
            "energy_loss",
            "moment_loss",
            "sparsity_loss",
            'voxel_energy_loss',
            'voxel_shape_loss',
            "laplacian_loss",
        ]
        return [k for k in default_order if k in loss_tensors]
    # ============================================================
    # GRADIENT STEP HELPERS  —  drop these into GenerativeModel
    # ============================================================
    #
    # Structure
    # ---------
    # _compute_per_objective_grads   shared gradient extraction (called by every _step_with_*)
    # _compute_gradient_diagnostics  correct norms / cosines / angles in degrees
    # _step_with_config              ConFIG bisector
    # _step_with_grad_blend          professor's blending method
    # _step_with_pcgrad              gradient surgery
    # _step_with_imtlg               impartial multi-task learning
    #
    # Every _step_with_* method follows the same four-step pattern:
    #   1. Extract per-objective gradients          (_compute_per_objective_grads)
    #   2. Fallback to diffusion if NaN/Inf found
    #   3. Compute the method-specific combined gradient
    #   4. Compute diagnostics                      (_compute_gradient_diagnostics)
    # ============================================================
    
    def _compute_per_objective_grads(self, loss_tensors: dict, objectives: list):
        """
        Backpropagate each objective separately and collect flat gradient vectors.
    
        Returns
        -------
        grads    : list[Tensor]  one flat gradient vector per objective, same order
        grad_ok  : bool          False if any gradient contains NaN or Inf
        """
        grads   = []
        grad_ok = True
    
        for i, name in enumerate(objectives):
            self.optimizer.zero_grad(set_to_none=True)
            retain = (i < len(objectives) - 1)
            loss_tensors[name].backward(retain_graph=retain)
            g = get_gradient_vector(self.net)
            grads.append(g)
    
            if torch.isnan(g).any() or torch.isinf(g).any():
                grad_ok = False
    
        return grads, grad_ok
    
    
    def _compute_gradient_diagnostics(self, grads: list, objectives: list,
                                       g_combined=None) -> dict:
        """
        Compute gradient diagnostics with correct geometry.
    
        All angles are in degrees. All cosines are clipped to [-1, 1].
        Norm ratios use objectives[0] (always diffusion_loss) as the denominator.
    
        Parameters
        ----------
        grads      : list[Tensor]   per-objective flat gradient vectors
        objectives : list[str]      objective names, same order as grads
        g_combined : Tensor | None  combined gradient (ConFIG / blend / pcgrad / imtlg)
    
        CSV columns produced
        --------------------
        gnorm_{name}                          per-objective gradient norm
        gnorm_ratio_{name}_over_diff          ||g_aux|| / ||g_diff||   (key imbalance diagnostic)
        cos_{obj_i}__{obj_j}                  pairwise cosine similarity between raw gradients
        angle_deg_{obj_i}__{obj_j}            pairwise angle in degrees between raw gradients
        gnorm_combined                        norm of the combined gradient
        cos_combined__{name}                  cosine between g_combined and each raw gradient
        angle_deg_combined__{name}            angle in degrees between g_combined and each raw gradient
        """
        info  = {}
        norms = [float(torch.linalg.vector_norm(g).item()) for g in grads]
    
        # ── per-objective norms ──────────────────────────────────────────────────
        for name, n in zip(objectives, norms):
            info[f"gnorm_{name}"] = n
    
        # ── norm ratio: aux / diffusion ──────────────────────────────────────────
        norm_diff = norms[0] + 1e-10
        for name, n in zip(objectives[1:], norms[1:]):
            info[f"gnorm_ratio_{name}_over_diff"] = n / norm_diff
    
        # ── pairwise angles between raw gradients ────────────────────────────────
        for i in range(len(objectives)):
            for j in range(i + 1, len(objectives)):
                denom  = (norms[i] * norms[j]) + 1e-10
                cos_ij = float(np.clip(
                    float(torch.dot(grads[i], grads[j]).item()) / denom,
                    -1.0, 1.0
                ))
                info[f"cos_{objectives[i]}__{objectives[j]}"]       = cos_ij
                info[f"angle_deg_{objectives[i]}__{objectives[j]}"] = float(np.degrees(np.arccos(cos_ij)))
    
        # ── angles between combined gradient and each raw gradient ───────────────
        if g_combined is not None:
            norm_combined = float(torch.linalg.vector_norm(g_combined).item())
            info["gnorm_combined"] = norm_combined
            for name, g, n in zip(objectives, grads, norms):
                cos_ci = float(np.clip(
                    float(torch.dot(g_combined, g).item()) / ((norm_combined + 1e-10) * (n + 1e-10)),
                    -1.0, 1.0
                ))
                info[f"cos_combined__{name}"]       = cos_ci
                info[f"angle_deg_combined__{name}"] = float(np.degrees(np.arccos(cos_ci)))
    
        return info
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ConFIG
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _step_with_config(self, loss_tensors: dict) -> dict:
        """
        ConFIG update: combined gradient is the unit bisector of all objective
        gradients, scaled by the length model (TrackSpecific or unit norm).
    
        For two objectives the combined direction is always proportional to
        û_diff + û_aux, so each objective sees an angle of θ/2.
        Apparent 0° readings in diagnostic plots are numerical artifacts from
        near-zero gradient norms, not real alignment.
        """
        info       = {}
        objectives = self._get_config_objectives(loss_tensors)
    
        assert objectives[0] == "diffusion_loss", (
            f"ConFIG requires diffusion_loss at index 0, got: {objectives}"
        )
        if len(objectives) < 2:
            return {"config_used": 0.0}
    
        # ── 1. per-objective gradients ───────────────────────────────────────────
        grads, grad_ok = self._compute_per_objective_grads(loss_tensors, objectives)
    
        # ── 2. fallback ──────────────────────────────────────────────────────────
        if not grad_ok:
            apply_gradient_vector(self.net, grads[0])
            return {"config_used": 0.0}
    
        # ── 3. ConFIG combined gradient ──────────────────────────────────────────
        g_config = ConFIG_update(grads, use_least_square=False)
        apply_gradient_vector(self.net, g_config)
    
        # ── 4. diagnostics ───────────────────────────────────────────────────────
        info = self._compute_gradient_diagnostics(grads, objectives, g_combined=g_config)
        info["config_used"]          = 1.0
        info["config_num_objectives"] = float(len(objectives))
    
        return info
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Gradient Blending (professor's method)
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _step_with_grad_blend(self, loss_tensors: dict) -> dict:
        """
        Gradient blending:
    
            g_new = β ( û_diff + α · mean(û_aux_i) )
    
        where
            û_i   = g_i / ||g_i||          unit gradient
            α     = self.blend_alpha        diffusion bias
                      1.0 → bisector (same direction as ConFIG)
                      <1.0 → bias toward diffusion
                      0.0 → pure diffusion
            β     = ||g_naive|| / ||direction||
                    preserves the magnitude of the naive sum g_diff + g_aux
    
        No solver needed. β is computed analytically.
        """
        info       = {}
        objectives = self._get_config_objectives(loss_tensors)
    
        assert objectives[0] == "diffusion_loss", (
            f"grad_blend requires diffusion_loss at index 0, got: {objectives}"
        )
        if len(objectives) < 2:
            return {"blend_used": 0.0}
    
        # ── 1. per-objective gradients ───────────────────────────────────────────
        grads, grad_ok = self._compute_per_objective_grads(loss_tensors, objectives)
    
        # ── 2. fallback ──────────────────────────────────────────────────────────
        if not grad_ok:
            apply_gradient_vector(self.net, grads[0])
            return {"blend_used": 0.0}
    
        # ── 3. blended combined gradient ─────────────────────────────────────────
        alpha = self.blend_alpha
        norms = [float(torch.linalg.vector_norm(g).item()) for g in grads]
        units = [g / (n + 1e-10) for g, n in zip(grads, norms)]
    
        # direction: û_diff + α * mean(û_aux)
        g_direction = units[0].clone()
        aux_weight  = alpha / max(len(objectives) - 1, 1)
        for u in units[1:]:
            g_direction = g_direction + aux_weight * u
    
        # magnitude: preserve ||naive sum||
        g_naive    = sum(grads[1:], grads[0].clone())
        g_naive_norm = float(torch.linalg.vector_norm(g_naive).item())
        dir_norm   = float(torch.linalg.vector_norm(g_direction).item())
        beta       = g_naive_norm / (dir_norm + 1e-10)
    
        g_blend = beta * g_direction
        apply_gradient_vector(self.net, g_blend)
    
        # ── 4. diagnostics ───────────────────────────────────────────────────────
        info = self._compute_gradient_diagnostics(grads, objectives, g_combined=g_blend)
        info["blend_used"]          = 1.0
        info["blend_alpha"]         = alpha
        info["blend_beta"]          = beta
        info["config_num_objectives"] = float(len(objectives))
    
        return info
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PCGrad
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _step_with_pcgrad(self, loss_tensors: dict) -> dict:
        """
        PCGrad — Gradient Surgery (Yu et al., NeurIPS 2020).
    
        For each gradient g_i, remove the projection onto g_j when they conflict:
            if dot(g_i, g_j) < 0:
                g_i = g_i - ( dot(g_i, g_j) / ||g_j||^2 ) * g_j
    
        The projected gradients are then summed.
    
        When gradients are near-orthogonal (cos ≈ 0), PCGrad reduces to
        a plain sum — no projection occurs. When they conflict (cos < 0),
        only the conflicting component is removed.
        """
        info       = {}
        objectives = self._get_config_objectives(loss_tensors)
    
        assert objectives[0] == "diffusion_loss", (
            f"pcgrad requires diffusion_loss at index 0, got: {objectives}"
        )
        if len(objectives) < 2:
            return {"pcgrad_used": 0.0}
    
        # ── 1. per-objective gradients ───────────────────────────────────────────
        grads, grad_ok = self._compute_per_objective_grads(loss_tensors, objectives)
    
        # ── 2. fallback ──────────────────────────────────────────────────────────
        if not grad_ok:
            apply_gradient_vector(self.net, grads[0])
            return {"pcgrad_used": 0.0}
    
        # ── 3. PCGrad combined gradient ──────────────────────────────────────────
        g_pc = PCGradOperator().calculate_gradient(grads)
        apply_gradient_vector(self.net, g_pc)
    
        # ── 4. diagnostics ───────────────────────────────────────────────────────
        info = self._compute_gradient_diagnostics(grads, objectives, g_combined=g_pc)
    
        # count conflicting pairs (unordered)
        n_conflicts = sum(
            1 for i in range(len(grads)) for j in range(i + 1, len(grads))
            if float(torch.dot(grads[i], grads[j]).item()) < 0
        )
        info["pcgrad_used"]            = 1.0
        info["pcgrad_conflict_pairs"]  = float(n_conflicts)
        info["config_num_objectives"]  = float(len(objectives))
    
        return info
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # IMTL-G
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _step_with_imtlg(self, loss_tensors: dict) -> dict:
        """
        IMTL-G — Impartial Multi-Task Learning via Gradient (Liu et al., ICLR 2021).
    
        Finds weights α_i such that:
            û_i · g_combined = û_j · g_combined  for all i, j   (impartiality)
    
        Unlike ConFIG (unit-space bisector) and grad_blend (unit direction, raw
        magnitude), IMTL-G works in raw gradient space and uses magnitudes to
        set weights automatically. For the Laplacian loss with its 119× norm
        amplification, IMTL-G automatically down-weights g_lap to enforce
        impartiality — no manual spectral normalization required.
    
        Verification: imtlg_proj_{name} values should be equal across all
        objectives. Inequality indicates the library's solver is struggling.
        """
        info       = {}
        objectives = self._get_config_objectives(loss_tensors)
    
        assert objectives[0] == "diffusion_loss", (
            f"imtlg requires diffusion_loss at index 0, got: {objectives}"
        )
        if len(objectives) < 2:
            return {"imtlg_used": 0.0}
    
        # ── 1. per-objective gradients ───────────────────────────────────────────
        grads, grad_ok = self._compute_per_objective_grads(loss_tensors, objectives)
    
        # ── 2. fallback ──────────────────────────────────────────────────────────
        if not grad_ok:
            apply_gradient_vector(self.net, grads[0])
            return {"imtlg_used": 0.0}
    
        # ── 3. IMTL-G combined gradient ──────────────────────────────────────────
        g_imtlg = IMTLGOperator().calculate_gradient(grads)
        apply_gradient_vector(self.net, g_imtlg)
    
        # ── 4. diagnostics ───────────────────────────────────────────────────────
        info = self._compute_gradient_diagnostics(grads, objectives, g_combined=g_imtlg)
    
        # impartiality check: û_i · g_imtlg should be equal for all i
        grads_stack = torch.stack(grads)
        norms_vec   = grads_stack.norm(dim=1, keepdim=True).clamp(min=1e-10)
        units       = grads_stack / norms_vec
        for name, u in zip(objectives, units):
            info[f"imtlg_proj_{name}"] = float(torch.dot(u, g_imtlg).item())
    
        info["imtlg_used"]            = 1.0
        info["config_num_objectives"] = float(len(objectives))
    
        return info
    # ─────────────────────────────────────────────────────────────────────────────
    # GradNorm
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _step_with_gradnorm(self, loss_tensors: dict) -> dict:
        if self.gradnorm_balancer is None:
            loss_tensors['diffusion_loss'].backward()
            return {"gradnorm_used": 0.0}
    
        objectives = getattr(self, 'gradnorm_objectives', None)
        if objectives is None:
            loss_tensors['diffusion_loss'].backward()
            return {"gradnorm_used": 0.0}
    
        # filter to keys present in loss_tensors this step
        objectives = [k for k in objectives if k in loss_tensors]
    
        # guard: assert replaced with clean fallback
        if not objectives or objectives[0] != "diffusion_loss":
            loss_tensors['diffusion_loss'].backward()
            return {"gradnorm_used": 0.0}
    
        if len(objectives) < 2:
            loss_tensors['diffusion_loss'].backward()
            return {"gradnorm_used": 0.0}
    
        # guard: task count must match what balancer was built with
        if len(objectives) != self.gradnorm_balancer.num_tasks:
            loss_tensors['diffusion_loss'].backward()
            return {"gradnorm_used": 0.0, "gradnorm_task_mismatch": 1.0}
    
        losses = [loss_tensors[k] for k in objectives]
    
        # last shared ViT block — GradNorm computes norms here only
        shared_params = list(self.net.transformer_blocks[-1].parameters())
    
        # GradNorm step: updates w_i, returns weighted total loss (w detached)
        L_total, gradnorm_info, per_obj_grads = self.gradnorm_balancer.step(
            losses        = losses,
            objectives    = objectives,
            shared_params = shared_params,
            all_params    = list(self.net.parameters()),
        )
    
        # main backward on weighted total
        self.optimizer.zero_grad(set_to_none=True)
        L_total.backward()
    
        info = {"gradnorm_used": 1.0}
        info.update(gradnorm_info)
    
        return info
        
    def train_one_epoch(self):
        """Train for one epoch. batch_loss must return (loss_tensors:dict, loss_scalars:dict)."""
        self.net.train()
    
        batch_metrics = {}
        skipped = 0
        n_batches = 0
    
        mo_method = getattr(self, "mo_method", "weighted_sum")
    
        for batch_id, x in enumerate(self.train_loader):

    
            n_batches += 1
            self.global_step += 1
            self.optimizer.zero_grad(set_to_none=True)
    
            # Forward: get individual losses
            loss_tensors, loss_scalars = self.batch_loss(x)
    
            # Force scalars to floats
            loss_scalars = {
                k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                for k, v in (loss_scalars or {}).items()
            }
    
            did_step = 0.0
            mo_info = {}
    
            # ==========================================
            # Branch 1: NO MULTI-OBJECTIVE (NEW)
            # ==========================================
            if mo_method == "none":
                # Simple case: only diffusion loss, no MO overhead
                total_loss = loss_tensors['diffusion_loss']
                loss_scalars["total_loss"] = float(total_loss.detach().item())
                
                if np.isfinite(loss_scalars["total_loss"]):
                    total_loss.backward()
                    
                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                    
                    self.optimizer.step()
                    did_step = 1.0
                    
                    if hasattr(self, "update_ema"):
                        self.update_ema()
                    
                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()
                else:
                    skipped += 1
                    print(f"Unstable loss at epoch {self.epoch}, batch {batch_id} (skipping step)")
    
            # ==========================================
            # Branch 2: NON-ConFIG (regular MO backward)
            # ==========================================
            elif mo_method == "weighted_sum":
                total_loss, mo_info = self.combine_losses(loss_tensors)
    
                # scalar-friendly mo_info
                mo_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (mo_info or {}).items()
                }
    
                total_loss_scalar = float(total_loss.detach().item())
                loss_scalars["total_loss"] = total_loss_scalar
                loss_scalars.update(mo_info)
    
                if np.isfinite(total_loss_scalar):
                    total_loss.backward()
    
                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)
    
                    self.optimizer.step()
                    did_step = 1.0
    
                    if hasattr(self, "update_ema"):
                        self.update_ema()
    
                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()
                else:
                    skipped += 1
                    print(f"Unstable loss at epoch {self.epoch}, batch {batch_id} (skipping step)")
    
            # ==========================================
            # Branch 3: ConFIG (apply gradient vector)
            # ==========================================
            elif mo_method == "config":
                # total_loss for reporting only
                report_total, report_info = self.combine_losses(loss_tensors)
                report_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (report_info or {}).items()
                }
                loss_scalars["total_loss"] = float(report_total.detach().item())
                loss_scalars.update(report_info)
    
                # If total is non-finite, skip (still logs raw scalars)
                if np.isfinite(loss_scalars["total_loss"]):
                    # Compute + apply ConFIG gradient
                    config_info = self._step_with_config(loss_tensors)
    
                    # Clip + step
                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                       
    
                    self.optimizer.step()
                    did_step = 1.0
    
                    if hasattr(self, "update_ema"):
                        self.update_ema()
    
                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()
    
                    # log config diagnostics
                    loss_scalars.update(config_info)
                else:
                    skipped += 1
                    loss_scalars["config_used"] = 0.0
                    print(f"Unstable total_loss (reporting) at epoch {self.epoch}, batch {batch_id} (skipping step)")

            # ==========================================
            # Branch 4: Gradient Blending (professor's method)
            # ==========================================
            elif mo_method == "grad_blend":
                # total_loss for reporting only — same pattern as config branch
                report_total, report_info = self.combine_losses(loss_tensors)
                report_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (report_info or {}).items()
                }
                loss_scalars["total_loss"] = float(report_total.detach().item())
                loss_scalars.update(report_info)

                if np.isfinite(loss_scalars["total_loss"]):
                    # Compute + apply blended gradient
                    blend_info = self._step_with_grad_blend(loss_tensors)

                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                    self.optimizer.step()
                    did_step = 1.0

                    if hasattr(self, "update_ema"):
                        self.update_ema()

                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()

                    loss_scalars.update(blend_info)
                else:
                    skipped += 1
                    loss_scalars["blend_used"] = 0.0
                    print(f"Unstable total_loss at epoch {self.epoch}, batch {batch_id} (skipping step)")

            # ==========================================
            # Branch 5: PCGrad (gradient surgery)
            # ==========================================
            elif mo_method == "pcgrad":
                report_total, report_info = self.combine_losses(loss_tensors)
                report_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (report_info or {}).items()
                }
                loss_scalars["total_loss"] = float(report_total.detach().item())
                loss_scalars.update(report_info)

                if np.isfinite(loss_scalars["total_loss"]):
                    pcgrad_info = self._step_with_pcgrad(loss_tensors)

                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                    self.optimizer.step()
                    did_step = 1.0

                    if hasattr(self, "update_ema"):
                        self.update_ema()

                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()

                    loss_scalars.update(pcgrad_info)
                else:
                    skipped += 1
                    loss_scalars["pcgrad_used"] = 0.0
                    print(f"Unstable total_loss at epoch {self.epoch}, batch {batch_id} (skipping step)")

            # ==========================================
            # Branch 6: IMTL-G (impartial multi-task)
            # ==========================================
            elif mo_method == "imtlg":
            
                report_total, report_info = self.combine_losses(loss_tensors)
                report_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (report_info or {}).items()
                }
                loss_scalars["total_loss"] = float(report_total.detach().item())
                loss_scalars.update(report_info)

                if np.isfinite(loss_scalars["total_loss"]):
                    imtlg_info = self._step_with_imtlg(loss_tensors)

                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                    self.optimizer.step()
                    did_step = 1.0

                    if hasattr(self, "update_ema"):
                        self.update_ema()

                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()

                    loss_scalars.update(imtlg_info)
                else:
                    skipped += 1
                    loss_scalars["imtlg_used"] = 0.0
                    print(f"Unstable total_loss at epoch {self.epoch}, batch {batch_id} (skipping step)")
            # ==========================================
            # Branch 7: GradNorm
            # ==========================================
            elif mo_method == "gradnorm":
                report_total, report_info = self.combine_losses(loss_tensors)
                report_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (report_info or {}).items()
                }
                loss_scalars["total_loss"] = float(report_total.detach().item())
                loss_scalars.update(report_info)
            
                if np.isfinite(loss_scalars["total_loss"]):
                    gradnorm_info = self._step_with_gradnorm(loss_tensors)
            
                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)
            
                    self.optimizer.step()
                    did_step = 1.0
            
                    if hasattr(self, "update_ema"):
                        self.update_ema()
            
                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()
            
                    loss_scalars.update(gradnorm_info)
                else:
                    skipped += 1
                    loss_scalars["gradnorm_used"] = 0.0
                    print(f"Unstable total_loss at epoch {self.epoch}, batch {batch_id} (skipping step)")
            
            else:
            
                raise ValueError(f"Unknown mo_method: '{mo_method}'. "
                     f"Valid options: none, weighted_sum, config, grad_blend, pcgrad, imtlg")
            loss_scalars["did_step"] = float(did_step)

            # Accumulate batch metrics
            for k, v in loss_scalars.items():
                batch_metrics.setdefault(k, []).append(float(v))
    
        # Epoch means
        epoch_stats = {k: float(np.mean(v)) if len(v) > 0 else float("nan") for k, v in batch_metrics.items()}
        epoch_stats["num_batches"] = float(n_batches)
        epoch_stats["num_skipped"] = float(skipped)
    
        self._append_epoch_metrics(prefix="train", epoch_stats=epoch_stats)
    
        # per-epoch schedulers
        if getattr(self, "use_scheduler", False) and not getattr(self, "scheduler_step_per_batch", True):
            self.scheduler.step()
    
        # TensorBoard
        if getattr(self, "log", False):
            for k, v in epoch_stats.items():
                self.logger.add_scalar(f"train/{k}", v, self.epoch)
            if getattr(self, "use_scheduler", False):
                try:
                    self.logger.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], self.epoch)
                except Exception:
                    pass
    
        if hasattr(self, "_print_epoch_summary"):
            self._print_epoch_summary(epoch_stats)
    
        return epoch_stats
    def _append_epoch_metrics(self, prefix: str, epoch_stats: dict):
        """
        Store metrics into numpy arrays with names like:
          train_total_loss_epoch, train_diffusion_loss_epoch, ...
          val_total_loss_epoch, ...
        """
        for k, v in epoch_stats.items():
            attr = f"{prefix}_{k}_epoch"
            if not hasattr(self, attr):
                setattr(self, attr, np.array([], dtype=np.float64))
            arr = getattr(self, attr)
            setattr(self, attr, np.append(arr, float(v)))

    @torch.inference_mode()
    def validate_one_epoch(self):
        self.net.eval()
        batch_metrics = {}
        n_batches = 0
        
        for batch_id, x in enumerate(self.val_loader):
            n_batches += 1
            loss_tensors, loss_scalars = self.batch_loss(x)
            
            loss_scalars = {k: float(v.item() if hasattr(v, "item") else v) 
                           for k, v in loss_scalars.items()}
            
            # ========================================
            # Compute total loss (UPDATED)
            # ========================================
            mo_method = getattr(self, "mo_method", "weighted_sum")
            
            if mo_method == "none":
                # Simple case: only diffusion loss
                total_loss = loss_tensors['diffusion_loss']
                loss_scalars["total_loss"] = float(total_loss.item())
            else:
                # Use combine_losses for consistency
                total_loss, mo_info = self.combine_losses(loss_tensors)
                mo_info = {k: float(v.item() if hasattr(v, "item") else v) 
                          for k, v in mo_info.items()}
                loss_scalars["total_loss"] = float(total_loss.item())
                loss_scalars.update(mo_info)
            
            # Accumulate metrics
            for k, v in loss_scalars.items():
                batch_metrics.setdefault(k, []).append(float(v))
        
        # Compute epoch means
        epoch_stats = {k: np.mean(v) for k, v in batch_metrics.items()}
        epoch_stats["num_batches"] = n_batches
        
        self._append_epoch_metrics("val", epoch_stats)
        
        if self.log:
            for k, v in epoch_stats.items():
                self.logger.add_scalar(f"val/{k}", v, self.epoch)
        
        return epoch_stats
    
    def save_all_metrics(self):
        """Save all training/validation metrics to CSV"""
        import pandas as pd
        
        # Collect all *_epoch metrics
        metrics = {}
        for attr_name in self.__dict__:
            if attr_name.endswith('_epoch') and not attr_name.startswith('_'):
                try:
                    arr = getattr(self, attr_name)
                    if isinstance(arr, np.ndarray) and arr.size > 0:
                        metrics[attr_name] = arr.reshape(-1).astype(np.float64)
                except Exception:
                    pass
        
        if not metrics:
            return
        
        # Pad to same length with NaN
        max_len = max(len(v) for v in metrics.values())
        padded = {k: np.concatenate([v, np.full(max_len - len(v), np.nan)]) 
                  if len(v) < max_len else v 
                  for k, v in metrics.items()}
        
        # Add epoch column
        padded['epoch'] = np.arange(max_len)
        
        # Create DataFrame with sorted columns
        df = pd.DataFrame(padded)
        train_cols = sorted([c for c in df.columns if c.startswith('train_')])
        val_cols = sorted([c for c in df.columns if c.startswith('val_')])
        other_cols = sorted([c for c in df.columns if c not in train_cols + val_cols + ['epoch']])
        df = df[['epoch'] + train_cols + val_cols + other_cols]
        
        # Save
        csv_path = self.doc.get_file("train_val_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved {len(df.columns)-1} metrics to CSV")

    def batch_loss(self, x):
        pass

    def generate_Einc_ds1(self, energy=None, sample_multiplier=1000):
        """ generate the incident energy distribution of CaloChallenge ds1
            sample_multiplier controls how many samples are generated: 10* sample_multiplier for low energies,
            and 5, 3, 2, 1 times sample multiplier for the highest energies

        """
        ret = np.logspace(8, 18, 11, base=2)
        ret = np.tile(ret, 10)
        ret = np.array(
            [*ret, *np.tile(2. ** 19, 5), *np.tile(2. ** 20, 3), *np.tile(2. ** 21, 2), *np.tile(2. ** 22, 1)])
        ret = np.tile(ret, sample_multiplier)
        if energy is not None:
            ret = ret[ret == energy]
        np.random.shuffle(ret)
        return ret
    @torch.inference_mode()
    def sample_trained_model_energy(self,size=10**5,sampling_type='ddim'):
        #print("inside the sample_trained_model_energy! ",self.params)
        self.eval()
        energy_model = self.load_other(self.params['energy_model'])
        #print("after loading the energy model: ",self.params)
        t_0 = time.time()

        Einc = torch.tensor(
            10**np.random.uniform(3, 6, size=get(self.params, "n_samples", 10**5)) 
            if self.params['eval_dataset'] in ['2', '3'] else
            self.generate_Einc_ds1(energy=self.single_energy),
            dtype=torch.get_default_dtype(),
            device=self.device
        ).unsqueeze(1)
        
        # transform Einc to basis used in training
        dummy, transformed_cond = None, Einc
        print("before starting the loop Einc min and max: ", Einc.min(), Einc.max())
        for fn in self.transforms:
            if hasattr(fn, 'cond_transform'):
                dummy, transformed_cond = fn(dummy, transformed_cond)
                print("Einc min and max: ",transformed_cond.min(), transformed_cond.max())

        batch_size_sample = get(self.params, "batch_size_sample", 10000)
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
        )
        #print("param----s", self.params)
        sample = torch.vstack([energy_model.sample_batch(c,sampling_type=sampling_type).cpu()
                               for c in transformed_cond_loader])
        
        t_1 = time.time()
        sampling_time = t_1 - t_0
        self.params["sample_time"] = sampling_time
        print(
            f"generate_samples: Finished generating {len(sample)} samples "
            f"after {sampling_time} s.", flush=True
        )
        
        return sample, transformed_cond.cpu()

    @torch.inference_mode()
    def sample_n(self, size=10**5):
        print("Inside sample_n")
        self.eval()
        

        # if self.net.bayesian:
        #     self.net.map = get(self.params, "fix_mu", False)
        #     for bay_layer in self.net.bayesian_layers:
        #         bay_layer.random = None
       
        # sample = []

        t_0 = time.time()

        Einc = torch.tensor(
            10**np.random.uniform(3, 6, size=get(self.params, "n_samples", 10**3)) # remember to change this to 10^5
            if self.params['eval_dataset'] in ['2', '3'] else
            self.generate_Einc_ds1(energy=self.single_energy),
            dtype=torch.get_default_dtype(),
            device=self.device
        ).unsqueeze(1)
        
        # transform Einc to basis used in training
        dummy, transformed_cond = None, Einc
        for fn in self.transforms:
            if hasattr(fn, 'cond_transform'):
                dummy, transformed_cond = fn(dummy, transformed_cond)

        batch_size_sample = get(self.params, "batch_size_sample", 100)
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
        )
        
        # sample u_i's if self is a shape model
        if self.params['model_type'] == 'shape': 
            # load energy model
            energy_model = self.load_other(self.params['energy_model'])

            if self.params.get('sample_us', False):
                # sample us
                u_samples = torch.vstack([
                    energy_model.sample_batch(c) for c in transformed_cond_loader
                ])
                if self.latent:
                    # post-process u-samples according to energy config
                    # CAUTION: shape config pre-processing may then also be necessary!
                    dummy = torch.empty(1, 1)
                    for fn in energy_model.transforms[:0:-1]: # skip NormalizeByElayer
                        u_samples, dummy = fn(u_samples, dummy, rev=True)                
                transformed_cond = torch.cat([transformed_cond, u_samples], dim=1)
            else: # optionally use truth us
                print("generating from real_ data")
                transformed_cond = CaloChallengeDataset(
                self.params.get('eval_hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                transform=self.transforms,
                device=self.device,
                single_energy=self.single_energy
                ).energy
            
            # concatenate with Einc
            transformed_cond_loader = DataLoader(
                dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
            )
        #print("Before calling the autoregressive_sample_batch: ", batch_size_sample)  

        # DEBUG_NUM_BATCHES = 20  # Change this to control how many batches
    
        # sample = torch.vstack([
        #     self.sample_batch(c).cpu() 
        #     for c in islice(transformed_cond_loader, DEBUG_NUM_BATCHES)
        # ])
        
        # # Also limit transformed_cond to match
        # transformed_cond_debug = torch.cat([
        #     c for c in islice(
        #         DataLoader(transformed_cond, batch_size=batch_size_sample, shuffle=False),
        #         DEBUG_NUM_BATCHES
        #     )
        # ])

        sample = torch.vstack([self.sample_batch(c).cpu() for c in (transformed_cond_loader)])
        

        t_1 = time.time()
        sampling_time = t_1 - t_0
        self.params["sample_time"] = sampling_time
        print(
            f"generate_samples: Finished generating {len(sample)} samples "
            f"after {sampling_time} s.", flush=True
        )

        return sample, transformed_cond.cpu()
    
    def reconstruct_n(self,):
        print("inside reconstruct_n")
        if not hasattr(self, 'train_loader'):
            self.train_loader, self.val_loader, self.bounds = get_loaders(
                self.params.get('hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                self.params.get('val_frac'),
                self.params.get('batch_size_sample'),
                self.transforms,
                self.params.get('eps', 1.e-10),
                device=self.device,
                shuffle=False,
                width_noise=self.params.get('width_noise', 1.e-6),
                single_energy=self.params.get('single_energy', None)
            )

        recos = []
        energies = []

        self.eval()
        for n, x in enumerate(self.train_loader):
            reco, cond = self.sample_batch(x)
            recos.append(reco)
            energies.append(cond)
        for n, x in enumerate(self.val_loader):
            reco, cond = self.sample_batch(x)
            recos.append(reco)
            energies.append(cond)

        recos = torch.vstack(recos)
        energies = torch.vstack(energies)
        return recos, energies

    def sample_batch(self, batch):
        pass
  
            
    def plot_samples_old(self, sample_path, reference_path, doc):
        
        samples = torch.load(sample_path)
        reference = torch.load(reference_path)

        samples[:,1:] = torch.clip(samples[:,1:], min=0., max=1.)
        reference[:,1:] = torch.clip(reference[:,1:], min=0., max=1.)

        samples_np = samples.detach().cpu().numpy()
        reference_np = reference.detach().cpu().numpy()

        print("LBL....", self.params['LBL'])

        plot_ui_dists(samples_np, reference_np, documenter=doc, LBL=self.params['LBL'])
        evaluate.eval_ui_dists(samples_np, reference_np, documenter=doc, params=self.params)
        

        
    def plot_samples(self, samples, conditions, name="", energy=None, doc=None):
        print(f"Samples requires_grad {samples.requires_grad}")
        transforms = self.transforms
        if doc is None: doc = self.doc

        if self.params['model_type'] == 'energy':
            reference = CaloChallengeDataset(
                self.params.get('eval_hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                transform=transforms, # TODO: Or, apply NormalizeEByLayer popped from model transforms
                device=self.device,
                single_energy=self.single_energy
            ).layers
            
            # postprocess
            #print("before plotting shape of SAMPLES: ",samples.shape)
            print("*******  Starting Reverse Transformation *******")
            for fn in transforms[::-1]:
                if fn.__class__.__name__ == 'NormalizeByElayer':
                    break # this might break plotting
                print("Sampling information")
                samples, _ = fn(samples, conditions, rev=True)
                print("Geant4 information")
                reference, _ = fn(reference, conditions, rev=True)
            torch.save(samples, doc.get_file("samples.pt"))
            torch.save(reference, doc.get_file("reference.pt"))
            # clip u_i's (except u_0) to [0,1] 
            samples[:,1:] = torch.clip(samples[:,1:], min=0., max=1.)
            reference[:,1:] = torch.clip(reference[:,1:], min=0., max=1.)
            
            print("LBL....",self.params['LBL'])
            plot_ui_dists(
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
                documenter=doc,
                LBL=self.params['LBL']
            )
            #plot_losses(self.train_losses,self.val_losses, documenter=doc)
            print("before passing to the evaluate: ",samples.requires_grad, reference.requires_grad)
            evaluate.eval_ui_dists(
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
                documenter=doc,
                params=self.params,
            )
        else:
            if self.latent:
                #save generated latent space
                self.save_sample(samples, conditions, name=name+'_latent', doc=doc) 
                
           
            print("\n" + "="*80)
            print("Transform Tracking")
            print("="*80)
            print(f"{'Step':<5} {'Transform':<30} {'Shape':<20} {'Min':<12} {'Max':<12}")
            print("-"*80)
            
            # Initial
            print(f"{'Init':<5} {'Model Output':<30} {str(samples.shape):<20} {samples.min().item():<12.6f} {samples.max().item():<12.6f}")
            torch.save(samples.cpu(), self.doc.get_file('debug_samples_0_initial.pt'))
            
            # Apply transforms
            for i, fn in enumerate(transforms[::-1]):
                samples, conditions = fn(samples, conditions, rev=True)
                #name = fn.__class__.__name__
                #print(f"{i+1:<5} {name:<30} {str(samples.shape):<20} {samples.min().item():<12.6f} {samples.max().item():<12.6f}")
                #torch.save(samples.cpu(), self.doc.get_file(f'debug_samples_{i+1}_{name}.pt'))
            
            print("="*80 + "\n")

            samples = samples.detach().cpu().numpy()
            conditions = conditions.detach().cpu().numpy()
            
            self.save_sample(samples, conditions, name=name, doc=doc)
            
            evaluate.run_from_py(samples, conditions, doc, self.params,name=name)

    def plot_saved_samples(self, name="", energy=None, doc=None):
        if doc is None: doc = self.doc
        mode = self.params.get('eval_mode', 'all')
        script_args = (
            f"-i {doc.basedir}/ "
            f"-r {self.params['eval_hdf5_file']} -m {mode} --cut {self.params['eval_cut']} "
            f"-d {self.params['eval_dataset']} --output_dir {doc.basedir}/final/ --save_mem"
        ) + (f" --energy {energy}" if energy is not None else '')
        evaluate.main(script_args.split())

    def save_sample(self, sample, energies, name="", doc=None):
        """Save sample in the correct format"""
        if doc is None: doc = self.doc
        save_file = h5py.File(doc.get_file(f'samples_{name}.hdf5'), 'w')
        save_file.create_dataset('incident_energies', data=energies)
        save_file.create_dataset('showers', data=sample)
        save_file.close()            
 
    def save(self, epoch=""):
        """ Save the model, and more if needed"""
        # torch.save({"opt": self.optimizer.state_dict(),
        #             "net": self.net.state_dict(),
        #             "losses": self.train_losses_epoch,
        #             "epoch": self.epoch,
        #             "scheduler": self.scheduler.state_dict()}
        #             , self.doc.get_file(f"model{epoch}.pt"))
        
        """ Save the model, and more if needed"""
        save_dict = {
            "opt": self.optimizer.state_dict(),
            "net": self.net.state_dict(),
            "epoch": self.epoch,
            "scheduler": self.scheduler.state_dict()
        }
        if self.gradnorm_balancer is not None:
            save_dict["gradnorm_log_w"] = self.gradnorm_balancer.log_w.data
            save_dict["gradnorm_L0"]    = self.gradnorm_balancer.L0
            save_dict["gradnorm_w_optim"] = self.gradnorm_balancer.w_optim.state_dict()
        # save_dict["energy_loss_cond_epoch"] = self.energy_loss_cond_epoch
        # save_dict["energy_loss_truth_epoch"] = self.energy_loss_truth_epoch

        # Add EMA model if it exists
        if hasattr(self, 'ema_model') and self.ema_model is not None:
            save_dict["ema_net"] = self.ema_model.state_dict()

        torch.save(save_dict, self.doc.get_file(f"model{epoch}.pt"))

    def load(self, epoch=""):
        """ Load the model, and more if needed"""
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device, weights_only=False)
        self.net.load_state_dict(state_dicts["net"])
        
        if "losses" in state_dicts:
            self.train_losses_epoch = state_dicts.get("losses", {})
        if "epoch" in state_dicts:
            self.epoch = state_dicts.get("epoch", 0)
       
        if "gradnorm_log_w" in state_dicts and self.gradnorm_balancer is not None:
            self.gradnorm_balancer.log_w.data = state_dicts["gradnorm_log_w"]
            self.gradnorm_balancer.L0         = state_dicts["gradnorm_L0"]
            self.gradnorm_balancer.w_optim.load_state_dict(state_dicts["gradnorm_w_optim"])
        #if "opt" in state_dicts:
        #    self.optimizer.load_state_dict(state_dicts["opt"])
        #if "scheduler" in state_dicts:
        #    self.scheduler.load_state_dict(state_dicts["scheduler"])
        self.net.to(self.device)

        

        
        # Load EMA model if it exists
        if "ema_net" in state_dicts and hasattr(self, 'ema_model') and self.ema_model is not None:
            self.ema_model.load_state_dict(state_dicts["ema_net"])

    def load_other(self, model_dir):
        """ Load a different model (e.g. to sample u_i's)"""
        
        with open(os.path.join(model_dir, 'params.yaml')) as f:
            params_old = yaml.load(f, Loader=yaml.FullLoader)

        model_class = params_old['model']
        # choose model
        if model_class == 'TBD':
            Model = self.__class__
        elif model_class == 'TransfusionAR':
            from Models import TransfusionAR
            Model = TransfusionAR
        elif model_class == 'AE':
            from Models import AE
            Model = AE
        elif model_class =='TransfusionDDPM':
            from Models import TransfusionDDPM
            Model = TransfusionDDPM
        elif model_class=='TBD_DIFF':
            from Models import TBD_DIFF
            Model=TBD_DIFF
        else:
            raise ValueError(f"Unknown model_class: '{model_class}'")

            

        # load model
        doc_trained = Documenter(None, existing_run=model_dir, read_only=True)
        other = Model(params_old, self.device, doc_trained)
        state_dicts = torch.load(
            os.path.join(model_dir, 'model.pt'), map_location=self.device, weights_only=False
        )
        other.net.load_state_dict(state_dicts["net"])
        
        # use eval mode and freeze weights
        other.eval()
        for p in other.parameters():
            p.requires_grad = False

        return other
