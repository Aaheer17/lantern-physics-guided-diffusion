# standard python libraries
import numpy as np
import torch
import torch.nn as nn
import os, time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages

import sys

# Other functions of project
from Util.util import *
from datasets import *
from documenter import Documenter
from plotting_util import *
from transforms import *
from challenge_files import *
from challenge_files import evaluate # avoid NameError: 'evaluate' is not defined
import Models
from Models import *
import time
from itertools import islice
import pwd
import pandas as pd
def get_username():
    return pwd.getpwuid(os.getuid())[0]


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
        self.attn_epoch=[0,10,20,30,40,50,60,70,80,90,99]
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
        self.train_losses = np.array([])
        self.train_losses_epoch = np.array([])
        self.val_losses = np.array([])
        self.val_losses_epoch = np.array([])

        self.moment_loss_epoch = np.array([])
        self.moment_loss = np.array([])   # optional, if you want batch-wise history too
        self.val_moment_loss_epoch = np.array([])

        self.sparsity_l1_real_epoch = np.array([])
        self.sparsity_l1_gen_epoch  = np.array([])
        self.occ_real_epoch         = np.array([])
        self.occ_gen_epoch          = np.array([])
        self.sparsity_match_epoch   = np.array([])
        self.occ_match_epoch        = np.array([])
        
        # if you want validation too:
        self.val_moment_loss_epoch = np.array([])
        self.val_sparsity_l1_real_epoch = np.array([])
        self.val_sparsity_l1_gen_epoch  = np.array([])
        self.val_occ_real_epoch         = np.array([])
        self.val_occ_gen_epoch          = np.array([])
        self.val_sparsity_match_epoch = np.array([])
        self.val_occ_match_epoch = np.array([])

        # NEW: energy-loss tracking (epoch-level)
        self.energy_loss_cond_epoch = np.array([])   # mean over batches each epoch
        self.energy_loss_truth_epoch = np.array([])

        # (optional) batch-level history if you want it later
        self.energy_loss_cond = np.array([])
        self.energy_loss_truth = np.array([])

        self.n_trainbatches = len(self.train_loader)
        self.n_traindata = self.n_trainbatches*self.batch_size
        self.set_optimizer(steps_per_epoch=self.n_trainbatches)
        
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
                eps = params.get("eps", 1e-6),
                weight_decay = params.get("weight_decay", 0.)
                )
        self.scheduler = set_scheduler(self.optimizer, params, steps_per_epoch, last_epoch=-1)

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
        self.save_all_metrics_csv()
        np.save(self.doc.get_file('train_losses_epoch.npy'),
                self.train_losses_epoch)
        
        np.save(self.doc.get_file('moment_loss_epoch.npy'),
                self.moment_loss_epoch)
        np.save(self.doc.get_file('moment_loss_batch.npy'),
                self.moment_loss)
        
        np.save(self.doc.get_file('sparsity_l1_real_epoch.npy'),
                self.sparsity_l1_real_epoch)
        
        np.save(self.doc.get_file('sparsity_l1_gen_epoch.npy'),
                self.sparsity_l1_gen_epoch)
        
        np.save(self.doc.get_file('occ_real_epoch.npy'),
                self.occ_real_epoch)
        
        np.save(self.doc.get_file('occ_gen_epoch.npy'),
                self.occ_gen_epoch)
        
        np.save(self.doc.get_file('sparsity_match_epoch.npy'),
                self.sparsity_match_epoch)
        
        np.save(self.doc.get_file('occ_match_epoch.npy'),
                self.occ_match_epoch)


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
    def train_one_epoch(self):
        train_losses = np.array([])
        moment_losses = []
    
        # NEW: sparsity stats per batch (store scalars)
        sp_l1_real_list, sp_l1_gen_list = [], []
        occ_real_list, occ_gen_list = [], []
        sp_match_list, occ_match_list = [], []
    
        for batch_id, x in enumerate(self.train_loader):
    
            self.optimizer.zero_grad(set_to_none=True)
    
            loss_tensors, loss_scalars = self.batch_loss(x) 
    
            if self.epoch in self.attn_epoch and batch_id == 0:
                if hasattr(self.net, 'plot_attention_rollout'):
                    self.net.plot_attention_rollout(
                        save_path=self.doc.get_file(f'attn_epoch{self.epoch}_batch{batch_id}.png')
                    )
    
            if np.isfinite(loss.item()):
                loss.backward()
    
                clip = self.params.get('clip_gradients_to', None)
                if clip:
                    nn.utils.clip_grad_norm_(self.net.parameters(), clip)
    
                self.optimizer.step()
                if hasattr(self, 'update_ema'):
                    self.update_ema()
    
                train_losses = np.append(train_losses, loss.item())
                moment_losses.append(moment_loss.item())
    
                # NEW: record sparsity stats if enabled
                if sparsity_stats is not None:
                    sp_l1_real_list.append(sparsity_stats["sparsity_l1_real"].item())
                    sp_l1_gen_list.append(sparsity_stats["sparsity_l1_gen"].item())
                    occ_real_list.append(sparsity_stats["occ_real"].item())
                    occ_gen_list.append(sparsity_stats["occ_gen"].item())
                    sp_match_list.append(sparsity_stats["sparsity_match_loss"].item())
                    occ_match_list.append(sparsity_stats["occ_match_loss"].item())
    
                if self.use_scheduler:
                    self.scheduler.step()
    
            else:
                print(f"train_model: Unstable loss. Skipped backprop for epoch {self.epoch}, batch_id {batch_id}")
    
        # ---- epoch means ----
        self.train_losses_epoch = np.append(self.train_losses_epoch, train_losses.mean())
        self.train_losses = np.concatenate([self.train_losses, train_losses], axis=0)
    
        mean_moment = float(np.mean(moment_losses)) if len(moment_losses) else float("nan")
        self.moment_loss_epoch = np.append(self.moment_loss_epoch, mean_moment)
    
        # optional: store all batch values
        self.moment_loss = np.concatenate([self.moment_loss, np.array(moment_losses)], axis=0)
    
        # NEW: epoch means for sparsity stats
        def _mean_or_nan(lst):
            return float(np.mean(lst)) if len(lst) else float("nan")
    
        self.sparsity_l1_real_epoch = np.append(self.sparsity_l1_real_epoch, _mean_or_nan(sp_l1_real_list))
        self.sparsity_l1_gen_epoch  = np.append(self.sparsity_l1_gen_epoch,  _mean_or_nan(sp_l1_gen_list))
        self.occ_real_epoch         = np.append(self.occ_real_epoch,         _mean_or_nan(occ_real_list))
        self.occ_gen_epoch          = np.append(self.occ_gen_epoch,          _mean_or_nan(occ_gen_list))
        self.sparsity_match_epoch   = np.append(self.sparsity_match_epoch,   _mean_or_nan(sp_match_list))
        self.occ_match_epoch        = np.append(self.occ_match_epoch,        _mean_or_nan(occ_match_list))
    
        if self.log:
            self.logger.add_scalar("train_losses_epoch", self.train_losses_epoch[-1], self.epoch)
            self.logger.add_scalar("moment_loss_epoch", self.moment_loss_epoch[-1], self.epoch)
    
            # NEW: log sparsity stats
            self.logger.add_scalar("sparsity_l1_real_epoch", self.sparsity_l1_real_epoch[-1], self.epoch)
            self.logger.add_scalar("sparsity_l1_gen_epoch",  self.sparsity_l1_gen_epoch[-1],  self.epoch)
            self.logger.add_scalar("occ_real_epoch",         self.occ_real_epoch[-1],         self.epoch)
            self.logger.add_scalar("occ_gen_epoch",          self.occ_gen_epoch[-1],          self.epoch)
    
            # optional
            self.logger.add_scalar("sparsity_match_epoch",   self.sparsity_match_epoch[-1],   self.epoch)
            self.logger.add_scalar("occ_match_epoch",        self.occ_match_epoch[-1],        self.epoch)
    
            if self.use_scheduler:
                self.logger.add_scalar("learning_rate_epoch", self.scheduler.get_last_lr()[0], self.epoch)


    # @torch.inference_mode()
    # def validate_one_epoch(self):
        
    #     val_losses = np.array([])
    #     # iterate batch wise over input
    #     for batch_id, x in enumerate(self.val_loader):

    #         # calculate batch loss
    #         loss,_ = self.batch_loss(x)
    #         val_losses = np.append(val_losses, loss.item())
    #         # if self.log:
    #         #     self.logger.add_scalar("val_losses", val_losses[-1], self.epoch*self.n_trainbatches + batch_id)

    #     self.val_losses_epoch = np.append(self.val_losses_epoch, val_losses.mean())
    #     self.val_losses = np.concatenate([self.val_losses, val_losses], axis=0)
    #     if self.log:
    #         self.logger.add_scalar("val_losses_epoch", self.val_losses_epoch[-1], self.epoch)


    @torch.inference_mode()
    def validate_one_epoch(self):
        val_losses = np.array([])
        val_moment = []
    
        sp_l1_real_list, sp_l1_gen_list = [], []
        occ_real_list, occ_gen_list = [], []
        sp_match_list, occ_match_list = [], []
    
        for batch_id, x in enumerate(self.val_loader):
            loss, moment_loss, sparsity_stats = self.batch_loss(x)
    
            val_losses = np.append(val_losses, loss.item())
            val_moment.append(moment_loss.item())
    
            if sparsity_stats is not None:
                sp_l1_real_list.append(sparsity_stats["sparsity_l1_real"].item())
                sp_l1_gen_list.append(sparsity_stats["sparsity_l1_gen"].item())
                occ_real_list.append(sparsity_stats["occ_real"].item())
                occ_gen_list.append(sparsity_stats["occ_gen"].item())

                
                # OPTIONAL: add these only if you compute them in batch_loss
                if "sparsity_match" in sparsity_stats:
                    sp_match_list.append(sparsity_stats["sparsity_match"].item())
                if "occ_match" in sparsity_stats:
                    occ_match_list.append(sparsity_stats["occ_match"].item())
    
        self.val_losses_epoch = np.append(self.val_losses_epoch, val_losses.mean())
        self.val_losses = np.concatenate([self.val_losses, val_losses], axis=0)
    
        mean_val_moment = float(np.mean(val_moment)) if len(val_moment) else float("nan")
        self.val_moment_loss_epoch = np.append(self.val_moment_loss_epoch, mean_val_moment)
    
        # sparsity epoch means
        def _mean_or_nan(lst):
            return float(np.mean(lst)) if len(lst) else float("nan")
    
        self.val_sparsity_l1_real_epoch = np.append(self.val_sparsity_l1_real_epoch, _mean_or_nan(sp_l1_real_list))
        self.val_sparsity_l1_gen_epoch  = np.append(self.val_sparsity_l1_gen_epoch,  _mean_or_nan(sp_l1_gen_list))
        self.val_occ_real_epoch         = np.append(self.val_occ_real_epoch,         _mean_or_nan(occ_real_list))
        self.val_occ_gen_epoch          = np.append(self.val_occ_gen_epoch,          _mean_or_nan(occ_gen_list))

        self.val_sparsity_match_epoch = np.append(
        self.val_sparsity_match_epoch, _mean_or_nan(sp_match_list)
        )
        self.val_occ_match_epoch = np.append(
            self.val_occ_match_epoch, _mean_or_nan(occ_match_list)
        )
        
        if self.log:
            self.logger.add_scalar("val_losses_epoch", self.val_losses_epoch[-1], self.epoch)
            self.logger.add_scalar("val_moment_loss_epoch", self.val_moment_loss_epoch[-1], self.epoch)
    
            self.logger.add_scalar("val_sparsity_l1_real_epoch", self.val_sparsity_l1_real_epoch[-1], self.epoch)
            self.logger.add_scalar("val_sparsity_l1_gen_epoch",  self.val_sparsity_l1_gen_epoch[-1],  self.epoch)
            self.logger.add_scalar("val_occ_real_epoch",         self.val_occ_real_epoch[-1],         self.epoch)
            self.logger.add_scalar("val_occ_gen_epoch",          self.val_occ_gen_epoch[-1],          self.epoch)

            self.logger.add_scalar(
            "val_sparsity_match_epoch", self.val_sparsity_match_epoch[-1], self.epoch
            )
            self.logger.add_scalar(
                "val_occ_match_epoch", self.val_occ_match_epoch[-1], self.epoch
            )
    
    def save_all_metrics_csv(self):
        data = {
            # train
            "train_losses_epoch": self.train_losses_epoch,
            "moment_loss_epoch": self.moment_loss_epoch,
            "sparsity_l1_real_epoch": self.sparsity_l1_real_epoch,
            "sparsity_l1_gen_epoch":  self.sparsity_l1_gen_epoch,
            "occ_real_epoch":         self.occ_real_epoch,
            "occ_gen_epoch":          self.occ_gen_epoch,
            "sparsity_match_epoch":   self.sparsity_match_epoch,
            "occ_match_epoch":        self.occ_match_epoch,
        
            # val (may be shorter)
            "val_losses_epoch": self.val_losses_epoch,
            "val_moment_loss_epoch": self.val_moment_loss_epoch,
            "val_sparsity_l1_real_epoch": self.val_sparsity_l1_real_epoch,
            "val_sparsity_l1_gen_epoch":  self.val_sparsity_l1_gen_epoch,
            "val_occ_real_epoch":         self.val_occ_real_epoch,
            "val_occ_gen_epoch":          self.val_occ_gen_epoch,
            "val_sparsity_match_epoch":   self.val_sparsity_match_epoch,
            "val_occ_match_epoch":        self.val_occ_match_epoch,
        }
        
        # Convert to 1D numpy arrays
        for k, v in data.items():
            data[k] = np.asarray(v).reshape(-1)
        
        # Pad all arrays to the same length with NaN
        max_len = max(len(v) for v in data.values())
        for k, v in data.items():
            if len(v) < max_len:
                pad = np.full((max_len - len(v),), np.nan, dtype=np.float64)
                data[k] = np.concatenate([v.astype(np.float64, copy=False), pad], axis=0)
            else:
                data[k] = v.astype(np.float64, copy=False)
        
        df = pd.DataFrame(data)
        
        csv_path = self.doc.get_file("train_val_metrics.csv")
        df.to_csv(csv_path, index=False)


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
        print("Before calling the autoregressive_sample_batch: ", batch_size_sample)  
        used_conds = []
        used_samples = []
        
        for c in islice(transformed_cond_loader, 2):  # first 2 batches
            used_conds.append(c.cpu())
            used_samples.append(self.sample_batch(c).cpu())
        
        sample = torch.vstack(used_samples)
        used_conditions = torch.vstack(used_conds)
     

        #sample = torch.vstack([self.sample_batch(c).cpu() for c in (transformed_cond_loader[0:3])])
        print("shape of sample: ",sample.shape)
        #sample = torch.vstack([self.autoregressive_sample_batch(c).cpu() for c in transformed_cond_loader])

        t_1 = time.time()
        sampling_time = t_1 - t_0
        self.params["sample_time"] = sampling_time
        print(
            f"generate_samples: Finished generating {len(sample)} samples "
            f"after {sampling_time} s.", flush=True
        )

        return sample, used_conditions.cpu()
    
    def reconstruct_n(self,):
        print("inside reconstruct_n")
        if ~hasattr(self, 'train_loader'):
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
    def plot_samples_mimic(self, samples, conditions,name="",energy=None, doc=None):
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
            )
            ref_layer=reference.layers
            ref_energy=reference.energy
            print("shape of ref_layer and ref_energy: ", ref_layer.shape, ref_energy.min(),ref_energy.max())
            print("----------------End of Geant4----------------")
            # postprocess
            print("before plotting shape of SAMPLES and Conditions: ",samples.shape,conditions.min(),conditions.max())
            for fn in transforms[::-1]:
                # if fn.__class__.__name__ == 'NormalizeByElayer':
                #     break # this might break plotting
                samples, conditions = fn(samples, conditions, rev=True)
                ref_layer, ref_energy = fn(ref_layer, ref_energy, rev=True)
                print("inside the loop : ", samples.shape, conditions.shape, ref_layer.shape, ref_energy.shape)
                
                #reference, conditions_r = fn(reference, conditions, rev=True)
            samples = samples.detach().cpu().numpy()
            conditions_s = conditions.detach().cpu().numpy()
#             samples = samples.detach().cpu().numpy()
#             conditions_s = conditions_s.detach().cpu().numpy()
            
#             reference=reference.detach().cpu().numpy()
#             conditions_r = conditions_r.detach().cpu().numpy()
                
            s_dict={'Layer' : samples,
                   'Incident Energy' : conditions_s }

            torch.save(s_dict, doc.get_file('s_dict.pt'))
        
           
            reference= ref_layer.detach().cpu().numpy()
            conditions_r = ref_energy.detach().cpu().numpy()
            
            r_dict={
                'Layer' : reference,
                'Incident Energy' : conditions_r
                
            }
            
        
            
            torch.save(r_dict, doc.get_file('r_dict.pt'))
            
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
                
            np.save(self.doc.get_file("samples_before_postprocess.npy"),samples.detach().cpu().numpy())
            np.save(self.doc.get_file("conditions_before_postprocess.npy"),conditions.detach().cpu().numpy())
            # postprocess
            for idx,fn in enumerate(transforms[::-1]):
                
                fn_name = fn.__class__.__name__

                # BEFORE
                print(
                    f"[BEFORE] {fn_name} | "
                    f"samples shape={tuple(samples.shape)} "
                    f"min={samples.min().item():.6e} "
                    f"max={samples.max().item():.6e} "
                    f"mean={samples.mean().item():.6e} | "
                    f"conditions shape={tuple(conditions.shape)} "
                    f"min={conditions.min().item():.6e} "
                    f"max={conditions.max().item():.6e} "
                    f"mean={conditions.mean().item():.6e}"
                )
                samples, conditions = fn(samples, conditions, rev=True)

                np.save(self.doc.get_file(f"samples_after_{fn_name}_{idx+1}.npy"),samples.detach().cpu().numpy())
                np.save(self.doc.get_file(f"conditions_after_{fn_name}_{idx+1}.npy"),conditions.detach().cpu().numpy())
            
            samples = samples.detach().cpu().numpy()
            conditions = conditions.detach().cpu().numpy()

            self.save_sample(samples, conditions, name=name, doc=doc)
            #REMOVE THIS COMMENT
            evaluate.run_from_py(samples, conditions, doc, self.params)

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
        save_file = h5py.File(doc.get_file(f'samples{name}_debug.hdf5'), 'w')
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
            "losses": self.train_losses_epoch,
            "epoch": self.epoch,
            "scheduler": self.scheduler.state_dict()
        }
        # save_dict["energy_loss_cond_epoch"] = self.energy_loss_cond_epoch
        # save_dict["energy_loss_truth_epoch"] = self.energy_loss_truth_epoch

        save_dict["moment_loss_epoch"] = self.moment_loss_epoch
        save_dict['moment_loss_batch'] = self.moment_loss
        save_dict["sparsity_l1_real_epoch"] = self.sparsity_l1_real_epoch
        save_dict["sparsity_l1_gen_epoch"]  = self.sparsity_l1_gen_epoch
        save_dict["occ_real_epoch"]         = self.occ_real_epoch
        save_dict["occ_gen_epoch"]          = self.occ_gen_epoch
        save_dict["sparsity_match_epoch"]   = self.sparsity_match_epoch
        save_dict["occ_match_epoch"]        = self.occ_match_epoch


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
        #if "opt" in state_dicts:
        #    self.optimizer.load_state_dict(state_dicts["opt"])
        #if "scheduler" in state_dicts:
        #    self.scheduler.load_state_dict(state_dicts["scheduler"])
        self.net.to(self.device)

        if "energy_loss_cond_epoch" in state_dicts:
            self.energy_loss_cond_epoch = state_dicts["energy_loss_cond_epoch"]
        else:
            self.energy_loss_cond_epoch = np.array([])
        
        if "energy_loss_truth_epoch" in state_dicts:
            self.energy_loss_truth_epoch = state_dicts["energy_loss_truth_epoch"]
        else:
            self.energy_loss_truth_epoch = np.array([])

        
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
        if model_class == 'TransfusionAR':
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
