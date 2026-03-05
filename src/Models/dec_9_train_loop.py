def train_one_epoch(self):
    
        # one-time setup
        if not hasattr(self, "opt_step"):
            self.opt_step = 0
    
        # CSV writer
        if not hasattr(self, "csv_writer") or self.csv_writer is None:
            import os, csv
            csv_path = self.doc.get_file("training_log.csv")
            need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
            self.csv_f = open(csv_path, "a", newline="")
            self.csv_fields = [
                "opt_step","epoch","batch_id",
                "loss_total","loss_param",
                "CFD_metric",
                "mse_noise","mse_x0","mse_v",
                "cfd_raw","w_t_mean","lambda_warm","lambda_cfd_eff","r_eff",
                "cfd_fro_pred","cfd_fro_shrunk","cfd_fro_smooth","cfd_ema_dist",
                "lr","mvn_loss", "mvn_mode", "mvn_trace_val","raw_mvn", "energy_loss"
            ]
            self.csv_writer = csv.DictWriter(self.csv_f, fieldnames=self.csv_fields)
            if need_header:
                self.csv_writer.writeheader()
                self.csv_f.flush()
    
        # accumulators for epoch-averages
        epoch_sums = {k: 0.0 for k in [
            "loss_total","loss_param","CFD_metric","mse_noise","mse_x0","mse_v",
            "cfd_raw","w_t_mean","lambda_warm","lambda_cfd_eff",
            "cfd_fro_pred","cfd_fro_shrunk","cfd_fro_smooth","cfd_ema_dist","mvn_loss", "mvn_trace_val", "raw_mvn", "energy_loss"
        ]}
        epoch_counts = 0
    
        def _csv_val(d, k):
            v = d.get(k, None)
            if v is None:
                return ""
            try:
                vv = float(v)
                if np.isnan(vv) or np.isinf(vv):
                    return ""
                return vv
            except Exception:
                return ""
    
        train_losses = []
        use_config = params.get('use_config', False)
        for batch_id, x in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
    
            total_loss,loss_ddpm,loss_energy, comps = self.batch_loss(x)
    
            # (optional) attention rollout snapshot
            if (self.epoch in getattr(self, "attn_epoch", [])) and batch_id == 0:
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
    
                # EMA update
                if hasattr(self, 'update_ema'):
                    self.update_ema()
    
                train_losses.append(loss.item())
    
                # Scheduler + lr
                cur_lr = None
                if getattr(self, "use_scheduler", False):
                    self.scheduler.step()
                    try:
                        cur_lr = float(self.scheduler.get_last_lr()[0])
                    except Exception:
                        cur_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
    
                # effective relative weight r_eff = (lambda_eff * cfd_raw) / mse_noise
                mse = comps.get("mse_noise", None)
                cfd = comps.get("cfd_raw", None)
                lam_eff = comps.get("lambda_cfd_eff", None)
                if (mse is not None) and (cfd is not None) and (lam_eff is not None):
                    try:
                        r_eff = (lam_eff * cfd) / mse if mse > 0 else ""
                    except Exception:
                        r_eff = ""
                else:
                    r_eff = ""
    
                self.opt_step += 1
    
                # CSV row
                row = {
                    "opt_step": int(self.opt_step),
                    "epoch": int(self.epoch),
                    "batch_id": int(batch_id),
                    "loss_total": float(loss.item()),
                    "loss_param": _csv_val(comps, "loss_param"),
                    "CFD_metric": _csv_val(comps, "CFD_metric"),
                    "mse_noise": _csv_val(comps, "mse_noise"),
                    "mse_x0":    _csv_val(comps, "mse_x0"),
                    "mse_v":     _csv_val(comps, "mse_v"),
                    "cfd_raw": _csv_val(comps, "cfd_raw"),
                    "w_t_mean": _csv_val(comps, "w_t_mean"),
                    "lambda_warm": _csv_val(comps, "lambda_warm"),
                    "lambda_cfd_eff": _csv_val(comps, "lambda_cfd_eff"),
                    "r_eff": r_eff if r_eff == "" else float(r_eff),
                    "cfd_fro_pred":   _csv_val(comps, "cfd_fro_pred"),
                    "cfd_fro_shrunk": _csv_val(comps, "cfd_fro_shrunk"),
                    "cfd_fro_smooth": _csv_val(comps, "cfd_fro_smooth"),
                    "cfd_ema_dist":   _csv_val(comps, "cfd_ema_dist"),
                    "lr": "" if cur_lr is None else cur_lr,
                    "mvn_loss": _csv_val(comps, "mvn_loss"),
                    "mvn_mode":comps.get("mvn_mode", ""),
                    "mvn_trace_val":  _csv_val(comps, "mvn_trace_val"),
                    "raw_mvn": _csv_val(comps, "raw_mvn"),
                    "energy_loss": _csv_val(comps, "energy_loss")
                    }
                self.csv_writer.writerow(row)
                self.csv_f.flush()
    
                # epoch sums
                for k in epoch_sums.keys():
                    v = _csv_val(comps, k)
                    if v != "":
                        epoch_sums[k] += float(v)
                epoch_counts += 1
    
            else:
                print(f"[warn] Unstable loss. Skipped backprop for epoch {self.epoch}, batch {batch_id}")
    
        # per-epoch arrays
        if train_losses:
            tl_mean = float(np.mean(train_losses))
            self.train_losses_epoch = np.append(self.train_losses_epoch, tl_mean)
            self.train_losses = np.concatenate([self.train_losses, np.array(train_losses)], axis=0)
    
            if getattr(self, "log", False):
                self.logger.add_scalar("train_losses_epoch", tl_mean, self.epoch)
                if getattr(self, "use_scheduler", False):
                    try:
                        self.logger.add_scalar("learning_rate_epoch", self.scheduler.get_last_lr()[0], self.epoch)
                    except Exception:
                        pass
    
        # (optional) write epoch-averages to a separate CSV
        if epoch_counts > 0:
            import os, csv
            epoch_csv = self.doc.get_file("training_log_epoch.csv")
            need_header = (not os.path.exists(epoch_csv)) or (os.path.getsize(epoch_csv) == 0)
            with open(epoch_csv, "a", newline="") as ef:
                ew = csv.DictWriter(ef, fieldnames=["epoch"] + list(epoch_sums.keys()))
                if need_header:
                    ew.writeheader()
                avg = {k: (epoch_sums[k] / epoch_counts) for k in epoch_sums}
                ew.writerow({"epoch": int(self.epoch), **avg})

    @torch.inference_mode()
    def validate_one_epoch(self):
        
        val_losses = np.array([])
        # iterate batch wise over input
        for batch_id, x in enumerate(self.val_loader):

            # calculate batch loss
            loss,_ = self.batch_loss(x)
            val_losses = np.append(val_losses, loss.item())
            # if self.log:
            #     self.logger.add_scalar("val_losses", val_losses[-1], self.epoch*self.n_trainbatches + batch_id)

        self.val_losses_epoch = np.append(self.val_losses_epoch, val_losses.mean())
        self.val_losses = np.concatenate([self.val_losses, val_losses], axis=0)
        if self.log:
            self.logger.add_scalar("val_losses_epoch", self.val_losses_epoch[-1], self.epoch)