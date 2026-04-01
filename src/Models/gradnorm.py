"""
GradNorm balancer — self-contained, no external dependencies beyond PyTorch.
Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing
in Deep Multitask Networks", ICML 2018.  arXiv:1711.02257
"""
import torch
import torch.nn as nn


class GradNormBalancer(nn.Module):

    def __init__(self, num_tasks: int, alpha: float = 1.5,
                 lr: float = 1e-3, device: str = 'cuda'):
        super().__init__()
        # log-weights keep w_i > 0 without clamping
        self.log_w     = nn.Parameter(torch.zeros(num_tasks, device=device))
        self.w_optim   = torch.optim.Adam([self.log_w], lr=lr)
        self.alpha     = alpha
        self.num_tasks = num_tasks
        self.L0        = None    # initial losses, set on first call
        self.device    = device

    @property
    def weights(self) -> torch.Tensor:
        """
        Softmax in log-space then rescale so sum = num_tasks
        (GradNorm paper renormalisation convention).
        """
        return torch.softmax(self.log_w, dim=0) * self.num_tasks

    def step(
        self,
        losses:        list,
        objectives:    list,
        shared_params: list,
        all_params:    list | None = None,
    ) -> tuple[torch.Tensor, dict, list]:
        """
        Parameters
        ----------
        losses        : list of scalar Tensors, one per task
        objectives    : list of str names, same order (for logging)
        shared_params : Parameters at the last shared layer —
                        GradNorm loss is computed here only (cheap)
        all_params    : full network Parameters — used to extract
                        per-objective flat gradient vectors for
                        _compute_gradient_diagnostics. If None,
                        full-network diagnostics are skipped and
                        an empty list is returned.

        Returns
        -------
        L_total      : weighted total loss with w detached —
                       call .backward() on this in _step_with_gradnorm
        info         : dict of scalars for CSV logging
        full_grads   : list of flat gradient Tensors, one per task,
                       over the full network (empty list if all_params
                       is None). Pass to _compute_gradient_diagnostics.
        """
        # ── Store L0 on first call ────────────────────────────────────────
        if self.L0 is None:
            self.L0 = torch.tensor(
                [l.detach().item() for l in losses],
                device=self.device
            )

        # w is connected to log_w — used for G_i and L_grad
        w = self.weights                             # (T,)  sum = T

        # w_detach breaks the graph so L_total.backward() does not
        # fight with the in-place Adam update on log_w (Bug 1 fix)
        w_detach = w.detach()
        L_total  = sum(w_detach[i] * losses[i] for i in range(self.num_tasks))

        # ── Per-task gradient norms at shared layer (for GradNorm loss) ──
        G = []
        for i in range(self.num_tasks):
            grads_shared = torch.autograd.grad(
                w[i] * losses[i],
                shared_params,
                retain_graph=True,
                create_graph=True,   # needed so L_grad can flow to log_w
                allow_unused=True,
            )
            grads_shared = [g for g in grads_shared if g is not None]
            if grads_shared:
                G.append(torch.norm(torch.cat([g.flatten() for g in grads_shared])))
            else:
                G.append(torch.tensor(0.0, device=self.device))

        G_bar = torch.stack(G).mean().detach()       # constant target

        # ── Relative inverse training rates ──────────────────────────────
        L_hat = torch.stack([
            losses[i].detach() / (self.L0[i] + 1e-8)
            for i in range(self.num_tasks)
        ])
        r = L_hat / (L_hat.mean() + 1e-8)            # (T,)

        # ── GradNorm loss — updates log_w only ───────────────────────────
        G_target = (G_bar * r ** self.alpha).detach()
        L_grad   = sum(torch.abs(G[i] - G_target[i])
                       for i in range(self.num_tasks))

        self.w_optim.zero_grad()
        L_grad.backward(retain_graph=True)   # retain: L_total.backward() follows
        self.w_optim.step()

        # ── Full-network gradient vectors per task (for diagnostics) ──────
        # Computed after L_grad.backward() while graph still alive.
        # create_graph=False — we only need the vectors, not higher-order grads.
        full_grads = []
        if all_params is not None:
            for i in range(self.num_tasks):
                retain = (i < self.num_tasks - 1)   # free graph on last task
                grads_full = torch.autograd.grad(
                    losses[i],
                    all_params,
                    retain_graph=retain,
                    create_graph=False,
                    allow_unused=True,
                )
                flat = torch.cat([
                    g.flatten() for g in grads_full if g is not None
                ])
                full_grads.append(flat.detach())

        # ── Logging info ──────────────────────────────────────────────────
        w_final = self.weights.detach()
        info = {"gradnorm_L_grad": float(L_grad.detach().item())}
        for i, name in enumerate(objectives):
            info[f"gradnorm_w_{name}"]       = float(w_final[i].item())
            info[f"gradnorm_G_{name}"]       = float(G[i].detach().item())
            info[f"gradnorm_Gtarget_{name}"] = float(G_target[i].item())
            info[f"gradnorm_r_{name}"]       = float(r[i].item())

        return L_total, info, full_grads