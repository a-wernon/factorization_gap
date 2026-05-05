"""Heads for the CP-vs-FF proxy on frozen drafter features.

All heads take a feature tensor h ∈ (B, n, H) — the drafter's final hidden states
on a fully-masked block — and produce a per-block log-likelihood of the block's
ground-truth token ids x_0 ∈ (B, n). The FF-native baseline is parameter-free; it
scores x_0 under the drafter's own factorized output. The CP head (shared-trunk,
rank r) factorizes the joint as a mixture of rank-1 products and shares the trunk
with the frozen target LM head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_log_softmax_over_vocab(logits: torch.Tensor) -> torch.Tensor:
    # cast to float32 for numerical stability in the softmax
    return F.log_softmax(logits.float(), dim=-1)


class FFNativeScorer:
    """Parameter-free. Scores targets under drafter's native factorized output."""

    def __init__(self, lm_head_weight: torch.Tensor) -> None:
        # lm_head_weight: (V, H), on whichever device it lives.
        self.lm_head_weight = lm_head_weight

    @torch.inference_mode()
    def nll(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return per-block NLL (B,) = -sum_i log p(x_i | h) under factorized p."""
        logits = F.linear(h.to(self.lm_head_weight.dtype), self.lm_head_weight)   # (B, n, V)
        log_p = _safe_log_softmax_over_vocab(logits)
        log_p_tgt = log_p.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, n)
        return -log_p_tgt.sum(dim=1)  # (B,)


class SharedTrunkCPHead(nn.Module):
    """Rank-r CP joint head sharing the V×H trunk with a frozen LM head.

    For each rank component α ∈ [r]:
        logits_α(i) = lm_head_weight @ (w_sh[i, α] ⊙ h_i)
        log p_α(x | h) = sum_i log softmax(logits_α(i))[x_i]
    Joint: log p(x|h) = logsumexp_α ( log π_α(h) + log p_α(x|h) )

    Params: w_sh ∈ (n, r, H) + gate: Linear(H, r). ≈ 16·r·H + H·r + r params.
    """

    def __init__(
        self,
        hidden_size: int,
        num_future: int,
        rank: int,
        lm_head_weight: torch.Tensor,   # (V, H) frozen
        aux_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_future = num_future
        self.rank = rank
        self.aux_lambda = float(aux_lambda)
        self.register_buffer("lm_head_weight", lm_head_weight.detach(), persistent=False)
        # init w_sh near 1 so early training matches the FF baseline path
        w = torch.ones(num_future, rank, hidden_size)
        w = w + 0.01 * torch.randn_like(w)
        self.w_sh = nn.Parameter(w)
        self.gate = nn.Linear(hidden_size, rank)

    def _log_prod_per_rank(
        self, h: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return (B, r) log-product-of-marginals per rank component."""
        B, n, H = h.shape
        r = self.rank
        # shape broadcasting: h (B,n,H) * w_sh (n,r,H) -> (B,n,r,H)
        h_expanded = h.unsqueeze(2) * self.w_sh.unsqueeze(0)  # (B,n,r,H)
        # logits over vocab via shared trunk
        # reshape to (B*n*r, H) for a single linear
        flat = h_expanded.reshape(B * n * r, H)
        logits = F.linear(flat.to(self.lm_head_weight.dtype), self.lm_head_weight)  # (B*n*r, V)
        log_p = _safe_log_softmax_over_vocab(logits)          # (B*n*r, V)
        # gather target
        tgt = targets.unsqueeze(-1).expand(B, n, r).reshape(B * n * r, 1)
        log_p_tgt = log_p.gather(1, tgt).squeeze(1)           # (B*n*r,)
        log_p_tgt = log_p_tgt.reshape(B, n, r)
        return log_p_tgt.sum(dim=1)                           # (B, r)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Log prior over ranks: (B, r). Used only by nll()."""
        pooled = h.mean(dim=1)                                 # (B, H)
        return F.log_softmax(self.gate(pooled), dim=-1)        # (B, r)

    def nll(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-block NLL (B,)."""
        log_prior = self.forward(h)                            # (B, r)
        log_prod = self._log_prod_per_rank(h, targets)         # (B, r)
        log_joint = torch.logsumexp(log_prior + log_prod, dim=-1)  # (B,)
        return -log_joint

    def aux_loss(self, h: torch.Tensor) -> torch.Tensor:
        """Load-balancing penalty on the batch-averaged mixture gate.

        Returns (max_H - H(mean_pi)) / max_H ∈ [0, 1]:
            0 when the batch-averaged gate is uniform across components,
            1 when fully collapsed to a single component.

        Encourages the optimizer to keep *all* components alive — otherwise
        an NLL-only training run typically collapses to whichever single
        rank wins the first few steps, and rank > 1 buys nothing.
        Rank-1 heads always return 0 (no mixture to balance).
        """
        if self.rank <= 1:
            return h.new_zeros(())
        log_pi = self.forward(h).float()                       # (B, r)
        B = h.size(0)
        log_mean_pi = torch.logsumexp(log_pi, dim=0) - math.log(B)   # (r,)
        mean_pi = log_mean_pi.exp()
        H_mean = -(mean_pi * log_mean_pi).sum()
        max_H = math.log(self.rank)
        return (max_H - H_mean) / max_H

    @torch.no_grad()
    def gate_stats(self, h: torch.Tensor) -> dict:
        """Diagnostics: gate-entropy statistics and mean-pi distribution.

        Returns:
            H_per_mean  — mean (over batch) of per-sample gate entropy.
            H_batch     — entropy of the batch-averaged gate distribution.
            H_max       — log(rank); the entropy of a uniform distribution.
            mean_pi     — list[r]; batch-averaged mixture weights.

        H_per small → each sample picks one component (peaky per-sample gate).
        H_batch small (with H_per also small) → the SAME component wins every
        sample, i.e. the mixture has collapsed. H_per small but H_batch large
        means per-sample specialization with overall balanced usage — healthy.
        """
        if self.rank <= 1:
            return {
                "H_per_mean": 0.0,
                "H_batch": 0.0,
                "H_max": 0.0,
                "mean_pi": [1.0] * self.rank,
            }
        log_pi = self.forward(h).float()                       # (B, r)
        pi = log_pi.exp()
        H_per_mean = float(-(pi * log_pi).sum(dim=-1).mean().item())
        mean_pi = pi.mean(dim=0)                               # (r,)
        H_batch = float(
            -(mean_pi * mean_pi.clamp_min(1e-40).log()).sum().item()
        )
        return {
            "H_per_mean": H_per_mean,
            "H_batch": H_batch,
            "H_max": math.log(self.rank),
            "mean_pi": mean_pi.cpu().tolist(),
        }


class FFTrainedHead(SharedTrunkCPHead):
    """A rank-1 shared-trunk head — learnable per-position scaling of features
    before the LM trunk, with a trivial single-component mixture. Serves as
    the "FF + some extra params trained on the same data" sanity baseline."""

    def __init__(
        self,
        hidden_size: int,
        num_future: int,
        lm_head_weight: torch.Tensor,
    ) -> None:
        super().__init__(
            hidden_size, num_future, rank=1, lm_head_weight=lm_head_weight
        )


class FFMLPHead(nn.Module):
    """Stronger factorized baseline: position-wise 2-layer MLP with residual,
    then the frozen LM trunk. Purely factorized — no cross-position structure
    — so the CP-vs-FFMLP comparison isolates *joint structure* from *head
    capacity*.

    Architecture:
        h2 = h + W_down( gelu( W_up(h) ) )    # same MLP applied per position
        logits_i = lm_head_weight @ h2_i

    Init: W_down and its bias are zero, so at step 0 the head computes
    exactly lm_head(h) — matching ff_native. Training then moves from that
    point. This keeps early-training dynamics comparable to the CP / rank-1
    heads (whose w_sh is init at 1 so they also start at ff_native).
    """

    def __init__(
        self,
        hidden_size: int,
        num_future: int,
        lm_head_weight: torch.Tensor,   # (V, H), frozen
        mlp_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_future = num_future
        self.mlp_hidden = mlp_hidden
        self.register_buffer("lm_head_weight", lm_head_weight.detach(), persistent=False)
        self.up = nn.Linear(hidden_size, mlp_hidden)
        self.down = nn.Linear(mlp_hidden, hidden_size)
        # Start at identity: delta = 0 -> h2 = h.
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def nll(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-block NLL. h: (B, n, H); targets: (B, n). Returns (B,)."""
        delta = self.down(F.gelu(self.up(h)))                      # (B, n, H)
        h2 = h + delta.to(h.dtype)
        logits = F.linear(h2.to(self.lm_head_weight.dtype), self.lm_head_weight)
        log_p = _safe_log_softmax_over_vocab(logits)               # (B, n, V)
        log_p_tgt = log_p.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, n)
        return -log_p_tgt.sum(dim=1)                               # (B,)

    # No gate / no mixture → aux and gate_stats are no-ops. Kept so the
    # training loop can call them uniformly across head types.
    aux_lambda = 0.0

    def aux_loss(self, h: torch.Tensor) -> torch.Tensor:
        return h.new_zeros(())

    def gate_stats(self, h: torch.Tensor) -> dict:
        return {"H_per_mean": 0.0, "H_batch": 0.0, "H_max": 0.0, "mean_pi": []}


class FFGainMLPHead(nn.Module):
    """Strong factorized control combining per-position gain + shared MLP.

    Matches CP's inductive bias MINUS the cross-position mixture. Specifically:
        - Per-position gain  w_i  ∈ R^H         (like FFTrainedHead)
        - Shared residual MLP  h + W_down(gelu(W_up(h)))   (like FFMLPHead)
        - Frozen LM trunk

        h' = h + W_down( gelu( W_up(h) ) )     # shared across positions
        h'' = h' ⊙ w_i                         # per-position multiplicative gain
        logits_i = W_LM @ h''_i

    Init: W_down zero-init so delta=0 at step 0, and w_i init near 1 so the
    gain is identity at step 0. Together: output = lm_head(h) = ff_native at
    step 0, matching the other heads' "start from identity" convention.

    This is the cleanest "strong FF" baseline: if CP still wins by a
    meaningful margin against this head, the win is *joint-structure*
    specific, not "per-position gain" or "extra capacity".
    """

    def __init__(
        self,
        hidden_size: int,
        num_future: int,
        lm_head_weight: torch.Tensor,
        mlp_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_future = num_future
        self.mlp_hidden = mlp_hidden
        self.register_buffer("lm_head_weight", lm_head_weight.detach(), persistent=False)
        self.up = nn.Linear(hidden_size, mlp_hidden)
        self.down = nn.Linear(mlp_hidden, hidden_size)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        # Per-position multiplicative gain, init at 1 + small noise (matches
        # FFTrainedHead / SharedTrunkCPHead.w_sh init convention).
        w = torch.ones(num_future, hidden_size)
        w = w + 0.02 * torch.randn_like(w)
        self.w = nn.Parameter(w)

    def nll(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-block NLL. h: (B, n, H); targets: (B, n). Returns (B,)."""
        delta = self.down(F.gelu(self.up(h)))                      # (B, n, H)
        h_mlp = h + delta.to(h.dtype)
        h2 = h_mlp * self.w.unsqueeze(0)                           # (B, n, H)
        logits = F.linear(h2.to(self.lm_head_weight.dtype), self.lm_head_weight)
        log_p = _safe_log_softmax_over_vocab(logits)
        log_p_tgt = log_p.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return -log_p_tgt.sum(dim=1)

    # No gate / no mixture → aux and gate_stats are no-ops.
    aux_lambda = 0.0

    def aux_loss(self, h: torch.Tensor) -> torch.Tensor:
        return h.new_zeros(())

    def gate_stats(self, h: torch.Tensor) -> dict:
        return {"H_per_mean": 0.0, "H_batch": 0.0, "H_max": 0.0, "mean_pi": []}
