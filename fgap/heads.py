"""Heads for the CP-vs-FF proxy on frozen drafter features.

All heads take a feature tensor h ∈ (B, n, H) — the drafter's final hidden states
on a fully-masked block — and produce a per-block log-likelihood of the block's
ground-truth token ids x_0 ∈ (B, n). The FF-native baseline is parameter-free; it
scores x_0 under the drafter's own factorized output. The CP head (shared-trunk,
rank r) factorizes the joint as a mixture of rank-1 products and shares the trunk
with the frozen target LM head.
"""

from __future__ import annotations

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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_future = num_future
        self.rank = rank
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


class FFTrainedHead(SharedTrunkCPHead):
    """A rank-1 shared-trunk head — learnable per-position scaling of features
    before the LM trunk, with a trivial single-component mixture. Serves as
    the "FF + some extra params trained on the same data" sanity baseline."""

    def __init__(self, hidden_size: int, num_future: int, lm_head_weight: torch.Tensor) -> None:
        super().__init__(hidden_size, num_future, rank=1, lm_head_weight=lm_head_weight)
