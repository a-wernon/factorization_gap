"""Factorization-gap estimation.

For a given (context, x_0, mask_ratio), compute:
  - one-step CLL: log p_phi(x_0_masked | x_t) under the drafter's factorized output.
  - sequential CLL: log p_joint(x_0_masked | x_t) via any-order AR decoding from the
    *same* drafter backbone (revealing one ground-truth token at a time).

Mean(sequential - one-step) over many (prompt, block, ratio) triples is our estimate
of the misspecification gap L_gap (CoDD §3 / Fig. 1 Left), per masked position.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .models import LoadedModels


@dataclass
class GapRecord:
    source: str
    block_idx: int
    mask_ratio: float
    num_masked: int
    one_step_sum: float        # sum of log p over masked positions
    sequential_sum: float
    per_pos_one_step: float    # mean per masked position (nats)
    per_pos_sequential: float
    per_pos_gap: float


def _sample_mask(n: int, ratio: float, generator: torch.Generator) -> torch.Tensor:
    """Return a boolean mask of length n with round(ratio*n) True entries."""
    k = int(round(ratio * n))
    k = max(1, min(n, k))  # at least one masked position so CLL is defined
    perm = torch.randperm(n, generator=generator)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[perm[:k]] = True
    return mask


def _drafter_logits(
    m: LoadedModels,
    target_hidden: torch.Tensor,   # (B_ctx, L_ctx, concat_H)
    block_ids: torch.Tensor,       # (B, n)
    context_len: int,
) -> torch.Tensor:
    """Batched drafter pass. target_hidden's B_ctx must equal block_ids.B (or 1)."""
    B, n = block_ids.shape
    if target_hidden.size(0) == 1 and B > 1:
        target_hidden = target_hidden.expand(B, -1, -1)
    elif target_hidden.size(0) != B:
        raise ValueError(
            f"target_hidden batch {target_hidden.size(0)} vs block batch {B}"
        )
    noise_embedding = m.target.get_input_embeddings()(block_ids)
    pos = torch.arange(
        context_len, context_len + n, device=block_ids.device
    ).unsqueeze(0).expand(B, -1)
    drafter_hidden = m.drafter(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=pos,
        past_key_values=DynamicCache(),
        use_cache=True,
        is_causal=False,
    )
    return m.target.lm_head(drafter_hidden)  # (B, n, V)


@torch.inference_mode()
def estimate_gap_for_block(
    m: LoadedModels,
    target_hidden: torch.Tensor,      # (1, L_ctx, concat_H)
    context_len: int,
    x0_ids: torch.Tensor,             # (1, n)
    mask_ratios: list[float],
    *,
    fast_sequential: bool,
    source: str,
    block_idx: int,
    torch_generator: torch.Generator,
) -> list[GapRecord]:
    """Compute gap records for all mask ratios on one block.

    Uses a single mask per ratio (order_samples=1 in config). Repeat across
    blocks for variance reduction rather than across orders within a block.
    """
    n = x0_ids.size(1)
    device = x0_ids.device
    results: list[GapRecord] = []

    for ratio in mask_ratios:
        mask_cpu = _sample_mask(n, ratio, torch_generator)
        mask = mask_cpu.to(device)
        num_masked = int(mask.sum().item())
        masked_positions = mask.nonzero(as_tuple=False).squeeze(-1)  # (k,)

        # ---- One-step CLL (single drafter pass at x_t) -----------------
        x_t = x0_ids.clone()
        x_t[0, mask] = m.mask_token_id
        logits = _drafter_logits(m, target_hidden, x_t, context_len).float()
        log_p = F.log_softmax(logits, dim=-1)  # (1, n, V)
        one_step_logp_all = log_p[0].gather(
            1, x0_ids[0].unsqueeze(-1)
        ).squeeze(-1)  # (n,)
        one_step_sum = float(one_step_logp_all[mask].sum().item())

        # ---- Sequential CLL (any-order AR) -----------------------------
        # A single random reveal order over the masked positions.
        order = masked_positions[
            torch.randperm(num_masked, generator=torch_generator, device="cpu").to(
                device
            )
        ]  # (k,)

        if fast_sequential:
            seq_sum = _sequential_cll_batched(
                m, target_hidden, context_len, x_t, x0_ids, order
            )
        else:
            seq_sum = _sequential_cll_loop(
                m, target_hidden, context_len, x_t, x0_ids, order
            )

        per_pos_one = one_step_sum / num_masked
        per_pos_seq = seq_sum / num_masked
        results.append(
            GapRecord(
                source=source,
                block_idx=block_idx,
                mask_ratio=ratio,
                num_masked=num_masked,
                one_step_sum=one_step_sum,
                sequential_sum=seq_sum,
                per_pos_one_step=per_pos_one,
                per_pos_sequential=per_pos_seq,
                per_pos_gap=per_pos_seq - per_pos_one,
            )
        )

    return results


@torch.inference_mode()
def _sequential_cll_batched(
    m: LoadedModels,
    target_hidden: torch.Tensor,   # (1, L_ctx, concat_H)
    context_len: int,
    x_t_init: torch.Tensor,        # (1, n) — masked state
    x0_ids: torch.Tensor,          # (1, n) — ground truth
    order: torch.Tensor,           # (k,) positions to reveal in that order
) -> float:
    """Batched sequential CLL.

    We form k partial-reveal states x_t^{(0)}=x_t_init, x_t^{(1)}, ..., x_t^{(k-1)},
    where x_t^{(j)} has order[:j] revealed. A single batched drafter pass gives
    log p(x_0[order[j]] | x_t^{(j)}) for every j.
    """
    k = order.size(0)
    if k == 0:
        return 0.0
    n = x_t_init.size(1)
    device = x_t_init.device

    # Build batch of partial-reveal states.
    states = x_t_init.expand(k, -1).clone()          # (k, n)
    for j in range(1, k):
        # state j has positions order[:j] already revealed.
        reveal_positions = order[:j]
        states[j, reveal_positions] = x0_ids[0, reveal_positions]

    logits = _drafter_logits(m, target_hidden, states, context_len).float()
    log_p = F.log_softmax(logits, dim=-1)  # (k, n, V)

    # For each batch row j, pick position=order[j], target=x0[order[j]].
    batch_idx = torch.arange(k, device=device)
    token_idx = order
    target_tokens = x0_ids[0, order]
    picked = log_p[batch_idx, token_idx, target_tokens]  # (k,)
    return float(picked.sum().item())


@torch.inference_mode()
def _sequential_cll_loop(
    m: LoadedModels,
    target_hidden: torch.Tensor,
    context_len: int,
    x_t_init: torch.Tensor,
    x0_ids: torch.Tensor,
    order: torch.Tensor,
) -> float:
    """Reference (slow) loop implementation, kept for verification."""
    k = order.size(0)
    if k == 0:
        return 0.0
    x = x_t_init.clone()
    total = 0.0
    for j in range(k):
        logits = _drafter_logits(m, target_hidden, x, context_len).float()
        log_p = F.log_softmax(logits, dim=-1)  # (1, n, V)
        p = int(order[j].item())
        t = int(x0_ids[0, p].item())
        total += float(log_p[0, p, t].item())
        x[0, p] = x0_ids[0, p]
    return total
