"""Load Qwen3 target + DFlash drafter; utilities for feature extraction."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Make dflash.model importable without installing the dflash package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DFLASH_DIR = _REPO_ROOT / "dflash"
if str(_DFLASH_DIR) not in sys.path:
    sys.path.insert(0, str(_DFLASH_DIR))

from dflash.model import extract_context_feature  # noqa: E402


@dataclass
class LoadedModels:
    target: torch.nn.Module
    drafter: torch.nn.Module
    tokenizer: "AutoTokenizer"
    device: torch.device
    dtype: torch.dtype
    block_size: int
    mask_token_id: int
    target_layer_ids: list[int]
    hidden_size: int


def load_models(
    target_name: str,
    drafter_name: str,
    dtype: torch.dtype,
    device: str = "cuda",
) -> LoadedModels:
    logger.info(f"Loading target: {target_name}")
    target = AutoModelForCausalLM.from_pretrained(
        target_name, dtype=dtype, device_map=device
    ).eval()
    logger.info(f"Loading drafter: {drafter_name}")
    drafter = AutoModel.from_pretrained(
        drafter_name, trust_remote_code=True, dtype=dtype, device_map=device
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(target_name)

    # Sanity checks on drafter-config state.
    block_size = int(getattr(drafter, "block_size"))
    mask_token_id = getattr(drafter, "mask_token_id", None)
    if mask_token_id is None:
        raise RuntimeError(
            "drafter has no mask_token_id set; check config.dflash_config"
        )
    target_layer_ids = list(drafter.target_layer_ids)
    hidden_size = int(target.config.hidden_size)

    logger.info(
        f"block_size={block_size} mask_token_id={mask_token_id} "
        f"target_layer_ids={target_layer_ids} hidden_size={hidden_size}"
    )
    return LoadedModels(
        target=target,
        drafter=drafter,
        tokenizer=tokenizer,
        device=torch.device(device if torch.cuda.is_available() else "cpu"),
        dtype=dtype,
        block_size=block_size,
        mask_token_id=int(mask_token_id),
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
    )


@torch.inference_mode()
def target_context_features(
    m: LoadedModels,
    context_ids: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run target over `context_ids` and return (target_hidden, position_ids).

    target_hidden: (1, context_len, len(target_layer_ids) * H).
    position_ids: (1, context_len) aligned with context_ids.
    """
    assert context_ids.ndim == 2 and context_ids.size(0) == 1
    ctx_len = context_ids.size(1)
    pos = torch.arange(ctx_len, device=context_ids.device).unsqueeze(0)
    out = m.target(
        input_ids=context_ids,
        position_ids=pos,
        output_hidden_states=True,
        use_cache=False,
    )
    target_hidden = extract_context_feature(out.hidden_states, m.target_layer_ids)
    return target_hidden, pos


@torch.inference_mode()
def drafter_block_logits(
    m: LoadedModels,
    target_hidden: torch.Tensor,
    block_ids: torch.LongTensor,
    context_len: int,
) -> torch.Tensor:
    """Single drafter pass over a block of token ids → logits.

    target_hidden: (1, ctx, concat_H) — result of `target_context_features`.
    block_ids: (1, n) — possibly-masked token ids for the block.
    context_len: int — number of tokens already processed by the target
        (drafter needs its own KV for those positions).
    Returns logits (1, n, V).
    """
    n = block_ids.size(1)
    noise_embedding = m.target.get_input_embeddings()(block_ids)
    draft_cache = DynamicCache()

    # The drafter needs positional ids that line up with the target's ctx.
    # dflash_generate passes ids covering [draft_cache_len .. start+block_size);
    # here we do a single fresh call so draft_cache_len == 0 and we cover the
    # full (0..ctx+n) range, but only feed `n` noise embeddings.  So we give
    # positions only for the block portion: ctx..ctx+n.
    pos = torch.arange(
        0, context_len + n, device=block_ids.device
    ).unsqueeze(0)

    drafter_hidden = m.drafter(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=pos,
        past_key_values=draft_cache,
        use_cache=True,
        is_causal=False,
    )
    logits = m.target.lm_head(drafter_hidden)
    return logits  # (1, n, V)
