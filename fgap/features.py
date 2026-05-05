"""Build + load frozen-drafter feature datasets for head-proxy training.

For each (prompt, block) sample:
  - x_0 := target-greedy-generated block tokens of size n
  - x_t := fully-masked block (all positions → mask_token_id)
  - h   := drafter(target_hidden=…, noise_embedding=embed(x_t), …).hidden
Saved as a single .pt file with tensors
  features: (N, n, H)  bf16
  targets:  (N, n)     int64
  sources:  list[str]  (one per block)

Files are reusable across multiple head-training runs; the filename encodes the
drafter/target/prompt-count so stale caches don't get silently reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
from transformers import DynamicCache

from .data import generate_prompt_blocks
from .models import LoadedModels, target_context_features


@dataclass
class FeatureCache:
    features: torch.Tensor          # (N, n, H) bf16/float16 on cpu
    targets: torch.Tensor           # (N, n) int64 on cpu
    sources: list[str]
    block_size: int
    hidden_size: int
    drafter_name: str
    target_name: str


def cache_path(cache_dir: Path, cache_name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_name}.pt"


@torch.inference_mode()
def build_feature_cache(
    m: LoadedModels,
    datasets: list[str],
    num_prompts: int,
    blocks_per_prompt: int,
    gen_max_new_tokens: int,
    enable_thinking: bool,
    seed: int,
) -> FeatureCache:
    """Generate target responses, fully-mask each block, and record drafter hiddens."""
    blocks = generate_prompt_blocks(
        target=m.target,
        tokenizer=m.tokenizer,
        datasets=datasets,
        num_prompts=num_prompts,
        blocks_per_prompt=blocks_per_prompt,
        block_size=m.block_size,
        max_new_tokens=gen_max_new_tokens,
        enable_thinking=enable_thinking,
        seed=seed,
        device=m.device,
    )

    feats_list: list[torch.Tensor] = []
    tgts_list: list[torch.Tensor] = []
    sources: list[str] = []

    for blk in tqdm(blocks, desc="features"):
        ctx = blk.context_ids.to(m.device)
        x0 = blk.x0_ids.to(m.device)
        # fully-masked x_t
        x_t = torch.full_like(x0, m.mask_token_id)

        target_hidden, _ = target_context_features(m, ctx)
        noise_embedding = m.target.get_input_embeddings()(x_t)
        pos = torch.arange(
            0, ctx.size(1) + m.block_size, device=m.device
        ).unsqueeze(0)
        drafter_hidden = m.drafter(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=pos,
            past_key_values=DynamicCache(),
            use_cache=True,
            is_causal=False,
        )  # (1, n, H)

        feats_list.append(drafter_hidden.detach().to("cpu"))
        tgts_list.append(x0.detach().to("cpu"))
        sources.append(blk.source)

    if not feats_list:
        raise RuntimeError("No blocks produced — check num_prompts / gen_max_new_tokens.")

    features = torch.cat(feats_list, dim=0)   # (N, n, H)
    targets = torch.cat(tgts_list, dim=0)     # (N, n)

    logger.info(
        f"Feature cache: features {tuple(features.shape)} {features.dtype}, "
        f"targets {tuple(targets.shape)} {targets.dtype}"
    )
    return FeatureCache(
        features=features,
        targets=targets,
        sources=sources,
        block_size=m.block_size,
        hidden_size=m.hidden_size,
        drafter_name=m.drafter.config._name_or_path,
        target_name=m.target.config._name_or_path,
    )


def save_feature_cache(fc: FeatureCache, path: Path) -> None:
    torch.save(
        {
            "features": fc.features,
            "targets": fc.targets,
            "sources": fc.sources,
            "block_size": fc.block_size,
            "hidden_size": fc.hidden_size,
            "drafter_name": fc.drafter_name,
            "target_name": fc.target_name,
        },
        path,
    )
    logger.info(f"Saved feature cache → {path}")


def load_feature_cache(path: Path) -> FeatureCache:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return FeatureCache(
        features=blob["features"],
        targets=blob["targets"],
        sources=list(blob["sources"]),
        block_size=int(blob["block_size"]),
        hidden_size=int(blob["hidden_size"]),
        drafter_name=str(blob["drafter_name"]),
        target_name=str(blob["target_name"]),
    )
