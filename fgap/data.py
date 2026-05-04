"""Prompt loading and target-generated block sampling.

Re-uses `dflash.benchmark.DATASETS` for prompt formatting; the block-ground-truth
`x_0` comes from sampling the target model at temperature 0 on each prompt, then
slicing the response into blocks of size n. This matches the distribution the
drafter sees at spec-decode time (DFlash paper §4.1).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DFLASH_DIR = _REPO_ROOT / "dflash"
if str(_DFLASH_DIR) not in sys.path:
    sys.path.insert(0, str(_DFLASH_DIR))

from dflash.benchmark import DATASETS, load_and_process_dataset  # noqa: E402


@dataclass
class PromptBlock:
    """A single (prompt, block-offset) example.

    context_ids: (1, L_ctx)  — input_ids + generated response up to block_start
    x0_ids:      (1, n)      — ground-truth block = response[block_start:+n]
    """

    context_ids: torch.LongTensor
    x0_ids: torch.LongTensor
    source: str
    block_idx: int


def _build_raw_prompts(datasets: list[str]) -> list[tuple[str, str]]:
    """Return list of (dataset_name, single_turn_text)."""
    rows: list[tuple[str, str]] = []
    for name in datasets:
        if name not in DATASETS:
            raise ValueError(f"unknown dataset {name}")
        data = load_and_process_dataset(name)
        for row in data:
            for turn in row["turns"]:
                rows.append((name, turn))
    return rows


def _apply_chat_template(tokenizer, text: str, enable_thinking: bool) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


@torch.inference_mode()
def generate_prompt_blocks(
    target: torch.nn.Module,
    tokenizer,
    datasets: list[str],
    num_prompts: int,
    blocks_per_prompt: int,
    block_size: int,
    max_new_tokens: int,
    enable_thinking: bool,
    seed: int,
    device: torch.device,
) -> list[PromptBlock]:
    """Generate target responses and slice into PromptBlock examples.

    Strategy: greedy (temp=0) generation; keeps the distribution deterministic so
    the same x_0 is produced across drafter evaluations.
    """
    raw = _build_raw_prompts(datasets)
    rng = random.Random(seed)
    rng.shuffle(raw)
    raw = raw[:num_prompts]

    results: list[PromptBlock] = []
    for i, (name, text) in enumerate(raw):
        input_ids = _apply_chat_template(tokenizer, text, enable_thinking).to(device)
        out = target.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        full = out  # (1, L_in + L_gen)
        prompt_len = input_ids.size(1)
        gen = full[:, prompt_len:]  # (1, L_gen)
        L_gen = gen.size(1)
        max_blocks = L_gen // block_size
        take = min(blocks_per_prompt, max_blocks)
        for b in range(take):
            start = b * block_size
            ctx = full[:, : prompt_len + start].clone()
            x0 = gen[:, start : start + block_size].clone()
            results.append(
                PromptBlock(
                    context_ids=ctx, x0_ids=x0, source=name, block_idx=b
                )
            )
        if (i + 1) % 50 == 0:
            logger.info(f"[gen] prompt {i+1}/{len(raw)}  blocks_total={len(results)}")
    logger.info(
        f"[gen] done: {len(results)} blocks from {len(raw)} prompts ({datasets})"
    )
    return results


def iter_batches(items: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
