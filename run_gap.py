"""R1 Part A entry point: estimate the factorization gap.

Usage:
    python run_gap.py --config configs/gap.yaml
    python run_gap.py --config configs/gap.yaml --smoke    # 50 prompts, fast

Outputs under runs/gap/<run_name>_<timestamp>/:
    run.log, metrics.jsonl, summary.json, gap_curve.png, tb/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fgap.data import generate_prompt_blocks
from fgap.gap import GapRecord, estimate_gap_for_block
from fgap.models import load_models, target_context_features
from fgap.utils import (
    bootstrap_mean_ci,
    configure_logger,
    dtype_from_str,
    dump_json,
    dump_jsonl,
    env_threads,
    load_yaml,
    make_run_dir,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true", help="Override num_prompts=50.")
    p.add_argument("--overrides", nargs="*", default=[], help="k.v=... extra yaml overrides")
    return p.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for o in overrides:
        k, v = o.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d[p]
        # best-effort type coercion via yaml
        import yaml as _yaml

        d[parts[-1]] = _yaml.safe_load(v)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    if args.smoke:
        cfg["data"]["num_prompts"] = 50
        cfg["run_name"] = cfg["run_name"] + "_smoke"

    env_threads()
    set_seed(cfg["data"]["seed"])

    run_dir = make_run_dir(cfg["output"]["root"], cfg["run_name"])
    configure_logger(run_dir)
    dump_json(run_dir / "config.json", cfg)

    tb = SummaryWriter(str(run_dir / "tb")) if cfg["output"].get("tb", True) else None

    # ---- load models -----------------------------------------------------
    dtype = dtype_from_str(cfg["models"]["dtype"])
    device = cfg["models"]["device"]
    m = load_models(
        target_name=cfg["models"]["target"],
        drafter_name=cfg["models"]["drafter"],
        dtype=dtype,
        device=device,
    )

    if m.block_size != cfg["gap"]["block_size"]:
        logger.warning(
            f"config block_size={cfg['gap']['block_size']} but drafter was trained at "
            f"{m.block_size}; using drafter's value."
        )
    block_size = m.block_size

    # ---- prompts + x_0 blocks -------------------------------------------
    logger.info("Generating target responses and slicing into blocks...")
    blocks = generate_prompt_blocks(
        target=m.target,
        tokenizer=m.tokenizer,
        datasets=cfg["data"]["datasets"],
        num_prompts=cfg["data"]["num_prompts"],
        blocks_per_prompt=cfg["data"]["blocks_per_prompt"],
        block_size=block_size,
        max_new_tokens=cfg["data"]["gen_max_new_tokens"],
        enable_thinking=cfg["models"]["enable_thinking"],
        seed=cfg["data"]["seed"],
        device=m.device,
    )

    # ---- estimate gap ----------------------------------------------------
    mask_ratios = cfg["gap"]["mask_ratios"]
    fast = cfg["gap"]["fast_sequential"]
    gen = torch.Generator(device="cpu").manual_seed(cfg["data"]["seed"])

    records: list[GapRecord] = []
    for blk in tqdm(blocks, desc="gap"):
        ctx = blk.context_ids.to(m.device)
        x0 = blk.x0_ids.to(m.device)
        target_hidden, _ = target_context_features(m, ctx)
        block_records = estimate_gap_for_block(
            m=m,
            target_hidden=target_hidden,
            context_len=ctx.size(1),
            x0_ids=x0,
            mask_ratios=mask_ratios,
            fast_sequential=fast,
            source=blk.source,
            block_idx=blk.block_idx,
            torch_generator=gen,
        )
        records.extend(block_records)

    # ---- persist raw rows + bootstrap summary ---------------------------
    raw_rows = [r.__dict__ for r in records]
    if cfg["output"].get("save_per_block_rows", True):
        dump_jsonl(run_dir / "metrics.jsonl", raw_rows)

    by_ratio: dict[float, list[GapRecord]] = defaultdict(list)
    for r in records:
        by_ratio[r.mask_ratio].append(r)

    summary: dict = {"per_mask_ratio": {}}
    rng = np.random.default_rng(cfg["data"]["seed"])
    for ratio, recs in sorted(by_ratio.items()):
        gaps = np.array([r.per_pos_gap for r in recs])
        seqs = np.array([r.per_pos_sequential for r in recs])
        ones = np.array([r.per_pos_one_step for r in recs])
        gap_mean, gap_lo, gap_hi = bootstrap_mean_ci(
            gaps, cfg["gap"]["bootstrap_resamples"], rng=rng
        )
        seq_mean, _, _ = bootstrap_mean_ci(seqs, cfg["gap"]["bootstrap_resamples"], rng=rng)
        one_mean, _, _ = bootstrap_mean_ci(ones, cfg["gap"]["bootstrap_resamples"], rng=rng)
        summary["per_mask_ratio"][f"{ratio:.2f}"] = {
            "n": len(recs),
            "gap_mean": gap_mean,
            "gap_ci_lo": gap_lo,
            "gap_ci_hi": gap_hi,
            "sequential_mean": seq_mean,
            "one_step_mean": one_mean,
        }
        if tb is not None:
            step = int(round(ratio * 100))
            tb.add_scalar("gap/mean_nats_per_pos", gap_mean, step)
            tb.add_scalar("gap/ci_lo", gap_lo, step)
            tb.add_scalar("gap/ci_hi", gap_hi, step)
            tb.add_scalar("cll/sequential_per_pos", seq_mean, step)
            tb.add_scalar("cll/one_step_per_pos", one_mean, step)

    # ---- kill criterion check -------------------------------------------
    target_ratio = cfg["kill_criterion"]["mask_ratio_target"]
    thr = cfg["kill_criterion"]["nats_threshold"]
    key = f"{target_ratio:.2f}"
    target_gap = summary["per_mask_ratio"].get(key, {}).get("gap_mean", float("nan"))
    decision = "KILL" if target_gap < thr else "GO"
    summary["kill_criterion"] = {
        "mask_ratio_target": target_ratio,
        "nats_threshold": thr,
        "observed_gap": target_gap,
        "decision": decision,
    }

    dump_json(run_dir / "summary.json", summary)
    logger.info(json.dumps(summary, indent=2))

    # ---- plot -----------------------------------------------------------
    ratios = sorted(by_ratio.keys())
    means = [summary["per_mask_ratio"][f"{r:.2f}"]["gap_mean"] for r in ratios]
    los = [summary["per_mask_ratio"][f"{r:.2f}"]["gap_ci_lo"] for r in ratios]
    his = [summary["per_mask_ratio"][f"{r:.2f}"]["gap_ci_hi"] for r in ratios]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(ratios, means, marker="o", label="gap (nats/pos)")
    ax.fill_between(ratios, los, his, alpha=0.2, label="95% CI")
    ax.axhline(thr, color="red", ls="--", lw=1, label=f"kill threshold ({thr})")
    ax.set_xlabel("mask ratio")
    ax.set_ylabel("per-position gap (nats)")
    ax.set_title(f"Factorization gap — {cfg['models']['drafter']}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "gap_curve.png", dpi=150)

    print("\n" + "=" * 60)
    print(f"DECISION: {decision}   (gap@m={target_ratio}: {target_gap:.4f} nats/pos)")
    print(f"Outputs: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
