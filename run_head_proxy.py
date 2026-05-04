"""R1 Part B entry point: CP-r8 vs FF proxy on frozen drafter features.

Usage:
    # First run:  extracts features (slow), trains, reports.
    python run_head_proxy.py --config configs/head.yaml
    # Smoke test: ~100 blocks, 2 epochs.
    python run_head_proxy.py --config configs/head.yaml --smoke
    # If feature cache already exists, training is fast and doesn't touch GPU models.
    python run_head_proxy.py --config configs/head.yaml --no-extract

Outputs under runs/head_proxy/<run_name>_<timestamp>/:
    run.log, config.json, summary.json, tb/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fgap.features import (
    build_feature_cache,
    cache_path,
    load_feature_cache,
    save_feature_cache,
)
from fgap.heads import FFNativeScorer, FFTrainedHead, SharedTrunkCPHead
from fgap.models import load_models
from fgap.utils import (
    configure_logger,
    dtype_from_str,
    dump_json,
    env_threads,
    load_yaml,
    make_run_dir,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true", help="100 blocks, 2 epochs.")
    p.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip feature extraction; cache must exist.",
    )
    return p.parse_args()


def _build_cache_or_load(cfg: dict, run_dir: Path) -> Path:
    cache_dir = Path(cfg["features"]["cache_dir"])
    path = cache_path(cache_dir, cfg["features"]["cache_name"])
    if path.exists():
        logger.info(f"Using existing feature cache: {path}")
        return path
    logger.info(f"Feature cache not found, building → {path}")
    dtype = dtype_from_str(cfg["models"]["dtype"])
    m = load_models(
        target_name=cfg["models"]["target"],
        drafter_name=cfg["models"]["drafter"],
        dtype=dtype,
        device=cfg["models"]["device"],
    )
    fc = build_feature_cache(
        m=m,
        datasets=[cfg["data"]["dataset"]],
        num_prompts=cfg["data"]["num_prompts"],
        blocks_per_prompt=cfg["data"]["blocks_per_prompt"],
        gen_max_new_tokens=cfg["data"]["gen_max_new_tokens"],
        enable_thinking=cfg["models"]["enable_thinking"],
        seed=cfg["data"]["seed"],
    )
    save_feature_cache(fc, path)
    # We deliberately keep the models resident — FF-native scoring needs
    # target.lm_head. Save a pointer to its weights for the training phase.
    # (Re-load from scratch in the training phase if running --no-extract.)
    return path


def _load_lm_head_weight(cfg: dict) -> torch.Tensor:
    """Load only target.lm_head.weight (V, H) to avoid re-hosting the full model."""
    from transformers import AutoModelForCausalLM

    dtype = dtype_from_str(cfg["models"]["dtype"])
    logger.info(f"Loading target lm_head weight from {cfg['models']['target']}")
    # `device_map="meta"` would skip weights; we want the lm_head weights
    # materialized on GPU once.  Loading the full model then grabbing the head
    # is the simplest reliable path (memory is fine on H200).
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["target"], dtype=dtype, device_map=cfg["models"]["device"]
    )
    w = mdl.lm_head.weight.detach().clone()
    del mdl
    torch.cuda.empty_cache()
    return w


def _evaluate(head, scorer, loader, device) -> tuple[float, float, float]:
    """Return (cp_nll, ff_native_nll, ff_trained_nll) — per-block means."""
    head["cp"].eval()
    head["ff_trained"].eval()
    cp_tot = 0.0
    ff_native_tot = 0.0
    ff_trained_tot = 0.0
    n = 0
    with torch.inference_mode():
        for feats, tgts in loader:
            feats = feats.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            cp_tot += head["cp"].nll(feats, tgts).sum().item()
            ff_trained_tot += head["ff_trained"].nll(feats, tgts).sum().item()
            ff_native_tot += scorer.nll(feats, tgts).sum().item()
            n += feats.size(0)
    return cp_tot / n, ff_native_tot / n, ff_trained_tot / n


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.smoke:
        cfg["data"]["num_prompts"] = 100
        cfg["training"]["epochs"] = 2
        cfg["features"]["cache_name"] = cfg["features"]["cache_name"] + "_smoke"
        cfg["run_name"] = cfg["run_name"] + "_smoke"

    env_threads()
    set_seed(cfg["data"]["seed"])

    run_dir = make_run_dir(cfg["output"]["root"], cfg["run_name"])
    configure_logger(run_dir)
    dump_json(run_dir / "config.json", cfg)
    tb = SummaryWriter(str(run_dir / "tb")) if cfg["output"].get("tb", True) else None

    # ---- features --------------------------------------------------------
    if args.no_extract:
        path = cache_path(Path(cfg["features"]["cache_dir"]), cfg["features"]["cache_name"])
        if not path.exists():
            raise FileNotFoundError(f"--no-extract set but cache missing: {path}")
    else:
        path = _build_cache_or_load(cfg, run_dir)

    fc = load_feature_cache(path)
    logger.info(
        f"Loaded features: N={fc.features.size(0)} n={fc.block_size} H={fc.hidden_size}"
    )

    # sanity: feature dtype (saved as whatever drafter produced — bf16)
    feats = fc.features.float()         # head lives on GPU but training data small enough to keep fp32 on cpu
    tgts = fc.targets.long()
    dataset = TensorDataset(feats, tgts)

    # 80/20 split (deterministic)
    n_total = len(dataset)
    n_val = int(round(cfg["training"]["val_frac"] * n_total))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(cfg["data"]["seed"])
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    logger.info(f"train blocks={n_train} val blocks={n_val}")

    # ---- lm_head weight + heads ------------------------------------------
    device = torch.device(cfg["models"]["device"])
    lm_head_weight = _load_lm_head_weight(cfg).to(device)

    cp = SharedTrunkCPHead(
        hidden_size=fc.hidden_size,
        num_future=fc.block_size,
        rank=cfg["training"]["rank"],
        lm_head_weight=lm_head_weight,
    ).to(device)
    ff_trained = FFTrainedHead(
        hidden_size=fc.hidden_size,
        num_future=fc.block_size,
        lm_head_weight=lm_head_weight,
    ).to(device)
    scorer = FFNativeScorer(lm_head_weight=lm_head_weight)

    heads = {"cp": cp, "ff_trained": ff_trained}
    opt_cp = torch.optim.AdamW(
        cp.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    opt_ff = torch.optim.AdamW(
        ff_trained.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ---- pre-training evaluation -----------------------------------------
    pre_cp, pre_ff_native, pre_ff_trained = _evaluate(heads, scorer, val_loader, device)
    logger.info(
        f"[pre] val per-block NLL: cp={pre_cp:.4f}  "
        f"ff_native={pre_ff_native:.4f}  ff_trained={pre_ff_trained:.4f}"
    )

    # ---- training --------------------------------------------------------
    gstep = 0
    for epoch in range(cfg["training"]["epochs"]):
        cp.train()
        ff_trained.train()
        ep_cp = 0.0
        ep_ff = 0.0
        n = 0
        for feats, tgts in tqdm(train_loader, desc=f"ep{epoch}"):
            feats = feats.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            # CP step
            loss_cp = cp.nll(feats, tgts).mean()
            opt_cp.zero_grad(set_to_none=True)
            loss_cp.backward()
            opt_cp.step()
            # FF-trained step
            loss_ff = ff_trained.nll(feats, tgts).mean()
            opt_ff.zero_grad(set_to_none=True)
            loss_ff.backward()
            opt_ff.step()

            bsz = feats.size(0)
            ep_cp += float(loss_cp.item()) * bsz
            ep_ff += float(loss_ff.item()) * bsz
            n += bsz
            if tb is not None and gstep % cfg["training"]["log_every"] == 0:
                tb.add_scalar("train/cp_nll", float(loss_cp.item()), gstep)
                tb.add_scalar("train/ff_trained_nll", float(loss_ff.item()), gstep)
            gstep += 1

        cp_val, ff_native_val, ff_trained_val = _evaluate(
            heads, scorer, val_loader, device
        )
        logger.info(
            f"[ep{epoch}] train cp={ep_cp/n:.4f} ff_trained={ep_ff/n:.4f} | "
            f"val cp={cp_val:.4f} ff_native={ff_native_val:.4f} ff_trained={ff_trained_val:.4f}"
        )
        if tb is not None:
            tb.add_scalar("val/cp_nll", cp_val, epoch)
            tb.add_scalar("val/ff_native_nll", ff_native_val, epoch)
            tb.add_scalar("val/ff_trained_nll", ff_trained_val, epoch)

    # ---- report ----------------------------------------------------------
    cp_final, ff_native_final, ff_trained_final = _evaluate(
        heads, scorer, val_loader, device
    )
    pct_vs_native = (ff_native_final - cp_final) / ff_native_final * 100.0
    pct_vs_trained = (ff_trained_final - cp_final) / ff_trained_final * 100.0
    min_pct = cfg["kill_criterion"]["min_improvement_pct"]
    decision = "KILL" if pct_vs_native < min_pct else "GO"

    summary = {
        "per_block_nll": {
            "cp_r{}".format(cfg["training"]["rank"]): cp_final,
            "ff_native": ff_native_final,
            "ff_trained_r1": ff_trained_final,
        },
        "improvement_pct": {
            "cp_vs_ff_native": pct_vs_native,
            "cp_vs_ff_trained": pct_vs_trained,
        },
        "kill_criterion": {
            "min_improvement_pct": min_pct,
            "observed_pct_vs_native": pct_vs_native,
            "decision": decision,
        },
        "n_train_blocks": n_train,
        "n_val_blocks": n_val,
        "rank": cfg["training"]["rank"],
        "block_size": fc.block_size,
    }
    dump_json(run_dir / "summary.json", summary)
    logger.info(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(
        f"DECISION: {decision}   "
        f"(CP vs FF_native: {pct_vs_native:.2f}%, threshold {min_pct:.1f}%)"
    )
    print(f"Outputs: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
