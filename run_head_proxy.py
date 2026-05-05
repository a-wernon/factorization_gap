"""R1 Part B (v2) entry point: CP rank sweep + MLP FF control on frozen drafter features.

Extends the original Part B to sweep multiple CP ranks in a single run AND
include a stronger factorized baseline (2-layer position-wise MLP + frozen
LM trunk), so the CP-specific contribution can be separated from
head-capacity effects.

Heads compared (per run):
    ff_native          — frozen LM head, parameter-free.
    ff_trained_r1      — rank-1 shared-trunk CP (learnable per-position scale).
    ff_mlp             — 2-layer position-wise MLP + residual + frozen trunk. (optional)
    cp_r{r}            — rank-r shared-trunk CP.               (one per entry in ranks)

Usage:
    # Full run — builds feature cache if missing, then trains all heads.
    python run_head_proxy.py --config configs/head.yaml
    # Smoke test — 100 blocks, 2 epochs.
    python run_head_proxy.py --config configs/head.yaml --smoke
    # Iterate on heads/ranks without re-extracting features:
    python run_head_proxy.py --config configs/head.yaml --no-extract

Outputs under runs/head_proxy/<run_name>_<timestamp>/:
    run.log, config.json, summary.json, tb/
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fgap.features import (
    build_feature_cache,
    cache_path,
    load_feature_cache,
    save_feature_cache,
)
from fgap.heads import (
    FFGainMLPHead,
    FFMLPHead,
    FFNativeScorer,
    FFTrainedHead,
    SharedTrunkCPHead,
)
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
    datasets_cfg = cfg["data"].get("datasets")
    if datasets_cfg is None:
        datasets_cfg = [cfg["data"]["dataset"]]
    datasets_cfg = list(datasets_cfg)
    fc = build_feature_cache(
        m=m,
        datasets=datasets_cfg,
        num_prompts=cfg["data"]["num_prompts"],
        blocks_per_prompt=cfg["data"]["blocks_per_prompt"],
        gen_max_new_tokens=cfg["data"]["gen_max_new_tokens"],
        enable_thinking=cfg["models"]["enable_thinking"],
        seed=cfg["data"]["seed"],
    )
    save_feature_cache(fc, path)
    return path


def _load_lm_head_weight(cfg: dict) -> torch.Tensor:
    """Load only target.lm_head.weight (V, H) to avoid re-hosting the full model."""
    from transformers import AutoModelForCausalLM

    dtype = dtype_from_str(cfg["models"]["dtype"])
    logger.info(f"Loading target lm_head weight from {cfg['models']['target']}")
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["target"], dtype=dtype, device_map=cfg["models"]["device"]
    )
    w = mdl.lm_head.weight.detach().clone()
    del mdl
    torch.cuda.empty_cache()
    return w


def _resolve_ranks(t_cfg: dict) -> list[int]:
    """Accept either `ranks: [..]` (new) or `rank: 8` (old) in the config."""
    if "ranks" in t_cfg and t_cfg["ranks"]:
        return [int(r) for r in t_cfg["ranks"]]
    if "rank" in t_cfg:
        return [int(t_cfg["rank"])]
    raise KeyError("training config must set either `ranks` (list) or `rank` (int)")


def _resolve_aux_lambdas(t_cfg: dict) -> list[float]:
    """Accept `aux_lambdas: [..]` list; default to [0.0] for backward compat."""
    if "aux_lambdas" in t_cfg and t_cfg["aux_lambdas"] is not None:
        return [float(x) for x in t_cfg["aux_lambdas"]]
    if "aux_lambda" in t_cfg:
        return [float(t_cfg["aux_lambda"])]
    return [0.0]


def _cp_head_name(rank: int, aux_lambda: float, single_lambda: bool) -> str:
    """Canonical head name. When only one λ is swept, drop the λ suffix for
    continuity with older run naming."""
    if single_lambda:
        return f"cp_r{rank}"
    # format lambda compactly without trailing zeros: 0.0 -> 0, 0.01 -> 0.01, 0.1 -> 0.1
    s = f"{aux_lambda:g}".rstrip("0").rstrip(".") or "0"
    return f"cp_r{rank}_l{s}"


def _build_heads(
    cfg: dict,
    hidden_size: int,
    num_future: int,
    lm_head_weight: torch.Tensor,
    device: torch.device,
) -> "collections.OrderedDict[str, torch.nn.Module]":
    t = cfg["training"]
    heads: collections.OrderedDict[str, torch.nn.Module] = collections.OrderedDict()

    if t.get("include_ff_r1", True):
        heads["ff_trained_r1"] = FFTrainedHead(
            hidden_size=hidden_size,
            num_future=num_future,
            lm_head_weight=lm_head_weight,
        ).to(device)

    if t.get("include_ff_mlp", False):
        heads["ff_mlp"] = FFMLPHead(
            hidden_size=hidden_size,
            num_future=num_future,
            lm_head_weight=lm_head_weight,
            mlp_hidden=int(t.get("mlp_hidden", 128)),
        ).to(device)

    if t.get("include_ff_gain_mlp", False):
        heads["ff_gain_mlp"] = FFGainMLPHead(
            hidden_size=hidden_size,
            num_future=num_future,
            lm_head_weight=lm_head_weight,
            mlp_hidden=int(t.get("mlp_hidden", 128)),
        ).to(device)

    ranks = _resolve_ranks(t)
    lambdas = _resolve_aux_lambdas(t)
    single_lambda = len(lambdas) == 1
    for r in ranks:
        for lam in lambdas:
            name = _cp_head_name(r, lam, single_lambda)
            heads[name] = SharedTrunkCPHead(
                hidden_size=hidden_size,
                num_future=num_future,
                rank=r,
                lm_head_weight=lm_head_weight,
                aux_lambda=lam,
            ).to(device)

    return heads


def _count_params(heads) -> dict[str, int]:
    return {name: sum(p.numel() for p in h.parameters()) for name, h in heads.items()}


def _make_optimizers(heads, lr: float, weight_decay: float) -> dict:
    return {
        name: torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=weight_decay)
        for name, h in heads.items()
    }


def _evaluate_all(
    heads: "collections.OrderedDict[str, torch.nn.Module]",
    scorer: FFNativeScorer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Return (per-block NLL per head, gate-entropy stats per head).

    Gate stats are reported for every head; for non-mixture heads (FF
    controls and rank-1) the entries are zero so downstream code can
    iterate uniformly.

    Gate stats are averaged across batches weighted by batch size. For
    mean_pi we average with equal weight on the batch-averaged pi at
    each step — this gives the overall mixture usage on the val set.
    """
    totals: dict[str, float] = {"ff_native": 0.0}
    for name in heads:
        totals[name] = 0.0

    # Gate diagnostics — accumulate mean_pi weighted by batch size.
    gate_accum: dict[str, dict] = {}
    for name, head in heads.items():
        r = getattr(head, "rank", 0)
        if r and r > 1:
            gate_accum[name] = {
                "H_per_sum": 0.0,
                "mean_pi_sum": torch.zeros(r, device=device, dtype=torch.float32),
                "batches": 0,
                "H_max": math.log(r),
            }

    n = 0
    for h in heads.values():
        h.eval()
    with torch.inference_mode():
        for feats_b, tgts_b in loader:
            feats_b = feats_b.to(device, non_blocking=True)
            tgts_b = tgts_b.to(device, non_blocking=True)
            bsz = feats_b.size(0)
            totals["ff_native"] += scorer.nll(feats_b, tgts_b).sum().item()
            for name, h in heads.items():
                totals[name] += h.nll(feats_b, tgts_b).sum().item()
                if name in gate_accum:
                    stats = h.gate_stats(feats_b)
                    gate_accum[name]["H_per_sum"] += stats["H_per_mean"] * bsz
                    gate_accum[name]["mean_pi_sum"] += (
                        torch.tensor(stats["mean_pi"], device=device) * bsz
                    )
                    gate_accum[name]["batches"] += bsz
            n += bsz

    per_block_nll = {k: v / n for k, v in totals.items()}

    entropy: dict[str, dict] = {}
    for name, head in heads.items():
        if name in gate_accum:
            a = gate_accum[name]
            H_per = a["H_per_sum"] / max(a["batches"], 1)
            mean_pi = (a["mean_pi_sum"] / max(a["batches"], 1)).cpu().tolist()
            H_batch = float(
                -sum(
                    p * math.log(max(p, 1e-40)) for p in mean_pi
                )
            )
            H_max = a["H_max"]
            entropy[name] = {
                "H_per_mean": H_per,
                "H_per_mean_norm": H_per / H_max if H_max > 0 else 0.0,
                "H_batch": H_batch,
                "H_batch_norm": H_batch / H_max if H_max > 0 else 0.0,
                "H_max": H_max,
                "mean_pi": mean_pi,
            }
        else:
            entropy[name] = {
                "H_per_mean": 0.0,
                "H_per_mean_norm": 0.0,
                "H_batch": 0.0,
                "H_batch_norm": 0.0,
                "H_max": 0.0,
                "mean_pi": [],
            }
    return per_block_nll, entropy


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

    feats = fc.features.float()
    tgts = fc.targets.long()
    dataset = TensorDataset(feats, tgts)

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

    heads = _build_heads(
        cfg,
        hidden_size=fc.hidden_size,
        num_future=fc.block_size,
        lm_head_weight=lm_head_weight,
        device=device,
    )
    scorer = FFNativeScorer(lm_head_weight=lm_head_weight)

    optimizers = _make_optimizers(
        heads,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    param_counts = _count_params(heads)
    logger.info("Heads + param counts (excl. frozen lm_head):")
    for name, n_p in param_counts.items():
        logger.info(f"  {name:<18s} params={n_p:,}")

    # ---- pre-training evaluation -----------------------------------------
    pre, pre_ent = _evaluate_all(heads, scorer, val_loader, device)
    logger.info(
        "[pre] val per-block NLL: "
        + "  ".join(f"{k}={v:.4f}" for k, v in pre.items())
    )

    # ---- training --------------------------------------------------------
    gstep = 0
    log_every = int(cfg["training"]["log_every"])
    for epoch in range(cfg["training"]["epochs"]):
        for h in heads.values():
            h.train()
        running_nll = {name: 0.0 for name in heads}
        running_aux = {name: 0.0 for name in heads}
        n = 0
        for feats_b, tgts_b in tqdm(train_loader, desc=f"ep{epoch}"):
            feats_b = feats_b.to(device, non_blocking=True)
            tgts_b = tgts_b.to(device, non_blocking=True)
            bsz = feats_b.size(0)
            for name, h in heads.items():
                loss_nll = h.nll(feats_b, tgts_b).mean()
                lam = float(getattr(h, "aux_lambda", 0.0))
                if lam > 0.0:
                    loss_aux = h.aux_loss(feats_b)
                    loss = loss_nll + lam * loss_aux
                    running_aux[name] += float(loss_aux.item()) * bsz
                else:
                    loss = loss_nll
                opt = optimizers[name]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running_nll[name] += float(loss_nll.item()) * bsz
            n += bsz
            if tb is not None and gstep % log_every == 0:
                for name in heads:
                    tb.add_scalar(f"train/{name}_nll", running_nll[name] / n, gstep)
                    if float(getattr(heads[name], "aux_lambda", 0.0)) > 0.0:
                        tb.add_scalar(
                            f"train/{name}_aux", running_aux[name] / n, gstep
                        )
            gstep += 1

        val, val_ent = _evaluate_all(heads, scorer, val_loader, device)
        # Succinct train/val line; entropy only for heads with mixture.
        train_parts = [f"{k}={running_nll[k]/n:.3f}" for k in heads]
        val_parts = [f"{k}={v:.3f}" for k, v in val.items()]
        ent_parts = [
            f"{k}:H_batch={val_ent[k]['H_batch_norm']:.2f}"
            for k in heads
            if val_ent.get(k, {}).get("H_max", 0.0) > 0.0
        ]
        logger.info(
            f"[ep{epoch}] train: "
            + " ".join(train_parts)
            + "  ||  val: "
            + " ".join(val_parts)
            + (("  ||  gate(norm): " + " ".join(ent_parts)) if ent_parts else "")
        )
        if tb is not None:
            for name, v in val.items():
                tb.add_scalar(f"val/{name}_nll", v, epoch)
            for name, ent in val_ent.items():
                if ent["H_max"] > 0.0:
                    tb.add_scalar(f"val/{name}_H_per_norm", ent["H_per_mean_norm"], epoch)
                    tb.add_scalar(f"val/{name}_H_batch_norm", ent["H_batch_norm"], epoch)

    # ---- report ----------------------------------------------------------
    final, final_ent = _evaluate_all(heads, scorer, val_loader, device)
    ff_native = final["ff_native"]
    ff_r1 = final.get("ff_trained_r1")
    ff_mlp = final.get("ff_mlp")
    ff_gain_mlp = final.get("ff_gain_mlp")

    improvements_vs_native = {
        name: (ff_native - v) / ff_native * 100.0
        for name, v in final.items()
        if name != "ff_native"
    }
    improvements_vs_ff_r1 = (
        {
            name: (ff_r1 - v) / ff_r1 * 100.0
            for name, v in final.items()
            if name not in ("ff_native", "ff_trained_r1")
        }
        if ff_r1 is not None
        else None
    )
    improvements_vs_ff_mlp = (
        {
            name: (ff_mlp - v) / ff_mlp * 100.0
            for name, v in final.items()
            if name not in ("ff_native", "ff_trained_r1", "ff_mlp")
        }
        if ff_mlp is not None
        else None
    )
    improvements_vs_ff_gain_mlp = (
        {
            name: (ff_gain_mlp - v) / ff_gain_mlp * 100.0
            for name, v in final.items()
            if name not in (
                "ff_native",
                "ff_trained_r1",
                "ff_mlp",
                "ff_gain_mlp",
            )
        }
        if ff_gain_mlp is not None
        else None
    )

    cp_final = {k: v for k, v in final.items() if k.startswith("cp_r")}
    best_cp_name = min(cp_final, key=cp_final.get) if cp_final else None
    best_cp_pct_vs_native = (
        improvements_vs_native[best_cp_name] if best_cp_name else None
    )

    min_pct = float(cfg["kill_criterion"]["min_improvement_pct"])
    decision = (
        "KILL"
        if (best_cp_pct_vs_native is None or best_cp_pct_vs_native < min_pct)
        else "GO"
    )

    summary = {
        "per_block_nll": {k: float(v) for k, v in final.items()},
        "param_counts": param_counts,
        "aux_lambdas_by_head": {
            name: float(getattr(h, "aux_lambda", 0.0))
            for name, h in heads.items()
        },
        "gate_entropy": {
            name: {k: v for k, v in ent.items() if k != "mean_pi"}
            for name, ent in final_ent.items()
        },
        "mean_pi": {name: ent["mean_pi"] for name, ent in final_ent.items()},
        "improvement_pct": {
            "vs_ff_native": improvements_vs_native,
            "vs_ff_trained_r1": improvements_vs_ff_r1,
            "vs_ff_mlp": improvements_vs_ff_mlp,
            "vs_ff_gain_mlp": improvements_vs_ff_gain_mlp,
        },
        "best_cp": (
            {
                "name": best_cp_name,
                "nll": float(cp_final[best_cp_name]),
                "pct_vs_native": best_cp_pct_vs_native,
                "pct_vs_ff_trained_r1": (
                    improvements_vs_ff_r1.get(best_cp_name)
                    if improvements_vs_ff_r1
                    else None
                ),
                "pct_vs_ff_mlp": (
                    improvements_vs_ff_mlp.get(best_cp_name)
                    if improvements_vs_ff_mlp
                    else None
                ),
                "pct_vs_ff_gain_mlp": (
                    improvements_vs_ff_gain_mlp.get(best_cp_name)
                    if improvements_vs_ff_gain_mlp
                    else None
                ),
                "gate_H_batch_norm": final_ent.get(best_cp_name, {}).get(
                    "H_batch_norm"
                ),
                "gate_H_per_norm": final_ent.get(best_cp_name, {}).get(
                    "H_per_mean_norm"
                ),
                "gate_aux_lambda": float(
                    getattr(heads[best_cp_name], "aux_lambda", 0.0)
                )
                if best_cp_name
                else None,
            }
            if best_cp_name
            else None
        ),
        "kill_criterion": {
            "min_improvement_pct": min_pct,
            "observed_best_pct_vs_native": best_cp_pct_vs_native,
            "decision": decision,
        },
        "n_train_blocks": n_train,
        "n_val_blocks": n_val,
        "ranks": _resolve_ranks(cfg["training"]),
        "aux_lambdas": _resolve_aux_lambdas(cfg["training"]),
        "block_size": fc.block_size,
    }
    dump_json(run_dir / "summary.json", summary)
    logger.info(json.dumps(summary, indent=2))

    print("\n" + "=" * 68)
    if best_cp_name:
        print(
            f"DECISION: {decision}   "
            f"(best CP = {best_cp_name} @ NLL {cp_final[best_cp_name]:.3f};  "
            f"{best_cp_pct_vs_native:.2f}% vs ff_native, threshold {min_pct:.1f}%)"
        )
    else:
        print(f"DECISION: {decision}  (no CP heads configured)")
    if improvements_vs_ff_r1 and best_cp_name in (improvements_vs_ff_r1 or {}):
        print(
            f"          {best_cp_name} vs ff_trained_r1:  "
            f"{improvements_vs_ff_r1[best_cp_name]:+.2f}%"
        )
    if improvements_vs_ff_mlp and best_cp_name in (improvements_vs_ff_mlp or {}):
        print(
            f"          {best_cp_name} vs ff_mlp:         "
            f"{improvements_vs_ff_mlp[best_cp_name]:+.2f}%"
        )
    if improvements_vs_ff_gain_mlp and best_cp_name in (improvements_vs_ff_gain_mlp or {}):
        print(
            f"          {best_cp_name} vs ff_gain_mlp:    "
            f"{improvements_vs_ff_gain_mlp[best_cp_name]:+.2f}%   <-- honest joint-structure test"
        )
    if ff_mlp is not None:
        print(
            f"          ff_mlp       vs ff_native:     "
            f"{improvements_vs_native['ff_mlp']:+.2f}%  (capacity control)"
        )
    if ff_gain_mlp is not None:
        print(
            f"          ff_gain_mlp  vs ff_native:     "
            f"{improvements_vs_native['ff_gain_mlp']:+.2f}%  (gain+MLP control)"
        )
    # Gate-collapse diagnostic for best CP.
    if best_cp_name and final_ent.get(best_cp_name, {}).get("H_max", 0.0) > 0.0:
        ent = final_ent[best_cp_name]
        collapse_msg = (
            "COLLAPSED" if ent["H_batch_norm"] < 0.3 else
            ("partial" if ent["H_batch_norm"] < 0.7 else "healthy")
        )
        print(
            f"          {best_cp_name} gate: "
            f"H_batch_norm={ent['H_batch_norm']:.3f}  "
            f"H_per_norm={ent['H_per_mean_norm']:.3f}  ({collapse_msg})"
        )
    print(f"Outputs: {run_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
