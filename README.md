# R1 — Factorization gap in DFlash-style drafters

Phase 0, experiment R1 from `../journal/extension_plan.md`.

Measures whether a KV-injected diffusion drafter (DFlash Qwen3-8B-b16) has
measurable **factorization-gap** headroom for a joint output head.

Two scripts, each with an explicit kill criterion:

- **Part A — gap estimation** (`run_gap.py`): sequential-vs-one-step conditional
  log-likelihood gap on the same drafter backbone, swept over mask ratios.
  **Kill** if per-position gap at mask_ratio=1.0 is < 0.05 nats.
- **Part B — CP vs FF proxy** (`run_head_proxy.py`): train a rank-8 shared-trunk
  CP head on frozen drafter features, compare per-block NLL against the drafter's
  native factorized head. **Kill** if CP improvement over FF_native is < 2%.

## Setup

```bash
cd factorization_gap
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
# dflash is imported via sys.path; no install needed (see fgap/models.py).

# HF login for gated Qwen3 (if your account needs it)
huggingface-cli login
```

## How to run

### Smoke test (first, before full runs)

```bash
# ~10 min on 1 H200 — sanity-checks the whole pipeline
python run_gap.py --config configs/gap.yaml --smoke

# ~10 min on 1 H200 — extracts ~400 feature blocks, trains 2 epochs
python run_head_proxy.py --config configs/head.yaml --smoke
```

Expected: gap curve monotonically increases with mask ratio; val NLL decreases
per epoch; both scripts print an explicit `DECISION: GO|KILL` line at the end.

### Full R1

```bash
# Part A — full mask-ratio sweep, 500 prompts from gsm8k+humaneval
python run_gap.py --config configs/gap.yaml

# Part B — extracts features (~15–30 min on H200), then trains (fast)
python run_head_proxy.py --config configs/head.yaml
```

Wall-time estimate on 1× H200: Part A ≈ 1–2 h, Part B ≈ 30–60 min. Peak VRAM
~25 GB (Qwen3-8B target + drafter + small KV).

### Reruns / iterating on the head

After Part B has extracted features once, the cache is reused:

```bash
python run_head_proxy.py --config configs/head.yaml --no-extract
# or vary hyperparams inline:
python run_head_proxy.py --config configs/head.yaml --no-extract  # edit rank/lr in yaml
```

## Outputs

Each run creates `runs/gap/<name>_<ts>/` or `runs/head_proxy/<name>_<ts>/` with:

- `config.json` — resolved config actually used
- `run.log` — loguru log
- `summary.json` — headline metrics + decision
- `metrics.jsonl` (gap only) — per-block rows for re-analysis
- `gap_curve.png` (gap only) — CoDD Fig. 1-style plot
- `tb/` — TensorBoard scalars

View: `tensorboard --logdir runs/`

## Kill-criterion quick reference

| Experiment | Metric (val) | Kill if |
|---|---|---|
| Part A gap | per-pos gap at mask_ratio=1.0 | < 0.05 nats |
| Part B proxy | (FF_native − CP) / FF_native | < 2% |

Both firing → Direction 1 pivot (CoDD-style end-generation with CP). Either
alone warrants a second look before pivoting.

## Notes / assumptions

- Block size fixed at 16 (drafter's trained size; the plan's n∈{4,8,16,32}
  sweep would require retraining drafters — out of scope for R1).
- Shared-trunk CP only. Full CP at Qwen3's 152k vocab is infeasible.
- `x_0` blocks are target-greedy-generated, not dataset ground truth. This
  matches the spec-decode setting (drafter is supposed to predict the target's
  continuations). If you want to also measure gap against reference solutions,
  swap the generator out in `fgap/data.py`.
- `dflash` is imported from `../dflash/` via `sys.path` — no install needed.
