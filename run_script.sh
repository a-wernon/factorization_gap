# Part A — full mask-ratio sweep, 500 prompts from gsm8k+humaneval
python run_gap.py --config configs/gap.yaml

# Part B — extracts features (~15–30 min on H200), then trains (fast)
python run_head_proxy.py --config configs/head.yaml
