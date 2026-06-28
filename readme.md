# countdown-rl

Fine tuning a small open LLM to solve Countdown (a verifiable arithmetic reasoning task) — supervised LoRA baseline, then check improvement against a from-scratch REINFORCE->GRPO RL loop in raw PyTorch.

Data: 3 000 train / 500 eval puzzles (3–4 numbers, 1–25, target ≤ 1 000), generated deterministically (seed 7); splits disjoint by construction.
Reward: AST-based safe evaluator (no `eval`); whitelists int/float literals + − × ÷ and unary ±, rejects everything else including `True`/`False`.
Status: phase 1 complete — data gen and reward function done; training loop is next.
