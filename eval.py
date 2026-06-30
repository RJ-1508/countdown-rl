import random
from collections import defaultdict
from tqdm.auto import tqdm

import configs
import data
from model import generate


def bootstrap_ci(flags, n_boot=None, lo=2.5, hi=97.5):
    n_boot = n_boot or configs.BOOTSTRAP_N
    n = len(flags)

    rng = random.Random(configs.SEED)
    means = []
    for _ in range(n_boot):
        resample = rng.choices(flags, k=n)
        means.append(sum(resample) / n)
    means.sort()
    return means[int(lo / 100 * n_boot)], means[int(hi / 100 * n_boot)]


def evaluate(model, tok, dataset, reward_fn, **gen_kwargs):
    reward_values = []
    solved_flags = []
    by_difficulty = defaultdict(list)
    n = len(dataset)
    for record in tqdm(dataset, desc="eval"):
        prompt = data.build_prompt(record["numbers"], record["target"])
        completion = generate(model, tok, prompt, **gen_kwargs)
        reward = reward_fn(prompt, completion)
        solved = reward >= configs.REWARD_CORRECT
        reward_values.append(reward)
        solved_flags.append(solved)
        by_difficulty[str(record["difficulty"])].append(solved)

    return {"n":n, "solve_rate":sum(solved_flags) / n, "mean_reward":sum(reward_values) / n, "solve_rate_ci": bootstrap_ci(solved_flags), "by_difficulty": {k: sum(v) / len(v) for k, v in by_difficulty.items()}}