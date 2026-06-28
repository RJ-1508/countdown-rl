import json
import random

import configs

def _apply(a, b, op):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    return a // b

def _random_solution(rng, k):
    numbers = rng.sample(range(configs.NUM_LOW, configs.NUM_HIGH+1), k)
    items = [(n, str(n)) for n in numbers]
    rng.shuffle(items)
    while len(items) > 1:
        a_val, a_str = items.pop()
        b_val, b_str = items.pop()
        op = rng.choice(["+", "-", "*", "/"])
        if op == "/" and (b_val == 0 or a_val % b_val != 0):
            op = rng.choice(["+", "-", "*"])
        val = _apply(a_val, b_val, op)
        items.append((val, f"({a_str} {op} {b_str})"))
    value, expr = items[0]
    return numbers, value, expr

def build_prompt(numbers, target):
    nums = ", ".join(str(n) for n in numbers)
    return (f"Reach the target {target} using the numbers {nums}. "
            f"Each number may be used at most once. Allowed operators: + - * /. "
            f"Show brief reasoning, then put the final expression inside "
            f"<answer></answer> tags.")

def generate(n, rng):
    seen, out = set(), []
    while len(out) < n:
        k = rng.randint(configs.MIN_NUMS, configs.MAX_NUMS)
        numbers, value, expr = _random_solution(rng, k)
        if value <= 0 or value > configs.TARGET_MAX:
            continue
        key = (tuple(sorted(numbers)), value)
        if key in seen:
            continue
        seen.add(key)
        out.append({"numbers": numbers, "target": value, "solution_expr": expr, "difficulty": k})
    return out

def load(split):
    return json.loads((configs.DATA_DIR / f"{split}.json").read_text())

def main():
    rng = random.Random(configs.SEED)
    data = generate(configs.N_TRAIN + configs.N_EVAL, rng)
    rng.shuffle(data)
    train, evald = data[:configs.N_TRAIN], data[configs.N_TRAIN:]
    configs.DATA_DIR.mkdir(exist_ok=True)
    (configs.DATA_DIR / "train.json").write_text(json.dumps(train, indent=2))
    (configs.DATA_DIR / "eval.json").write_text(json.dumps(evald, indent=2))
    print(f"train={len(train)}  eval={len(evald)}")
    for name, split in [("train", train), ("eval", evald)]:
        easy = sum(1 for r in split if r["difficulty"] == 3)
        print(f"  {name}: easy(3#)={easy}  hard(4#)={len(split) - easy}")


if __name__ == "__main__":
    main()
