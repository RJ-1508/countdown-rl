import ast 
import operator
import re
from collections import Counter
import configs

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}

def safe_eval(expr):
    return _ev(ast.parse(expr, mode="eval").body)

def _ev(node):
    if (isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.operand))
    raise ValueError(f"disallowed node: {ast.dump(node)}")

def used_numbers(expr):
    tree = ast.parse(expr, mode="eval")
    return [int(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]

def last_answer(completion):
    m = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
    return m[-1].strip() if m else None

def parse_prompt(prompt):
    target = int(re.search(r"target (\d+)", prompt).group(1))
    nums = re.search(r"numbers ([\d, ]+?)\.", prompt).group(1)
    return [int(x) for x in nums.split(",")], target

def reward_components(prompt, completion):
    response = last_answer(completion)
    if response is None:
        return {"has_answer": False, "valid": False,
                "correct": False, "reward": 0.0}
    numbers, target = parse_prompt(prompt)
    try:
        val = safe_eval(response)
    except (ValueError, ZeroDivisionError, SyntaxError, RecursionError):
        return {"has_answer": True, "valid": False,
        "correct": False, "reward": configs.REWARD_FORMAT_BONUS}
    valid = Counter(used_numbers(response)) <= Counter(numbers)
    correct = valid and abs(val - target) < 1e-6
    reward_val = (configs.REWARD_CORRECT + configs.REWARD_FORMAT_BONUS
                  if correct else configs.REWARD_FORMAT_BONUS)
    return {"has_answer": True, "valid": valid,
            "correct": correct, "reward": reward_val}



def reward(prompt, completion):
    return reward_components(prompt, completion)["reward"]