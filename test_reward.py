import data
import reward as R

NUMS   = [25, 8, 3, 7]
TARGET = (25 - 8) * 3 + 7        # = 58, a reachable target
PROMPT = data.build_prompt(NUMS, TARGET)

CORRECT = 1.1   # REWARD_CORRECT + REWARD_FORMAT_BONUS
BONUS   = 0.1   # tag present, but not valid+correct
ZERO    = 0.0   # no parseable <answer>


def test_correct():
    c = "Reasoning here. <answer>(25 - 8) * 3 + 7</answer>"
    assert R.reward(PROMPT, c) == CORRECT

def test_valid_but_wrong():
    c = "<answer>25 + 8 + 3</answer>"          # allowed numbers, != 58
    assert R.reward(PROMPT, c) == BONUS

def test_number_reuse():
    c = "<answer>25 + 25 + 8</answer>"         # 25 provided once, used twice
    assert R.reward(PROMPT, c) == BONUS

def test_uses_outside_number():
    c = "<answer>25 + 8 + 99</answer>"         # 99 not provided
    assert R.reward(PROMPT, c) == BONUS

def test_malformed_expr():
    c = "<answer>(25 - )</answer>"             # tag present, won't parse
    assert R.reward(PROMPT, c) == BONUS

def test_missing_tag():
    c = "I think it's 58 but I forgot the tag."
    assert R.reward(PROMPT, c) == ZERO

def test_uses_last_answer():
    c = "<answer>1 + 1</answer> wait — <answer>(25 - 8) * 3 + 7</answer>"
    assert R.reward(PROMPT, c) == CORRECT

def test_components_shape():
    comp = R.reward_components(PROMPT, "<answer>(25 - 8) * 3 + 7</answer>")
    assert comp["has_answer"] and comp["valid"] and comp["correct"]
    assert comp["reward"] == CORRECT
