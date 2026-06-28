from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_files"
OUT_DIR = ROOT / "outputs"

SEED = 7

# data gen 
N_TRAIN = 3000
N_EVAL = 500
MIN_NUMS = 3
MAX_NUMS = 4 # diff: 3(easy), 4(hard)
NUM_LOW = 1
NUM_HIGH = 25
TARGET_MAX = 1000 

# output contract
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

# reward
REWARD_CORRECT = 1.0
REWARD_FORMAT_BONUS = 0.1

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 256