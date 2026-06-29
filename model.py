import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import configs

SYSTEM = "You are a careful arithmetic solver"

def load_model_and_tokenizer(adapter_dir=None, dtype = torch.float16):
    tok = AutoTokenizer.from_pretrained(configs.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(configs.MODEL_NAME, dtype=dtype).to("cuda")
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    return model, tok

@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens = None, **gen_kwargs):
    model.eval()
    max_new_tokens = max_new_tokens or configs.MAX_NEW_TOKENS
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt = True)
    inputs = tok(text, return_tensors ="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens = max_new_tokens, do_sample = False, pad_token_id = tok.eos_token_id, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)