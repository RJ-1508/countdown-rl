import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

import configs
import data
from model import load_model_and_tokenizer, SYSTEM

def build_sft_example(record, tok):
    user = data.build_prompt(record["numbers"], record["target"])
    assistant = (f"Let me solve it step by step.\n"
                f"<answer>{record['solution_expr']}</answer>")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]

    full_chat = tok.apply_chat_template(messages, tokenize=False)
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    input_ids = tok(full_chat)["input_ids"]
    prompt_len = len(tok(prompt)["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
    

def main():
    model, tok = load_model_and_tokenizer(dtype=torch.float32)

    lora_cfg = LoraConfig(
        r=configs.LORA_R,
        lora_alpha=configs.LORA_ALPHA,
        lora_dropout=configs.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    records = data.load("train")
    examples = [build_sft_example(r, tok) for r in records]
    ds = Dataset.from_list(examples)

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id = -100)

    args = TrainingArguments(  output_dir=str(configs.OUT_DIR / "sft"),
        per_device_train_batch_size=configs.SFT_BATCH,
        gradient_accumulation_steps=configs.GRAD_ACCUM,
        num_train_epochs=configs.SFT_EPOCHS,
        learning_rate=configs.SFT_LR,
        fp16=True,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=configs.SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(str(configs.ADAPTER_DIR))
    print(f"saved adapter -> {configs.ADAPTER_DIR}")
    
if __name__ == "__main__":
    main()
