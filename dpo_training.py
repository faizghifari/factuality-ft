import os
import argparse

from unsloth import FastLanguageModel, PatchDPOTrainer

PatchDPOTrainer()

import torch

from transformers import TrainingArguments
from datasets import load_from_disk
from accelerate import Accelerator
from trl import DPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir", help="path to the output directory", required=True, type=str
)
parser.add_argument(
    "--dataset_dir",
    help="path to the dataset directory",
    default=None,
    type=str,
)
args = parser.parse_args()

dataset = load_from_disk(args.dataset_dir)
output_dir = args.output_dir
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="huggyllama/llama-7b",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    max_seq_length=max_seq_length,
)

accelerator = Accelerator()
dpo_trainer = accelerator.prepare(
    DPOTrainer(
        model=model,
        ref_model=None,
        args=TrainingArguments(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            warmup_ratio=0.01,
            save_strategy="epoch",
            num_train_epochs=2,
            learning_rate=2e-6,
            fp16=True,
            logging_steps=1,
            seed=42,
            output_dir=output_dir,
        ),
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_prompt_length=512,
    )
)


print("Training...")
train_result = dpo_trainer.train()
metrics = train_result.metrics
dpo_trainer.log_metrics("train", metrics)
dpo_trainer.save_metrics("train", metrics)
dpo_trainer.save_state()

print("Saving last checkpoint of the model...")
os.makedirs(output_dir, exist_ok=True)

# Free memory for merging weights
del dpo_trainer
torch.cuda.empty_cache()

model.save_pretrained_merged(
    f"{output_dir}-lora",
    tokenizer,
    save_method="lora",
)

# model.save_pretrained_merged(
#     output_dir,
#     tokenizer,
#     save_method="merged_16bit",
# )
