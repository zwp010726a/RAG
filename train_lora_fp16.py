# train_lora_fp16.py
# Usage (simple):
# python train_lora_fp16.py --train_file train.jsonl --output_dir out_lora_fp16
# Or use accelerate if you prefer: accelerate launch train_lora_fp16.py --train_file ...

import os, json
os.environ.setdefault("HF_HOME", "/root/Agent")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="/root/Agent/baichuan-7b")
parser.add_argument("--train_file", default="/root/Agent/train.jsonl")
parser.add_argument("--output_dir", default="./out_lora_fp16")
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess(example):
    prompt = example["prompt"].strip()
    response = example["response"].strip()
    full = prompt + "\n" + response
    tok = tokenizer(full, truncation=True, max_length=args.max_length, padding="max_length")
    tok["labels"] = tok["input_ids"].copy()
    return tok

print("Loading dataset...")
ds = load_dataset("json", data_files={"train": args.train_file})
ds = ds.map(lambda x: preprocess(x), batched=False)
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("Loading model (fp16) ...")
model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_4bit=True, device_map="auto",trust_remote_code=True)

# prepare LoRA (we don't change prepare_model_for_kbit_training because not using kbit)
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=args.r,
    lora_alpha=args.lora_alpha,
    target_modules=args.target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
    optim="paged_adamw_32bit",
    report_to=[]
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("Starting training...")
trainer.train()
trainer.save_model(args.output_dir)
print("Saved adapters to", args.output_dir)
