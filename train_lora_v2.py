# train_lora_v3.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===============================
# 基本配置
# ===============================
base_model = "/root/Agent/baichuan-7b"
train_file = "/root/Agent/train.jsonl"
output_dir = "/root/Agent/out_lora_qlo_v3"

# ===============================
# 加载 tokenizer
# ===============================
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===============================
# 加载模型（4-bit QLoRA）
# ===============================
print("🔹 Loading base model (4-bit QLoRA)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 开启 gradient checkpoint 节省显存
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# ===============================
# 应用 LoRA
# ===============================
print("🔹 Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ===============================
# 加载数据集
# ===============================
print("🔹 Loading dataset...")
dataset = load_dataset("json", data_files={"train": train_file})

# ===============================
# Tokenize 函数（关键修正）
# ===============================
def tokenize_function(example):
    # 拼接 prompt + response
    text = example["prompt"].strip() + "\n" + example["response"].strip()
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    # Label 直接复制 input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = dataset["train"].map(tokenize_function, batched=False)

# ===============================
# Data Collator
# ===============================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ===============================
# 训练参数（单卡 24GB 适配）
# ===============================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=5,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

# ===============================
# Trainer 启动
# ===============================
print("🚀 Starting LoRA QLoRA fine-tuning...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(output_dir)
print("✅ Training complete! Model saved to:", output_dir)
