import torch
import os
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from accelerate import Accelerator

# ==============================
# 基本配置（全部本地）
# ==============================
model_name = "/root/Agent/baichuan-7b"   # 本地模型路径
train_file = "/root/Agent/train.jsonl"   # 本地训练数据
output_dir = "/root/Agent/out_lora_qlo_test"

# ==============================
# 初始化 Accelerate
# ==============================
accelerator = Accelerator()
device = accelerator.device
print(f"🚀 Using device: {device}")

# ==============================
# 加载分词器（本地，不联网）
# ==============================
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded from local path.")

# ==============================
# QLoRA 量化配置
# ==============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ==============================
# 加载模型（本地，使用GPU）
# ==============================
print("🔹 Loading base model (4-bit QLoRA, local only)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
print("✅ Model loaded successfully from local path.")

# ==============================
# LoRA 配置
# ==============================
target_modules = [
    "W_pack", "o_proj", "down_proj", "up_proj", "gate_proj"
]

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print("✅ LoRA adapters applied.")

# ==============================
# 加载本地数据集
# ==============================
print("🔹 Loading local dataset...")
with open(train_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
print("Example sample:", data[0])

# 假设每行数据格式为 {"text": "..."}，若字段不同请修改
dataset = Dataset.from_list(data)

def tokenize_fn(examples):
    # 将 prompt + response 拼接为单条文本输入
    texts = [
        f"用户：{p}\n助手：{r}"
        for p, r in zip(examples['prompt'], examples['response'])
    ]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256
    )


tokenized_dataset = dataset.map(tokenize_fn, batched=True)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"✅ Loaded {len(train_dataset)} training samples and {len(eval_dataset)} eval samples.")

# ==============================
# 数据整理器
# ==============================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ==============================
# 训练参数
# ==============================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=False,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    evaluation_strategy="no",
    report_to="none",
)

# ==============================
# Trainer + Accelerate
# ==============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 准备阶段 (Accelerate)
model, trainer = accelerator.prepare(model, trainer)

# ==============================
# 显存信息
# ==============================
if torch.cuda.is_available():
    print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
    print(f"   Reserved:  {torch.cuda.memory_reserved(0)/1024**2:.1f} MB")

# ==============================
# 启动训练
# ==============================
print("🚀 Starting QLoRA fine-tuning (offline)...")
trainer.train()
print("🎯 Training completed successfully!")

# ==============================
# 保存模型
# ==============================
trainer.save_model(output_dir)
print(f"✅ Model saved to {output_dir}")
