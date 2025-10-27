from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)  # 打印加载进度

model_name = "/root/Agent/baichuan-7b"
train_file = "/root/Agent/train.jsonl"
output_dir = "/root/Agent/out_lora_fp16_test"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# bitsandbytes 4-bit 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("=== Start loading model on GPU ===")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"0":"24GB"},  # 全部放在 GPU0
    trust_remote_code=True
)
print("=== Model loaded on GPU ===")

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 数据集
dataset = load_dataset("json", data_files={"train": train_file})["train"]

# Trainer
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args
)

# 开始训练
trainer.train()
