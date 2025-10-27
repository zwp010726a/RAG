import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model = "/root/Agent/baichuan-7b"
lora_model = "/root/Agent/out_lora_qlo_test"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")

print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)

print("🔹 Loading base model (4-bit quantization)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

print(f"🔹 Loading LoRA adapter from: {lora_model}")
adapter_file = os.path.join(lora_model, "adapter_model.safetensors")

if os.path.exists(adapter_file):
    print("✅ Found adapter_model.safetensors, loading it properly...")
    model = PeftModel.from_pretrained(model, lora_model, is_trainable=False)
else:
    raise FileNotFoundError("❌ adapter_model.safetensors not found!")

print("🔹 Model parameter info:")
model.print_trainable_parameters()

# -------------------- 推理 --------------------
prompt = "请根据证据回答：中国的首都是哪里？"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("🚀 Generating answer...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128)

print("\n✅ 模型输出：\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
