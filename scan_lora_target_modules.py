# scan_lora_target_modules.py
import torch
from transformers import AutoModelForCausalLM

BASE_MODEL = "/root/Agent/baichuan-7b"  # 替换成你的模型路径

print(f"🔹 Loading base model from {BASE_MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    load_in_4bit=True,  # 或 False，如果你只想扫描模块
    trust_remote_code=True
)

print("🔹 Scanning for Linear layers (potential LoRA target modules)...\n")
linear_modules = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_modules.append(name)
        print(name)

print("\n✅ Scan complete!")
print(f"Found {len(linear_modules)} Linear layers.")
print("\n💡 Example LoRA target_modules could be one or more of these:")
print(linear_modules[:10], "...")  # 只显示前10个示例
