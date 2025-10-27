# inference_lora_v3.py
import os
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --------------------
# 配置（按需修改）
# --------------------
BASE_MODEL = "/root/Agent/baichuan-7b"            # 本地基础模型路径
LORA_WEIGHTS = "/root/Agent/out_lora_qlo_v3"     # 训练输出目录（含 adapter_model.safetensors/bin）
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.1
TOP_P = 0.95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------

def load_tokenizer_and_base(model_path):
    print(f"🔹 Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_base_model_4bit(model_path):
    print(f"🔹 Loading base model (4-bit) from {model_path} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    return model

def load_peft_model(base_model, lora_path):
    # 自动识别 safetensors / bin
    if os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        print("✅ Found adapter_model.safetensors, will load adapter from safetensors.")
    elif os.path.exists(os.path.join(lora_path, "adapter_model.bin")):
        print("✅ Found adapter_model.bin, will load adapter from bin.")
    else:
        print("⚠️ No adapter_model.* found in LORA path; attempting to load whatever is present.")
    print(f"🔹 Loading LoRA adapter from: {lora_path} ...")
    # 注意 device_map="auto" 让 PEFT 尝试把 adapter 放到 GPU
    pefted = PeftModel.from_pretrained(base_model, lora_path, device_map="auto", torch_dtype=torch.float16, local_files_only=True)
    return pefted

def print_param_info(model):
    try:
        s = model.print_trainable_parameters()
        # model.print_trainable_parameters prints to stdout; return None. We still call it to show info.
    except Exception:
        pass

def generate_answer(model, tokenizer, question, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P):
    # 与训练格式保持一致的 prompt 前缀
    prompt = f"请根据证据回答：{question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    t1 = time.time()
    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    # 去掉 prompt 前缀（若模型直接输出包含 prompt）
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    used_tokens = out.shape[1]
    elapsed = t1 - t0
    # GPU mem info
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = {
            "allocated_MB": torch.cuda.memory_allocated(0) / 1024**2,
            "reserved_MB": torch.cuda.memory_reserved(0) / 1024**2
        }
    return answer, elapsed, used_tokens, gpu_mem

def main():
    print(f"✅ 使用设备: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")
    tokenizer = load_tokenizer_and_base(BASE_MODEL)
    base = load_base_model_4bit(BASE_MODEL)
    model = load_peft_model(base, LORA_WEIGHTS)
    print_param_info(model)
    model.eval()

    print("\n🚀 LoRA 推理已就绪（输入 'exit' 退出）\n" + "="*60)
    try:
        while True:
            question = input("🧠 请输入你的问题：\n> ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                break
            answer, elapsed, used_tokens, gpu_mem = generate_answer(model, tokenizer, question)
            result = {
                "question": question,
                "answer": answer,
                "time_s": round(elapsed, 3),
                "used_tokens": int(used_tokens),
                "gpu_mem": {k: round(v,1) for k,v in gpu_mem.items()} if gpu_mem is not None else None
            }
            print("\n🤖 模型回答：")
            print(json.dumps(result, ensure_ascii=False, indent=4))
            print("\n" + "="*60 + "\n")
    except KeyboardInterrupt:
        print("\n退出。")
    finally:
        # 清理
        try:
            del model
            del base
            torch.cuda.empty_cache()
        except Exception:
            pass

if __name__ == "__main__":
    main()
