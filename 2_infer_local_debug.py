import os, json, time, sys
os.environ.setdefault("HF_HOME", "/root/Agent")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import torch
print("START", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print("device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
import faiss, numpy as np

E5_PATH = "/root/Agent/e5-small"
BAICHUAN_PATH = "/root/Agent/baichuan-7b"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

try:
    log("Loading E5 tokenizer...")
    t0 = time.time()
    tok_e5 = AutoTokenizer.from_pretrained(E5_PATH, use_fast=False, trust_remote_code=True)
    log(f"Loaded E5 tokenizer in {time.time()-t0:.2f}s")
except Exception as e:
    log("E5 tokenizer error: " + repr(e))
    raise

try:
    log("Loading E5 model (AutoModel)...")
    t0 = time.time()
    model_e5 = AutoModel.from_pretrained(E5_PATH, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loaded E5 model in {time.time()-t0:.2f}s")
except Exception as e:
    log("E5 model error: " + repr(e))
    raise

try:
    log("Loading baichuan tokenizer (trust_remote_code=True)...")
    t0 = time.time()
    tok_gen = AutoTokenizer.from_pretrained(BAICHUAN_PATH, use_fast=False, trust_remote_code=True)
    log(f"Loaded baichuan tokenizer in {time.time()-t0:.2f}s")
except Exception as e:
    log("baichuan tokenizer error: " + repr(e))
    raise

# Try 4-bit load but log timestamps
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
try:
    log("Attempting baichuan 4-bit load (quantization_config)...")
    t0 = time.time()
    model_gen = AutoModelForCausalLM.from_pretrained(BAICHUAN_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    log(f"Baichuan 4-bit loaded in {time.time()-t0:.2f}s")
except Exception as e:
    log("4-bit load FAILED: " + repr(e))
    log("Attempting fp16 fallback load...")
    try:
        t1 = time.time()
        model_gen = AutoModelForCausalLM.from_pretrained(BAICHUAN_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        log(f"Baichuan fp16 loaded in {time.time()-t1:.2f}s")
    except Exception as e2:
        log("fp16 fallback FAILED: " + repr(e2))
        raise

log("DEBUG SCRIPT COMPLETED.")