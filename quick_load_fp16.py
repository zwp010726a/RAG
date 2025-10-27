import os, time
os.environ.setdefault("HF_HOME","/root/Agent")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("HF_DATASETS_OFFLINE","1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print("START", time.strftime("%Y-%m-%d %H:%M:%S"))
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
BAI="/root/Agent/baichuan-7b"
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(BAI, use_fast=False, trust_remote_code=True)
print("Tokenizer loaded.")
t0=time.time()
print("Loading model (fp16, low_cpu_mem_usage=True, device_map='auto') ...")
model = AutoModelForCausalLM.from_pretrained(BAI, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
print("Model loaded in", time.time()-t0, "seconds")
print("DONE")