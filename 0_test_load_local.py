import os, sys
os.environ.setdefault("HF_HOME", "/root/Agent")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

# paths (如有不同请改)
e5_path = "/root/Agent/e5-small"
baichuan_path = "/root/Agent/baichuan-7b"

# load embed
try:
    emb = SentenceTransformer(e5_path, device="cuda" if torch.cuda.is_available() else "cpu")
    print("E5 loaded. dim:", emb.get_sentence_embedding_dimension())
except Exception as e:
    print("E5 load error:", e)
    raise

# load tokenizer
try:
    tok = AutoTokenizer.from_pretrained(baichuan_path, use_fast=False)
    print("Baichuan tokenizer loaded.")
except Exception as e:
    print("Tokenizer load error:", e)
    raise

# try 4-bit load (bitsandbytes)
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4")
try:
    model = AutoModelForCausalLM.from_pretrained(baichuan_path, quantization_config=bnb, device_map="auto")
    print("Baichuan loaded with 4-bit quantization OK")
except Exception as e:
    print("4-bit load failed:", e)
    print("Trying fp16 fallback...")
    model = AutoModelForCausalLM.from_pretrained(baichuan_path, torch_dtype=torch.float16, device_map="auto")
    print("Baichuan loaded in fp16 fallback.")
print("Test load completed.")