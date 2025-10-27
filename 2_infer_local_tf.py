import os, json, time
import faiss, torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
# 环境（离线）
os.environ.setdefault("HF_HOME", "/root/Agent")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "torch:", torch.__version__)

E5_PATH = "/root/Agent/e5-small"
BAICHUAN_PATH = "/root/Agent/baichuan-7b"

# E5 embedding
print("[1/6] Loading E5 tokenizer and model...")
tokenizer_e5 = AutoTokenizer.from_pretrained(E5_PATH, use_fast=False, trust_remote_code=True)
model_e5 = AutoModel.from_pretrained(E5_PATH, trust_remote_code=True).to(device)
model_e5.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

# FAISS load
print("[2/6] Loading FAISS index and metadata...")
index = faiss.read_index("faiss_index.bin")
with open("doc_meta.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# Generator: force load to GPU with max_memory
print("[3/6] Preparing generator tokenizer and model (fp16 -> GPU)...")
tokenizer_gen = AutoTokenizer.from_pretrained(BAICHUAN_PATH, use_fast=False, trust_remote_code=True)
# ensure pad_token
if tokenizer_gen.pad_token is None:
    tokenizer_gen.pad_token = tokenizer_gen.eos_token

# Set max_memory map for a single 24GB GPU (adjust if needed)
max_mem = {"cpu": "120000MB", "cuda:0": "22000MB"}

print("[3.1] Clearing CUDA cache...")
try:
    torch.cuda.empty_cache()
except:
    pass

t0 = time.time()
model_gen = AutoModelForCausalLM.from_pretrained(
    BAICHUAN_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda")
model_gen.eval()
print(f"[3.2] Model loaded in {time.time()-t0:.1f}s. CUDA mem allocated (MB):", torch.cuda.memory_allocated()//1024**2)

# Retrieval / prompt
def embed_query(query):
    with torch.no_grad():
        encoded = tokenizer_e5([query], truncation=True, padding=True, return_tensors="pt").to(device)
        out = model_e5(**encoded)
        pooled = mean_pooling(out, encoded["attention_mask"]).cpu().numpy()
    faiss.normalize_L2(pooled)
    return pooled

def retrieve(query, k=3):
    q_emb = embed_query(query)
    D, I = index.search(q_emb, k)
    return [docs[i] for i in I[0]], D[0]

def build_prompt(query, retrieved):
    prompt = "你是基于证据的问答助手。仅基于下面证据回答，不要编造。\n"
    for i, r in enumerate(retrieved):
        prompt += f"[证据{i+1} | {r['id']}] {r['text']}\n"
    prompt += f"\n问题: {query}\n回答："
    return prompt

# Inference helper - ensure inputs moved to GPU
def answer_with_rag(query, k=3, max_new_tokens=64):
    retrieved, sims = retrieve(query, k=k)
    prompt = build_prompt(query, retrieved)
    print("[RAG] retrieved sims:", sims)
    # tokenize (CPU) then move tensors to device
    inputs = tokenizer_gen(prompt, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # generate
    t1 = time.time()
    out = model_gen.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    t2 = time.time()
    print(f"[RAG] generate time: {t2-t1:.2f}s")
    ans = tokenizer_gen.decode(out[0], skip_special_tokens=True)
    return ans

# Run test
if __name__ == "__main__":
    q = "中国的首都是哪里？"
    print("[TEST] Query:", q)
    ans = answer_with_rag(q, k=2, max_new_tokens=64)
    print("=== ANSWER ===")
    print(ans)