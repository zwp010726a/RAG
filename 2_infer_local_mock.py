import os, json, time
os.environ.setdefault("HF_HOME", "/root/Agent")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from transformers import AutoTokenizer, AutoModel
import torch, faiss
# using local e5 for embeddings (already tested)
E5_PATH = "/root/Agent/e5-small"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# load e5 encoder
tokenizer_e5 = AutoTokenizer.from_pretrained(E5_PATH, use_fast=False, trust_remote_code=True)
model_e5 = AutoModel.from_pretrained(E5_PATH, trust_remote_code=True).to(device)
model_e5.eval()
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

# load faiss index & docs
index = faiss.read_index("faiss_index.bin")
with open("doc_meta.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

def embed_query(query):
    encoded = tokenizer_e5([query], truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_e5(**encoded)
        v = mean_pooling(out, encoded["attention_mask"]).cpu().numpy()
    faiss.normalize_L2(v)
    return v

def retrieve(query, k=3):
    v = embed_query(query)
    D, I = index.search(v, k)
    return [docs[i] for i in I[0]], D[0]

# MOCK GENERATOR: 拼接 top-k 证据并生成一个简短回答（节约资源、无需 baichuan）
def mock_generate_from_retrieved(query, k=3):
    retrieved, sims = retrieve(query, k=k)
    # 简易拼接策略：把 top-k 的文本拼成“证据 + 简短结论”
    out = "以下是检索到的证据：\n"
    for i, r in enumerate(retrieved):
        out += f"[{i+1} | {r['id']}] {r['text']}\n"
    out += "\n基于以上证据，简短回答："
    # 这里简单返回第1条的简短核心（真实场景换成真实 LLM）
    core = retrieved[0]["text"].split("。")[0]
    out += core
    return out, sims, retrieved

if __name__ == "__main__":
    q = "中国的首都是哪里？"
    ans, sims, ret = mock_generate_from_retrieved(q, k=2)
    print("sims:", sims)
    print("=== MOCK ANSWER ===")
    print(ans)
    # 保存一个训练样本示例（用于 LoRA 数据）
    sample = {"prompt": "请根据证据回答：" + q, "response": ans}
    with open("sample_train.jsonl", "w", encoding="utf-8") as f:
        import json
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved sample_train.jsonl (1 example)")