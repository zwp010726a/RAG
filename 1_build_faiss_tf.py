# 1_build_faiss_tf.py
import os, json
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

os.environ.setdefault("HF_HOME", "/root/Agent")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "torch:", torch.__version__)

# 替换为你的本地 e5 路径
E5_PATH = "/root/Agent/e5-small"

# load tokenizer & encoder with transformers (not sentence-transformers)
tokenizer = AutoTokenizer.from_pretrained(E5_PATH, use_fast=False)
model = AutoModel.from_pretrained(E5_PATH, trust_remote_code=True).to(device)
model.eval()

def mean_pooling(model_output, attention_mask):
    # model_output[0] is last_hidden_state
    token_embeddings = model_output[0]  # (batch_size, seq_len, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

# 示例文档（请替换为你的语料并做段落切分）
docs = [
    {"id":"doc1","text":"中国的首都是北京。"},
    {"id":"doc2","text":"Python 是一种广泛使用的编程语言。"},
    {"id":"doc3","text":"FAISS 是用于高效相似度搜索的库。"}
]

texts = [d["text"] for d in docs]
batch_size = 8
embs = []
with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt").to(device)
        out = model(**encoded)
        pooled = mean_pooling(out, encoded["attention_mask"])  # (B, hidden)
        pooled = pooled.cpu().numpy()
        embs.append(pooled)
embs = np.vstack(embs).astype("float32")

# L2 normalize for cosine similarity with IndexFlatIP
faiss.normalize_L2(embs)
d = embs.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embs)

# 保存
faiss.write_index(index, "faiss_index.bin")
with open("doc_meta.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("Saved faiss_index.bin and doc_meta.json. dim =", d)
