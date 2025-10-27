# simple_rag_run.py
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# demo knowledge base
docs = [
    "中国的首都是北京。",
    "Python 是一种广泛使用的编程语言。",
    "FAISS 是一个高效的向量检索库。"
]

# embedding model (small and fast)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
doc_embs = embed_model.encode(docs, convert_to_numpy=True, show_progress_bar=False)

d = doc_embs.shape[1]
# normalize for cosine with IndexFlatIP
faiss.normalize_L2(doc_embs)
index = faiss.IndexFlatIP(d)
index.add(doc_embs)

# small generator for demo
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

def retrieve(query, k=2):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [docs[i] for i in I[0]]

def make_prompt(query, retrieved):
    prompt = "下面是检索到的证据：\n"
    for i, r in enumerate(retrieved):
        prompt += f"[证据{i+1}] {r}\n"
    prompt += f"\n请根据上面证据回答问题：{query}\n回答："
    return prompt

if __name__ == "__main__":
    q = "中国的首都是哪里？"
    retrieved = retrieve(q, k=2)
    print("retrieved:", retrieved)
    prompt = make_prompt(q, retrieved)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(input_ids, max_new_tokens=64, do_sample=False)
    print("=== OUTPUT ===")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
