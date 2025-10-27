import argparse, json, os
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import tqdm

os.environ.setdefault("HF_HOME", "/root/Agent")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

E5_PATH = "/root/Agent/e5-small"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load e5 (transformers + mean pooling)
tokenizer_e5 = AutoTokenizer.from_pretrained(E5_PATH, use_fast=False, trust_remote_code=True)
model_e5 = AutoModel.from_pretrained(E5_PATH, trust_remote_code=True).to(device)
model_e5.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

# load index & docs
index = faiss.read_index("faiss_index.bin")
with open("doc_meta.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

def embed_query(query):
    encoded = tokenizer_e5([query], truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_e5(**encoded)
        pooled = mean_pooling(out, encoded["attention_mask"]).cpu().numpy()
    faiss.normalize_L2(pooled)
    return pooled

def retrieve(query, topk=3):
    emb = embed_query(query)
    D,I = index.search(emb, topk)
    retrieved = [docs[i] for i in I[0]]
    scores = D[0].tolist()
    return retrieved, scores

def build_prompt_from_retrieved(question, retrieved):
    s = "下面是用于回答的问题的证据片段：\n"
    for i,r in enumerate(retrieved):
        s += f"[证据{i+1}|{r['id']}] {r['text']}\n"
    s += f"\n请基于以上证据回答问题：\n问题: {question}\n回答："
    return s

def main(args):
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in tqdm.tqdm(fin):
            item = json.loads(line)
            q = item.get("question") or item.get("prompt") or item.get("input")
            ans = item.get("answer") or item.get("response") or item.get("output")
            if q is None or ans is None:
                continue
            retrieved, scores = retrieve(q, topk=args.topk)
            prompt = build_prompt_from_retrieved(q, retrieved)
            out = {"prompt": prompt, "response": ans}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    print("Wrote", args.output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="train.jsonl")
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()
    main(args)