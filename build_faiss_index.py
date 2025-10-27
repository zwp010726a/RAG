from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os

model_path = "/root/Agent/all-MiniLM-L6-v2"
data_path = "./knowledge_base_text.jsonl"
index_path = "./faiss_index"

os.makedirs(index_path, exist_ok=True)
model = SentenceTransformer(model_path)

texts = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        texts.append(item["text"])

embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, os.path.join(index_path, "knowledge.index"))

np.save(os.path.join(index_path, "texts.npy"), np.array(texts))
print("✅ FAISS 索引构建完成！")
