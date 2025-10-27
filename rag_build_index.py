import os
import json
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "all-MiniLM-L6-v2")  # 本地模型
DATA_PATH = os.path.join(BASE_DIR, "knowledge_base_text.jsonl")  # 你的知识库
INDEX_PATH = os.path.join(BASE_DIR, "rag_index.faiss")  # 输出 FAISS 索引
TEXTS_PATH = os.path.join(BASE_DIR, "rag_texts.pkl")  # 保存文本映射

# === 加载模型 ===
print("🔹 Loading local embedding model...")
model = SentenceTransformer(MODEL_PATH)
print("✅ 模型加载成功！")

# === 加载知识库数据 ===
print(f"🔹 Loading dataset: {DATA_PATH}")
texts = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "text" in obj:
                texts.append(obj["text"])
        except Exception as e:
            print("❌ 解析错误:", e)

print(f"✅ 共加载文本 {len(texts)} 条")

# === 生成嵌入向量 ===
print("🔹 Encoding texts...")
embeddings = []
for text in tqdm(texts, desc="Encoding"):
    emb = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
    embeddings.append(emb)

embeddings = np.array(embeddings).astype("float32")
print(f"✅ 向量生成完成，形状: {embeddings.shape}")

# === 构建 FAISS 索引 ===
print("🔹 Building FAISS index...")
index = faiss.IndexFlatIP(embeddings.shape[1])  # 使用内积（IP）匹配
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
print(f"✅ FAISS 索引已保存至: {INDEX_PATH}")

# === 保存文本映射 ===
with open(TEXTS_PATH, "wb") as f:
    pickle.dump(texts, f)
print(f"✅ 文本映射已保存至: {TEXTS_PATH}")

print("\n🎉 索引构建完成，可以直接用于 RAG 检索！")
