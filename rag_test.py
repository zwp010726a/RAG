import os
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 基础路径设置 ==========
EMBED_MODEL_PATH = "/root/Agent/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "/root/Agent/faiss_index/knowledge.index"
TEXTS_PATH = "/root/Agent/faiss_index/texts.npy"
LLM_PATH = "/root/Agent/baichuan-7b"   # ← 你的Baichuan或其他模型目录

# ========== 加载模型 ==========
print("🚀 正在加载 SentenceTransformer 向量模型...")
embedder = SentenceTransformer(EMBED_MODEL_PATH)

print("📦 正在加载 FAISS 索引...")
index = faiss.read_index(FAISS_INDEX_PATH)
texts = np.load(TEXTS_PATH, allow_pickle=True)

print("🤖 正在加载本地语言模型...")
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

print("✅ 模型加载完成！\n")

# ========== 构造 Prompt ==========
def build_prompt(query, retrieved_docs):
    context = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)])
    prompt = (
        f"以下是与问题相关的知识片段，请结合这些知识，用简洁、准确的语言回答用户的问题。\n"
        f"【相关知识】\n{context}\n\n"
        f"【问题】{query}\n\n"
        f"【回答】"
    )
    return prompt


# ========== RAG 查询函数 ==========
def rag_query(query, top_k=3, max_new_tokens=256):
    print(f"🧩 输入问题：{query}\n")

    # 1️⃣ 检索相关文档
    query_vec = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec, top_k)
    retrieved = [texts[i] for i in I[0]]

    print("🔍 检索到的知识片段：\n")
    for i, t in enumerate(retrieved, 1):
        print(f"[{i}] {t[:200]}...\n")

    # 2️⃣ 构造 Prompt
    prompt = build_prompt(query, retrieved)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3️⃣ 模型生成
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = output.split("【回答】")[-1].strip()

    print("💡 模型回答：\n")
    print(answer)
    print("\n" + "=" * 80 + "\n")

    return answer


# ========== 主程序 ==========
if __name__ == "__main__":
    print("🧠 RAG 问答系统启动成功！（Baichuan兼容版）")
    print("输入问题后按回车获取增强回答（输入 exit 退出）\n")

    while True:
        query = input("❓请输入你的问题：").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 再见！")
            break
        rag_query(query)
