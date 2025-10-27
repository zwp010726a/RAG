# offline_rag_demo.py
# 目的：无网络环境下快速验证检索->prompt->生成的完整流程（使用 TF-IDF + 模板生成）
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

# 简单知识库（本地）
docs = [
    {"id": "doc1", "text": "中国的首都是北京。北京是中华人民共和国的首都。"},
    {"id": "doc2", "text": "Python 是一种广泛使用的高级编程语言，适合快速开发。"},
    {"id": "doc3", "text": "FAISS 是一个用于高效相似度搜索的向量检索库。"}
]

texts = [d["text"] for d in docs]
vectorizer = TfidfVectorizer().fit(texts)
doc_vecs = vectorizer.transform(texts)  # sparse matrix

def retrieve_tfidf(query, k=2):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, doc_vecs)[0]
    idx = np.argsort(-sims)[:k]
    return [docs[i] for i in idx], sims[idx]

def make_prompt(query, retrieved):
    prompt = "以下是检索到的证据段落（按相关性排序）：\n"
    for i, r in enumerate(retrieved):
        prompt += f"[证据{i+1} | {r['id']}] {r['text']}\n"
    prompt += f"\n请仅基于上面证据回答问题：{query}\n简洁回答："
    return prompt

def simple_generator(prompt):
    # 极简“生成器”：尝试从证据中抽出显式短句作为答案（非生成模型）
    # 若证据中包含“首都”“北京”关键字则直接返回相关句子；否则返回拼接摘要
    if "首都" in prompt:
        for line in prompt.splitlines():
            if "首都" in line and "北京" in line:
                return "北京。（依据：证据中说明“北京是...首都”）"
        # fallback
    # fallback: return first sentence of top evidence
    lines = prompt.splitlines()
    for line in lines:
        if line.startswith("[证据1"):
            # try to extract first sentence
            sent = line.split("] ",1)[1].split("。")[0]
            return sent + "。 (来自证据1)"
    return "抱歉，未能根据证据生成确定答案。"

if __name__ == "__main__":
    q = "中国的首都是哪里？"
    retrieved, sims = retrieve_tfidf(q, k=2)
    print("检索相似度：", sims)
    print("检索到的段落：")
    for r in retrieved:
        print("-", r["id"], r["text"])
    prompt = make_prompt(q, retrieved)
    print("\n=== Prompt 给生成器 ===\n", prompt)
    ans = simple_generator(prompt)
    print("\n=== Generated Answer ===\n", ans)
