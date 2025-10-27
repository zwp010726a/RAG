# rag_evaluate.py (高级版 + Top-K 检索命中统计)
import os
import time
import json
import csv
from tqdm import tqdm
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# =====================
# 配置
# =====================
EMBEDDING_MODEL_PATH = './all-MiniLM-L6-v2'
LLM_MODEL_PATH = './baichuan-7b'
TOP_K = 3
TEST_QUESTIONS_FILE = './test_questions.json'  # [{"question": "...", "answer": "..."}]
DOCUMENTS_FILE = './documents.json'             # [{"doc_id": 1, "text": "..."}]
OUTPUT_FILE = './rag_results.csv'

# =====================
# 加载模型
# =====================
print("🚀 加载 SentenceTransformer 向量模型...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

print("📦 加载文档并生成 FAISS 索引...")
with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
    docs = json.load(f)
doc_texts = [d['text'] for d in docs]

doc_embeddings = embed_model.encode(doc_texts, convert_to_numpy=True, batch_size=64)
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)
print(f"📌 FAISS 索引生成完成，文档数量：{len(doc_texts)}")

print("🤖 加载本地 LLM 模型...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
generator = pipeline('text-generation', model=llm_model, tokenizer=tokenizer, device=0 if os.getenv("CUDA_VISIBLE_DEVICES") else -1)
print("✅ 所有模型加载完成！")

# =====================
# 读取测试问题
# =====================
with open(TEST_QUESTIONS_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# =====================
# RAG 查询函数
# =====================
def rag_query(question, top_k=TOP_K):
    start_time = time.time()

    # 向量检索
    q_vec = embed_model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, top_k)
    retrieved_docs = [doc_texts[i] for i in I[0]]

    # 构造 prompt
    prompt = "以下是检索到的知识片段：\n"
    for idx, doc in enumerate(retrieved_docs, 1):
        prompt += f"[{idx}] {doc}\n"
    prompt += f"\n请根据以上内容简洁回答问题：{question}"

    # 生成回答
    output = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.3)[0]['generated_text']
    elapsed = time.time() - start_time
    return output.strip(), retrieved_docs, elapsed

# =====================
# 批量测试
# =====================
results = []
times = []
lengths = []
top_k_hits = []

for item in tqdm(test_data, desc="Evaluating"):
    question = item['question']
    gold_answer = item.get('answer', '')
    generated, retrieved_docs, elapsed = rag_query(question)

    # Top-K 检索命中统计（简单字符串包含判断）
    hit_flags = [int(gold_answer in doc) for doc in retrieved_docs]
    top_k_hits.append(hit_flags)

    results.append({
        'question': question,
        'gold_answer': gold_answer,
        'generated_answer': generated,
        'retrieved_docs': "|".join(retrieved_docs),
        'time_sec': elapsed
    })
    times.append(elapsed)
    lengths.append(len(generated.split()))

# =====================
# 保存结果 CSV
# =====================
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    for row in results:
        writer.writerow(row)
print(f"✅ 测试完成，结果保存至 {OUTPUT_FILE}")

# =====================
# 基础统计分析
# =====================
avg_time = np.mean(times)
avg_length = np.mean(lengths)
all_words = " ".join([r['generated_answer'] for r in results]).split()
unique_ratio = len(set(all_words)) / max(1, len(all_words))

# Top-K 命中率统计
top_k_hits = np.array(top_k_hits)
top1_rate = np.mean(top_k_hits[:,0])
top3_rate = np.mean(np.any(top_k_hits[:,:3], axis=1))
top5_rate = np.mean(np.any(top_k_hits[:,:5], axis=1)) if top_k_hits.shape[1]>=5 else None

print("\n📊 基础统计指标:")
print(f"- 平均生成时间: {avg_time:.2f} 秒")
print(f"- 平均回答长度: {avg_length:.1f} 个词")
print(f"- 生成词唯一率: {unique_ratio:.3f}")
print(f"- Top-1 检索命中率: {top1_rate:.3f}")
print(f"- Top-3 检索命中率: {top3_rate:.3f}")
if top5_rate is not None:
    print(f"- Top-5 检索命中率: {top5_rate:.3f}")
