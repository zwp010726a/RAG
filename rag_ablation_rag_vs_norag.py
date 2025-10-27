# rag_ablation_real_fixed.py
# 完整 RAG vs NoRAG：documents.csv ("text") + rag_test_questions.csv (QA)

import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import gc

# ========== 1️⃣ 路径配置 ==========
E5_PATH = "/root/Agent/e5-base-v2_sbert"
BAICHUAN_PATH = "/root/Agent/baichuan-7b"
DOCS_FILE = "/root/Agent/documents.csv"
QUESTIONS_FILE = "/root/Agent/rag_test_questions.csv"
SAVE_DIR = "./results/ablation_real"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ 设备: {device}")

# ========== 2️⃣ 加载模型 ==========
print("🤖 加载 E5-base-v2_sbert embedding 模型 ...")
retriever = SentenceTransformer(E5_PATH, device=device)

print("🧠 加载 Baichuan-7B 模型 ...")
tokenizer = AutoTokenizer.from_pretrained(
    BAICHUAN_PATH, 
    local_files_only=True,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(
    BAICHUAN_PATH,
    torch_dtype=torch.float16,
    device_map="auto" if device == "cuda" else None,  
    local_files_only=True,
    trust_remote_code=True
)
if device == "cuda":
    llm_model = llm_model.to(device)
llm_model.eval()
if device == "cuda":
    torch.cuda.empty_cache()
print("✅ 模型加载完成！")

# ========== 3️⃣ 加载数据集 ==========
# 文档
docs_df = pd.read_csv(DOCS_FILE)
print("📄 文档列名:", docs_df.columns.tolist())
documents = docs_df["text"].dropna().tolist()
doc_embeds = retriever.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
print(f"📚 文档库: {len(documents)} 条")

# 测试集
test_df = pd.read_csv(QUESTIONS_FILE)
print("❓ 测试列名:", test_df.columns.tolist())
if "question" not in test_df.columns or "gold_answer" not in test_df.columns:
    raise ValueError(f"测试集列错误！预期 'question'/'gold_answer'，实际: {test_df.columns.tolist()}. 请修复 CSV。")
test_data = [{"query": row["question"], "gold": row["gold_answer"]} for _, row in test_df.iterrows() if pd.notna(row["question"])]
print(f"❓ 测试集: {len(test_data)} 条")

# ========== 4️⃣ 指标 ==========
def evaluate_metrics(pred, gold, gen_time):
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([gold.split()], pred.split(), smoothing_function=smooth)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(gold, pred)["rougeL"].fmeasure
    em = 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0
    return bleu, rouge, em, gen_time

# ========== 5️⃣ 生成 ==========
def generate(prompt, max_new_tokens=100):
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    gen_time = time.time() - start_time
    del inputs, outputs
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    return generated, gen_time

# ========== 6️⃣ RAG ==========
def rag_generate(query, top_k=5):
    q_emb = retriever.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = (q_emb @ doc_embeds.T).squeeze(0)
    top_idx = torch.topk(scores, top_k).indices.tolist()
    retrieved_text = "\n".join([documents[i] for i in top_idx])
    prompt = f"已知资料：{retrieved_text}\n问题：{query}\n请基于资料简洁回答。"
    return generate(prompt)

# ========== 7️⃣ NoRAG ==========
def no_rag_generate(query):
    prompt = f"问题：{query}\n请直接回答。"
    return generate(prompt)

# ========== 8️⃣ 执行 ==========
rag_results, norag_results = [], []
print("🚀 执行实验 ...")
for item in tqdm(test_data):
    q, gold = item["query"], item["gold"]
    rag_pred, rag_time = rag_generate(q)
    no_rag_pred, no_rag_time = no_rag_generate(q)
    bleu_rag, rouge_rag, em_rag, _ = evaluate_metrics(rag_pred, gold, rag_time)
    bleu_no, rouge_no, em_no, _ = evaluate_metrics(no_rag_pred, gold, no_rag_time)
    rag_results.append([q, "RAG", rag_pred, bleu_rag, rouge_rag, em_rag, rag_time])
    norag_results.append([q, "NoRAG", no_rag_pred, bleu_no, rouge_no, em_no, no_rag_time])

# ========== 9️⃣ 汇总 ==========
df = pd.DataFrame(rag_results + norag_results, columns=["query", "mode", "prediction", "BLEU", "ROUGE-L", "EM", "Time (s)"])
csv_path = os.path.join(SAVE_DIR, "rag_vs_norag_real.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

rag_avg = df[df["mode"] == "RAG"][["BLEU", "ROUGE-L", "EM", "Time (s)"]].mean()
norag_avg = df[df["mode"] == "NoRAG"][["BLEU", "ROUGE-L", "EM", "Time (s)"]].mean()

print("\n📊 平均结果：")
print(f"RAG:\n{rag_avg}")
print(f"NoRAG:\n{norag_avg}")

# ========== 10️⃣ 绘图 ==========
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels = ["BLEU", "ROUGE-L", "EM"]
rag_scores = rag_avg[:-1].values
norag_scores = norag_avg[:-1].values
x = np.arange(len(labels))
width = 0.35
ax1.bar(x - width/2, rag_scores, width, label="RAG", color="#4472C4")
ax1.bar(x + width/2, norag_scores, width, label="NoRAG", color="#ED7D31")
ax1.set_ylabel("Score")
ax1.set_title("质量指标对比")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0, 1)
ax1.legend()

times = ['RAG', 'NoRAG']
ax2.bar(times, [rag_avg['Time (s)'], norag_avg['Time (s)']], color=['#4472C4', '#ED7D31'])
ax2.set_ylabel("Time (s)")
ax2.set_title("生成时间对比")

plt.tight_layout()
chart_path = os.path.join(SAVE_DIR, "rag_vs_norag_chart_real.png")
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ 完成！\n{csv_path}\n{chart_path}")