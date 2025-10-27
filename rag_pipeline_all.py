#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一键式离线 RAG 流程：
- 多 embedding 比较（本地模型）
- 固定生成模型：Baichuan-7B（本地）
- 输出：每模型 CSV、总 Excel 报表、论文级图表
"""

import os
import time
import json
import faiss
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ================= CONFIG =================
ROOT = Path("/root/Agent")
BAICHUAN_PATH = ROOT / "baichuan-7b"               # 你的 Baichuan-7B 本地目录
RAG_RESULTS_CSV = ROOT / "rag_results.csv"         # 测试问题（question,gold_answer,...）
DOCUMENTS_CSV = ROOT / "documents.csv"             # 文档库，包含 doc_id,text 列
OUTPUT_DIR = ROOT / "results"
CHARTS_DIR = OUTPUT_DIR / "charts"

# 本地 embedding 模型映射： key=display name, value=本地模型目录
EMBED_MODELS = {
    "e5-base-v2": ROOT / "e5-base-v2",
    "bge-large-zh": ROOT / "bge-large-zh",
    "gte-large": ROOT / "gte-large",
}

TOP_K = 5
MAX_CONTEXT_TOKENS = 1500  # 给生成模型拼接的上下文最大 token（粗略截断）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_EMBED = 128
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# --------- 载入测试集 ----------
if not RAG_RESULTS_CSV.exists():
    raise FileNotFoundError(f"测试集文件 {RAG_RESULTS_CSV} 不存在，请提供 rag_results.csv")

df_test = pd.read_csv(RAG_RESULTS_CSV)
if "question" not in df_test.columns:
    raise KeyError("rag_results.csv 需要包含 'question' 列（以及可选的 gold_answer）")

questions = df_test["question"].astype(str).tolist()
gold_answers = df_test["gold_answer"].astype(str).tolist() if "gold_answer" in df_test.columns else ["" for _ in questions]

# --------- 载入文档库 ----------
if not DOCUMENTS_CSV.exists():
    raise FileNotFoundError(f"documents.csv 未找到 ({DOCUMENTS_CSV})。请先生成文档库 CSV（包含列 doc_id,text）。")

df_docs = pd.read_csv(DOCUMENTS_CSV)
if "text" not in df_docs.columns:
    raise KeyError("documents.csv 必须包含列 'text'")

docs = df_docs["text"].astype(str).tolist()
doc_ids = df_docs["doc_id"].astype(str).tolist() if "doc_id" in df_docs.columns else [str(i) for i in range(len(docs))]

print(f"📌 文档库加载，文档数：{len(docs)}")

# --------- 载入 Baichuan-7B tokenizer + model ----------
print(f"🤖 加载生成模型（Baichuan）到 {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(str(BAICHUAN_PATH), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(str(BAICHUAN_PATH), trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
model.eval()

# 准备评测工具
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
smooth = SmoothingFunction().method1

def compute_metrics(ref, pred):
    # BLEU (sentence-level, unigram-4 weighted)
    try:
        ref_tok = [ref.split()]
        pred_tok = pred.split()
        bleu = sentence_bleu(ref_tok, pred_tok, smoothing_function=smooth)
    except Exception:
        bleu = 0.0
    # ROUGE-L
    r = scorer.score(ref, pred)
    rouge_l = r["rougeL"].fmeasure
    # EM
    em = 1.0 if ref.strip() == pred.strip() and len(ref.strip())>0 else 0.0
    return bleu, rouge_l, em

# --------- 主流程：对每个 embedding 模型分别建立索引、检索、生成、评估 ----------
summary_rows = []
all_model_results = {}

for embed_name, embed_dir in EMBED_MODELS.items():
    print(f"\n=== 处理 embedding 模型：{embed_name} (dir={embed_dir}) ===")
    if not Path(embed_dir).exists():
        print(f"⚠️ 本地 embedding 模型目录不存在：{embed_dir} ，跳过该模型。")
        continue

    # 加载 sentence-transformers
    print("🔁 加载 SentenceTransformer ...")
    embedder = SentenceTransformer("/root/Agent/e5-base-v2_sbert", device=DEVICE)


    # 计算文档向量并建立 FAISS 索引（如果已缓存，可加缓存逻辑，这里每次重新计算）
    print("📦 计算文档向量并建立 FAISS 索引 ...")
    doc_embeddings = []
    for i in tqdm(range(0, len(docs), BATCH_EMBED), desc="embedding docs"):
        batch_texts = docs[i:i+BATCH_EMBED]
        embs = embedder.encode(batch_texts, convert_to_numpy=True, batch_size=len(batch_texts), show_progress_bar=False, normalize_embeddings=True)
        doc_embeddings.append(embs)
    doc_embeddings = np.vstack(doc_embeddings).astype("float32")
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)

    # 对每个问题检索 + 用 Baichuan 生成
    model_rows = []
    timings = []
    print("🔎 开始对测试集进行检索+生成 ...")
    for q_idx, question in enumerate(tqdm(questions, desc="questions")):
        start_t = time.time()
        q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        D, I = index.search(q_emb.astype("float32"), TOP_K)
        retrieved_texts = [docs[idx] for idx in I[0]]
        retrieved_ids = [doc_ids[idx] for idx in I[0]]

        # 构造 prompt：把 Top-K 文档合并到上下文（按简单模板）
        context = "\n\n".join([f"【文档_{retrieved_ids[i]}】 {retrieved_texts[i]}" for i in range(len(retrieved_texts))])
        prompt = f"请根据下面的检索到的证据，简洁准确回答问题。\n\n检索到的证据：\n{context}\n\n问题：{question}\n\n简短回答："

        # 限制 prompt 长度（粗略截断为字符级）
        if len(prompt) > 16000:
            prompt = prompt[-16000:]  # 保留后面的一部分（如若过长，可改更合理的截断策略）

        # tokenize + generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        gen_kwargs = dict(max_new_tokens=256, do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.1, eos_token_id=tokenizer.eos_token_id)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        generated = tokenizer.decode(out[0], skip_special_tokens=True)
        # 生成文本通常包含 prompt + answer，取 prompt 后面的内容
        if "简短回答：" in generated:
            generated_answer = generated.split("简短回答：")[-1].strip()
        else:
            # 尝试把 prompt 的最后一部分去掉
            generated_answer = generated[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
        elapsed = time.time() - start_t
        timings.append(elapsed)

        # 评估（有 gold_answer 则计算）
        gold = gold_answers[q_idx] if q_idx < len(gold_answers) else ""
        bleu, rouge_l, em = compute_metrics(gold, generated_answer)

        # 检索命中（简单：参考答案是否为检索文本子串）
        hit = any((gold.strip() != "" and gold.strip() in txt) for txt in retrieved_texts)

        row = {
            "question": question,
            "gold_answer": gold,
            "generated_answer": generated_answer,
            "retrieved_ids": "|".join(retrieved_ids),
            "retrieved_texts": " || ".join(retrieved_texts),
            "time_sec": elapsed,
            "bleu": bleu,
            "rouge_l": rouge_l,
            "em": em,
            "retrieval_hit": int(hit),
        }
        model_rows.append(row)

    # 汇总并保存每模型 CSV
    df_model = pd.DataFrame(model_rows)
    out_csv = OUTPUT_DIR / f"results_{embed_name}.csv"
    df_model.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存每题结果：{out_csv} (n={len(df_model)})")

    mean_bleu = df_model["bleu"].mean()
    mean_rouge = df_model["rouge_l"].mean()
    mean_em = df_model["em"].mean()
    mean_time = df_model["time_sec"].mean()
    retrieval_topk_hit = df_model["retrieval_hit"].mean()

    summary_rows.append({
        "embed_model": embed_name,
        "mean_bleu": mean_bleu,
        "mean_rouge_l": mean_rouge,
        "mean_em": mean_em,
        "mean_time_sec": mean_time,
        "retrieval_hit_rate": retrieval_topk_hit,
        "n": len(df_model)
    })

    all_model_results[embed_name] = df_model

# --------- 汇总所有模型结果到 Excel ----------
df_summary = pd.DataFrame(summary_rows)
excel_path = OUTPUT_DIR / "rag_all_results.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="summary", index=False)
    for name, dfm in all_model_results.items():
        dfm.to_excel(writer, sheet_name=name[:30], index=False)
print(f"\n📊 汇总已保存到 Excel: {excel_path}")

# --------- 生成论文图表 ----------
print("📈 生成论文用图表 ...")
# 1) BLEU/ROUGE 柱状比较
plt.figure(figsize=(8,5))
x = df_summary["embed_model"]
x_pos = np.arange(len(x))
plt.bar(x_pos - 0.15, df_summary["mean_bleu"], width=0.3, label="BLEU")
plt.bar(x_pos + 0.15, df_summary["mean_rouge_l"], width=0.3, label="ROUGE-L")
plt.xticks(x_pos, x, rotation=30)
plt.ylabel("score")
plt.title("Embedding model comparison (BLEU / ROUGE-L)")
plt.legend()
plt.tight_layout()
plt.savefig(CHARTS_DIR / "bleu_rouge_comparison.png")
plt.close()

# 2) Time vs BLEU scatter
plt.figure(figsize=(8,5))
plt.scatter(df_summary["mean_time_sec"], df_summary["mean_bleu"])
for i, txt in enumerate(df_summary["embed_model"]):
    plt.annotate(txt, (df_summary["mean_time_sec"].iat[i], df_summary["mean_bleu"].iat[i]))
plt.xlabel("avg gen time (s)")
plt.ylabel("avg BLEU")
plt.title("Time vs BLEU (per embedding model)")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "time_vs_bleu.png")
plt.close()

# 3) retrieval hit rate bar
plt.figure(figsize=(8,5))
plt.bar(df_summary["embed_model"], df_summary["retrieval_hit_rate"])
plt.title(f"Top-{TOP_K} retrieval hit rate (gold included in retrieved text)")
plt.ylabel("hit rate")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "retrieval_hit_rate.png")
plt.close()

print(f"✅ 图表保存在：{CHARTS_DIR}")
print("🎉 流程完成。请查看 results/ 下的 Excel/CSV 与 charts/ 图片，用于论文绘图和表格。")
