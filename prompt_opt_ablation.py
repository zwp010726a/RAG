# prompt_opt_ablation.py
# Prompt 优化 Ablation: 原 vs 优化 prompt，提升 BLEU/EM (n=19 queries)

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
import json  # LLM Score JSON 解析

# ========== 路径配置 ==========
E5_PATH = "/root/Agent/e5-base-v2_sbert"
BAICHUAN_PATH = "/root/Agent/baichuan-7b"
DOCS_FILE = "/root/Agent/documents.csv"
QUESTIONS_FILE = "/root/Agent/rag_test_questions.csv"
SAVE_DIR = "./results/prompt_opt"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ 设备: {device}")

# ========== 加载模型 ==========
print("🤖 加载 E5 embedding ...")
retriever = SentenceTransformer(E5_PATH, device=device)

print("🧠 加载 Baichuan-7B ...")
tokenizer = AutoTokenizer.from_pretrained(BAICHUAN_PATH, local_files_only=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(
    BAICHUAN_PATH, torch_dtype=torch.float16, device_map="auto" if device == "cuda" else None,
    local_files_only=True, trust_remote_code=True
)
if device == "cuda":
    llm_model = llm_model.to(device)
llm_model.eval()
if device == "cuda":
    torch.cuda.empty_cache()
print("✅ 模型加载完成！")

# ========== 加载数据 ==========
docs_df = pd.read_csv(DOCS_FILE)
documents = docs_df["text"].dropna().tolist()
doc_embeds = retriever.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
print(f"📚 文档库: {len(documents)} 条")

test_df = pd.read_csv(QUESTIONS_FILE)
test_data = [{"query": row["question"], "gold": row["gold_answer"]} for _, row in test_df.iterrows() if pd.notna(row["question"])]
print(f"❓ 测试集: {len(test_data)} 条")

# ========== 指标函数 ==========
def evaluate_metrics(pred, gold, gen_time):
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([gold.split()], pred.split(), smoothing_function=smooth)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(gold, pred)["rougeL"].fmeasure
    em = 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0
    return bleu, rouge, em, gen_time

def llm_evaluate(generated, gold):
    eval_prompt = f"Rate the generation vs gold answer on 0-1 scale: semantic (consistency), coverage (evidence use), expression (fluency). Output JSON: {{\"semantic\":0.85, \"coverage\":0.7, \"expression\":0.9, \"overall\":0.82}}"
    inputs = tokenizer(eval_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=False)
    score_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        scores = json.loads(score_text.split("JSON:")[-1].strip())  # 解析 JSON
        overall = scores.get("overall", 0.5)
    except:
        overall = 0.5  # fallback
    return overall

# ========== 生成函数 (通用) ==========
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

# ========== RAG 生成 (指定 prompt 变体) ==========
def rag_generate(query, top_k=5, prompt_template="original"):
    q_emb = retriever.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = (q_emb @ doc_embeds.T).squeeze(0)
    top_idx = torch.topk(scores, top_k).indices.tolist()
    retrieved_text = "\n".join([documents[i] for i in top_idx])
    
    if prompt_template == "original":
        prompt = f"以下是检索到的证据：\n{retrieved_text}\n\n基于以上内容，简短回答：{query}"
    elif prompt_template == "optimized":
        prompt = f"基于证据，输出简洁答案，不要重复问题或证据，只输出核心事实。问题：{query} 证据：{retrieved_text} 答案："
    else:
        raise ValueError("Prompt template must be 'original' or 'optimized'")
    
    return generate(prompt)

# ========== 实验执行 ==========
results = []
print("🚀 Prompt Ablation: Original vs Optimized (n=19) ...")
for item in tqdm(test_data):
    q, gold = item["query"], item["gold"]
    
    # Original Prompt
    rag_orig_pred, rag_orig_time = rag_generate(q, prompt_template="original")
    bleu_orig, rouge_orig, em_orig, _ = evaluate_metrics(rag_orig_pred, gold, rag_orig_time)
    llm_orig = llm_evaluate(rag_orig_pred, gold)
    results.append([q, "RAG_Original", rag_orig_pred, bleu_orig, rouge_orig, em_orig, rag_orig_time, llm_orig])
    
    # Optimized Prompt
    rag_opt_pred, rag_opt_time = rag_generate(q, prompt_template="optimized")
    bleu_opt, rouge_opt, em_opt, _ = evaluate_metrics(rag_opt_pred, gold, rag_opt_time)
    llm_opt = llm_evaluate(rag_opt_pred, gold)
    results.append([q, "RAG_Optimized", rag_opt_pred, bleu_opt, rouge_opt, em_opt, rag_opt_time, llm_opt])

# ========== 汇总 ==========
df = pd.DataFrame(results, columns=["query", "mode", "prediction", "BLEU", "ROUGE-L", "EM", "Time (s)", "LLM Score"])
csv_path = os.path.join(SAVE_DIR, "prompt_opt_results.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

orig_avg = df[df["mode"] == "RAG_Original"][["BLEU", "ROUGE-L", "EM", "Time (s)", "LLM Score"]].mean()
opt_avg = df[df["mode"] == "RAG_Optimized"][["BLEU", "ROUGE-L", "EM", "Time (s)", "LLM Score"]].mean()

print("\n📊 平均结果：")
print(f"RAG Original:\n{orig_avg}")
print(f"RAG Optimized:\n{opt_avg}")
print(f"提升%:\n{((opt_avg - orig_avg) / orig_avg * 100).round(1)}")

# ========== 绘图 ==========
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 质量指标
labels = ["BLEU", "ROUGE-L", "EM"]
orig_scores = orig_avg[:3].values
opt_scores = opt_avg[:3].values
x = np.arange(len(labels))
width = 0.35
ax1.bar(x - width/2, orig_scores, width, label="Original", color="#4472C4", alpha=0.7)
ax1.bar(x + width/2, opt_scores, width, label="Optimized", color="#ED7D31", alpha=0.7)
ax1.set_ylabel("Score")
ax1.set_title("质量指标对比 (Pre/Post Opt)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0, 0.3)
ax1.legend()

# LLM Score & Time
ax2.bar(["LLM Score", "Time (s)"], [orig_avg["LLM Score"], orig_avg["Time (s)"]], color="#4472C4", alpha=0.7, label="Original")
ax2.bar(["LLM Score", "Time (s)"], [opt_avg["LLM Score"], opt_avg["Time (s)"]], color="#ED7D31", alpha=0.7, label="Optimized", bottom=[orig_avg["LLM Score"], orig_avg["Time (s)"]])
ax2.set_ylabel("Value")
ax2.set_title("LLM Score & Time 对比")
ax2.legend()

plt.tight_layout()
chart_path = os.path.join(SAVE_DIR, "prompt_opt_chart.png")
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Ablation 完成！CSV: {csv_path}, 图: {chart_path}")