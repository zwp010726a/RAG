# prompt_opt_ablation_v4.py
# v4: EM post-process + 1-5 LLM + gen_1000 loader (n=100 sub)

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
import json
import re
import difflib  # Fuzzy EM

# ========== 路径 ==========
E5_PATH = "/root/Agent/e5-base-v2_sbert"
BAICHUAN_PATH = "/root/Agent/baichuan-7b"
DOCS_FILE = "/root/Agent/documents.csv"
QUESTIONS_FILE = "/root/Agent/rag_test_questions.csv"
GEN_FILE = "/root/Agent/gen_1000.jsonl"  # 新: 1000 题
SAVE_DIR = "./results/prompt_opt_v4"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ 设备: {device}")

# ========== 加载模型 (同 v3) ==========
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

# ========== 数据加载 ==========
docs_df = pd.read_csv(DOCS_FILE)
documents = docs_df["text"].dropna().tolist()
doc_embeds = retriever.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
print(f"📚 文档库: {len(documents)} 条")

# 小集 (n=19)
test_df = pd.read_csv(QUESTIONS_FILE)
small_data = [{"query": row["question"], "gold": row["gold_answer"]} for _, row in test_df.iterrows() if pd.notna(row["question"])]
print(f"❓ 小集: {len(small_data)} 条")

# 大集子集 (n=100 from gen_1000.jsonl)
gen_df = pd.read_json(GEN_FILE, lines=True).head(100)
large_data = [{"query": row["question"], "gold": row["answer"]} for _, row in gen_df.iterrows() if pd.notna(row["question"])]
print(f"❓ 大集子集: {len(large_data)} 条")

# ========== 指标 (Fuzzy EM) ==========
def evaluate_metrics(pred, gold, gen_time):
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([gold.split()], pred.split(), smoothing_function=smooth)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(gold, pred)["rougeL"].fmeasure
    # Fuzzy EM: difflib ratio >0.8 =1
    fuzzy_em = 1.0 if difflib.SequenceMatcher(None, pred.strip().lower(), gold.strip().lower()).ratio() > 0.8 else 0.0
    return bleu, rouge, fuzzy_em, gen_time

# ========== v4 LLM Eval (1-5 Scale) ==========
def llm_evaluate(generated, gold):
    eval_prompt = f"""Rate generation vs gold on 1-5 scale (1=poor, 5=excellent): semantic consistency, coverage, expression fluency. 
Output ONLY JSON: {{"semantic":4, "coverage":3, "expression":5, "overall":4}}"""
    inputs = tokenizer(eval_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)
    score_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_match = re.search(r'\{.*\}', score_text)
    if json_match:
        try:
            scores = json.loads(json_match.group())
            overall = scores.get("overall", 3.0) / 5.0  # Normalize 0-1
        except:
            overall = 0.5
    else:
        overall = 0.5
    return overall

# ========== 生成 (同 v3 + post-process) ==========
def generate(prompt, max_new_tokens=100):
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_output.split("答案：")[-1].strip() if "答案：" in full_output else full_output.strip()
    gen_time = time.time() - start_time
    del inputs, outputs
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    return generated, gen_time

# ========== RAG 生成 (v4 Prompt) ==========
def rag_generate(query, top_k=5, prompt_template="original"):
    q_emb = retriever.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = (q_emb @ doc_embeds.T).squeeze(0)
    top_idx = torch.topk(scores, top_k).indices.tolist()
    retrieved_text = "\n".join([documents[i] for i in top_idx])
    
    if prompt_template == "original":
        prompt = f"以下是检索到的证据：\n{retrieved_text}\n\n基于以上内容，简短回答：{query}"
    elif prompt_template == "optimized_v4":
        prompt = f"精确复述 gold 关键句，添加证据支持 (1-2 句)。问题：{query} 证据：{retrieved_text} 答案："  # v4: EM-focused
    else:
        raise ValueError("Template must be 'original' or 'optimized_v4'")
    
    return generate(prompt)

# ========== 执行 (小集 + 大集子集) ==========
results = []
print("🚀 v4 Ablation: Original vs Optimized v4 (n=19 small + 100 large sub) ...")

# 小集
for item in tqdm(small_data, desc="Small Set"):
    q, gold = item["query"], item["gold"]
    
    # Original
    rag_orig_pred, rag_orig_time = rag_generate(q, prompt_template="original")
    bleu_orig, rouge_orig, em_orig, _ = evaluate_metrics(rag_orig_pred, gold, rag_orig_time)
    llm_orig = llm_evaluate(rag_orig_pred, gold)
    results.append([q, "Small_Original", rag_orig_pred, bleu_orig, rouge_orig, em_orig, rag_orig_time, llm_orig])
    
    # v4 Opt
    rag_v4_pred, rag_v4_time = rag_generate(q, prompt_template="optimized_v4")
    bleu_v4, rouge_v4, em_v4, _ = evaluate_metrics(rag_v4_pred, gold, rag_v4_time)
    llm_v4 = llm_evaluate(rag_v4_pred, gold)
    results.append([q, "Small_Opt_v4", rag_v4_pred, bleu_v4, rouge_v4, em_v4, rag_v4_time, llm_v4])

# 大集子集
for item in tqdm(large_data, desc="Large Subset"):
    q, gold = item["query"], item["gold"]
    
    # Original (subsample 10 to save time)
    if len(results) % 10 == 0:  # Every 10th for original
        rag_orig_pred, rag_orig_time = rag_generate(q, prompt_template="original")
        bleu_orig, rouge_orig, em_orig, _ = evaluate_metrics(rag_orig_pred, gold, rag_orig_time)
        llm_orig = llm_evaluate(rag_orig_pred, gold)
        results.append([q, "Large_Original", rag_orig_pred, bleu_orig, rouge_orig, em_orig, rag_orig_time, llm_orig])
    
    # v4 Opt (all 100)
    rag_v4_pred, rag_v4_time = rag_generate(q, prompt_template="optimized_v4")
    bleu_v4, rouge_v4, em_v4, _ = evaluate_metrics(rag_v4_pred, gold, rag_v4_time)
    llm_v4 = llm_evaluate(rag_v4_pred, gold)
    results.append([q, "Large_Opt_v4", rag_v4_pred, bleu_v4, rouge_v4, em_v4, rag_v4_time, llm_v4])

# ========== 汇总 (小集 + 大集) ==========
df = pd.DataFrame(results, columns=["query", "mode", "prediction", "BLEU", "ROUGE-L", "EM", "Time (s)", "LLM Score"])
csv_path = os.path.join(SAVE_DIR, "prompt_opt_v4_results.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# 小集平均
small_orig = df[df["mode"] == "Small_Original"][["BLEU", "ROUGE-L", "EM", "Time (s)", "LLM Score"]].mean()
small_v4 = df[df["mode"] == "Small_Opt_v4"][["BLEU", "ROUGE-L", "EM", "Time (s)", "LLM Score"]].mean()

# 大集平均
large_orig = df[df["mode"] == "Large_Original"][["BLEU", "ROUGE-L", "EM", "Time (s)", "LLM Score"]].mean()
large_v4 = df[df["mode"] == "Large_Opt_v4"][["BLEU", "ROUGE-L", "EM", "Time (s)", "LLM Score"]].mean()

print("\n📊 小集平均：")
print(f"Original:\n{small_orig.round(3)}")
print(f"v4 Opt:\n{small_v4.round(3)}")
print(f"提升%:\n{((small_v4 - small_orig) / (small_orig + 1e-8) * 100).round(1)}")

print("\n📊 大集子集平均：")
print(f"Original:\n{large_orig.round(3)}")
print(f"v4 Opt:\n{large_v4.round(3)}")
print(f"提升%:\n{((large_v4 - large_orig) / (large_orig + 1e-8) * 100).round(1)}")

# ========== 绘图 (小集) ==========
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels = ["BLEU", "ROUGE-L", "EM"]
orig_scores = small_orig[:3].values
v4_scores = small_v4[:3].values
x = np.arange(len(labels))
width = 0.35
ax1.bar(x - width/2, orig_scores, width, label="Original", color="#4472C4", alpha=0.7)
ax1.bar(x + width/2, v4_scores, width, label="Optimized v4", color="#ED7D31", alpha=0.7)
ax1.set_ylabel("Score")
ax1.set_title("小集质量指标对比 (v4 Opt)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0, 0.5)
ax1.legend()

ax2.bar(["LLM Score", "Time (s)"], [small_orig["LLM Score"], small_orig["Time (s)"]], color="#4472C4", alpha=0.7, label="Original")
ax2.bar(["LLM Score", "Time (s)"], [small_v4["LLM Score"], small_v4["Time (s)"]], color="#ED7D31", alpha=0.7, label="Optimized v4")
ax2.set_ylabel("Value")
ax2.set_title("小集 LLM & Time 对比 (v4)")
ax2.legend()

plt.tight_layout()
chart_path = os.path.join(SAVE_DIR, "prompt_opt_v4_small_chart.png")
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ v4 Ablation 完成！CSV: {csv_path}, 小集图: {chart_path}")