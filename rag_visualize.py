import pandas as pd
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm

# 文件路径
RESULTS_FILE = "rag_results.csv"
OUTPUT_DIR = "./charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📊 正在加载评估结果...")
df = pd.read_csv(RESULTS_FILE)

# 初始化指标
bleu_scores, rouge_l_scores, em_scores = [], [], []

print("⚙️ 正在计算 BLEU / ROUGE-L / EM 指标...")
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1

for _, row in tqdm(df.iterrows(), total=len(df)):
    ref = str(row["gold_answer"]).strip()
    pred = str(row["generated_answer"]).strip()
    
    # BLEU
    bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
    bleu_scores.append(bleu)
    
    # ROUGE-L
    rouge_l = scorer.score(ref, pred)['rougeL'].fmeasure
    rouge_l_scores.append(rouge_l)
    
    # EM
    em = 1.0 if ref == pred else 0.0
    em_scores.append(em)

df["BLEU"] = bleu_scores
df["ROUGE-L"] = rouge_l_scores
df["EM"] = em_scores

# 计算总体平均
metrics_summary = {
    "BLEU": df["BLEU"].mean(),
    "ROUGE-L": df["ROUGE-L"].mean(),
    "EM": df["EM"].mean(),
    "Avg Time (s)": df["time_sec"].mean(),
}

print("\n✅ 指标汇总：")
for k, v in metrics_summary.items():
    print(f"- {k}: {v:.4f}")

# 保存增强后的结果文件
df.to_csv("rag_results_scored.csv", index=False)
print("\n💾 已保存增强版结果文件：rag_results_scored.csv")

# ======== 图表 1：总体指标柱状图 ========
plt.figure(figsize=(8,5))
plt.bar(metrics_summary.keys(), metrics_summary.values(), color='steelblue')
plt.title("RAG Model Evaluation Summary")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, "summary_bar.png"))
plt.close()

# ======== 图表 2：生成时间分布 ========
plt.figure(figsize=(8,5))
plt.hist(df["time_sec"], bins=20, color='orange', alpha=0.7)
plt.title("Generation Time Distribution")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "time_distribution.png"))
plt.close()

# ======== 图表 3：BLEU/ROUGE 对比折线图 ========
plt.figure(figsize=(8,5))
plt.plot(df["BLEU"], label='BLEU', marker='o')
plt.plot(df["ROUGE-L"], label='ROUGE-L', marker='x')
plt.title("Per-Sample BLEU & ROUGE-L Scores")
plt.xlabel("Sample Index")
plt.ylabel("Score")
plt.legend()
plt.grid(True, alpha=0.4)
plt.savefig(os.path.join(OUTPUT_DIR, "bleu_rouge_line.png"))
plt.close()

print(f"\n📈 已生成图表文件在：{OUTPUT_DIR}/")
print("包括：summary_bar.png, time_distribution.png, bleu_rouge_line.png")
