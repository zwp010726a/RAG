# -*- coding: utf-8 -*-
"""
LLM-based 生成质量评估模块
作者: GPT-5
说明: 使用 Baichuan-7B 模型对 RAG 生成答案的质量进行 LLM 打分。
评分维度: 正确性 / 完整性 / 表达清晰度 (总分 1-10)
"""

import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# ==============================
# 参数配置
# ==============================
BAICHUAN_PATH = "/root/Agent/baichuan-7b"  # 你的 Baichuan 模型目录
RESULTS_DIR = "./results"
INPUT_CSV = os.path.join(RESULTS_DIR, "results_e5-base-v2.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "results_with_llm_score.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# 加载模型（半精度 + eval 模式）
# ==============================
print("🤖 加载 Baichuan-7B 模型 (半精度 + eval 模式)...")
tokenizer = AutoTokenizer.from_pretrained(BAICHUAN_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BAICHUAN_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
model.eval()

# ==============================
# 定义打分函数
# ==============================
def llm_score_response(question, evidence, answer):
    """
    使用 Baichuan 生成打分（输出一个 1~10 的整数）
    """
    prompt = f"""
请你扮演一个评估专家，针对以下问答结果打分。
【评分标准】
1. 正确性（是否与事实一致）；
2. 完整性（是否包含必要要点）；
3. 表达清晰度（是否语句通顺易懂）；
总分 1~10 分，输出一个整数分数，不要解释。

【问题】
{question}

【检索到的证据】
{evidence}

【模型回答】
{answer}

请直接输出一个整数分数：
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取分数（简单正则）
        import re
        match = re.search(r"(\d{1,2})", text)
        if match:
            score = int(match.group(1))
            score = max(1, min(score, 10))  # 限制范围
        else:
            score = None
    except Exception as e:
        print(f"⚠️ LLM 打分失败: {e}")
        score = None
    return score


# ==============================
# 读取已有结果
# ==============================
print(f"📂 读取结果文件: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
# ---- 自动列名映射 ----
col_map = {
    "retrieved_docs": "retrieved_texts",
    "generated_text": "generated_answer"
}

for old, new in col_map.items():
    if old not in df.columns and new in df.columns:
        df[old] = df[new]

# ---- 检查列是否齐全 ----
if not all(col in df.columns for col in ["question", "retrieved_docs", "generated_text"]):
    raise ValueError("CSV 文件缺少必要列: question / retrieved_docs / generated_text")


scores = []

# ==============================
# 执行评分
# ==============================
for i, row in tqdm(df.iterrows(), total=len(df), desc="LLM-based 生成质量评估中"):
    q = str(row["question"])
    evi = str(row["retrieved_docs"])[:800]  # 避免超长
    ans = str(row["generated_text"])
    score = llm_score_response(q, evi, ans)
    scores.append(score)

df["LLM_Score"] = scores
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 已保存 LLM 打分结果: {OUTPUT_CSV}")

# ==============================
# 汇总统计与图表输出
# ==============================
mean_score = df["LLM_Score"].dropna().mean()
print(f"📊 平均 LLM 评分: {mean_score:.2f}")

# 生成论文用图
plt.figure(figsize=(7, 5))
plt.hist(df["LLM_Score"].dropna(), bins=10, color="#5DADE2", edgecolor="black")
plt.title("LLM-based 生成质量得分分布", fontsize=14)
plt.xlabel("评分 (1~10)")
plt.ylabel("样本数量")
plt.grid(alpha=0.3)
chart_path = os.path.join(RESULTS_DIR, "charts_paper/llm_score_distribution.png")
os.makedirs(os.path.dirname(chart_path), exist_ok=True)
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"📈 图表已保存: {chart_path}")

print("🎯 LLM-based 质量评估完成，可直接用于论文结果章节。")
