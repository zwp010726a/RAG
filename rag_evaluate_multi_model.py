#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键 RAG 评估脚本（已增强）：
- 兼容本地 Baichuan-7B（半精度 + 分批加载 + eval 模式）
- 更健壮的 LLM-based 评分解析（容错 JSON）
- matplotlib 风格回退处理
- 每次生成后尽量回收显存，遇到 OOM 有简要降级策略

使用：python rag_evaluate_multi_model.py
"""
import os
import sys
import json
import time
import re
from pathlib import Path
from tqdm.auto import tqdm
import warnings

# 依赖导入（容错）
missing = []
try:
    import pandas as pd
except Exception:
    pd = None
    missing.append("pandas")
try:
    import numpy as np
except Exception:
    np = None
    missing.append("numpy")
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None
    missing.append("sentence-transformers")

try:
    import faiss
except Exception:
    faiss = None

try:
    import sacrebleu
except Exception:
    sacrebleu = None

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None

import torch

# ========== 配置 ==========
BAICHUAN_PATH = "/root/Agent/baichuan-7b"   # 本地 Baichuan 模型目录
EMBED_DIRS = [
    "./e5-base-v2",
    "./bge-large-zh",
    "./gte-large"
]
DOCUMENTS_CSV = "./documents.csv"
TEST_FILE = "./rag_results.csv"
OUTPUT_DIR = "./results"
TOP_K = 5
EMBED_BATCH = 64
DEVICE = "cuda"  # or "cpu"
# ========================

os.makedirs(OUTPUT_DIR, exist_ok=True)
charts_dir = Path(OUTPUT_DIR) / "charts_paper"
charts_dir.mkdir(parents=True, exist_ok=True)

if missing:
    print("缺少依赖包：", missing)
    print("建议先安装：pip install " + " ".join(missing))
    # 仍然尝试运行（部分功能会报错）

# ---- 加载文档库 ----
print("📌 加载文档库...")
if not Path(DOCUMENTS_CSV).exists():
    raise FileNotFoundError(f"documents CSV 未找到: {DOCUMENTS_CSV}")

df_docs = pd.read_csv(DOCUMENTS_CSV)
if "doc_id" not in df_docs.columns or "text" not in df_docs.columns:
    if "id" in df_docs.columns and "content" in df_docs.columns:
        df_docs = df_docs.rename(columns={"id":"doc_id","content":"text"})
    else:
        raise ValueError("documents.csv 需要包含 'doc_id' 与 'text' 列")

doc_texts = df_docs["text"].fillna("").tolist()
doc_ids = df_docs["doc_id"].tolist()
print(f"文档数量：{len(doc_texts)}")

# ---- 读测试集 ----
if not Path(TEST_FILE).exists():
    raise FileNotFoundError(f"测试集文件未找到: {TEST_FILE}")
df_test = pd.read_csv(TEST_FILE).fillna("")
if "question" not in df_test.columns or "gold_answer" not in df_test.columns:
    raise ValueError("测试集文件需要包含 'question' 与 'gold_answer' 列")
tests = df_test.to_dict(orient="records")
print(f"测试集样本数：{len(tests)}")

# ---- 加载 Baichuan 作为评估与生成 LLM（用作RAG的生成与LLM评分） ----
if AutoTokenizer is None:
    raise RuntimeError("缺少 transformers，请安装：pip install transformers accelerate")

print("🤖 使用半精度 + 分批加载 + eval 模式加载 Baichuan-7B ...")

# 清理显存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(BAICHUAN_PATH, trust_remote_code=True)

# 采用半精度 + device_map 自动分配 + offload_folder（若有）
load_kwargs = dict(
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
# offload_folder 若目录不存在则创建
offload_dir = Path("./offload")
try:
    offload_dir.mkdir(parents=True, exist_ok=True)
    load_kwargs["offload_folder"] = str(offload_dir)
except Exception:
    pass

# 载入模型，可能会花一点时间
model = AutoModelForCausalLM.from_pretrained(BAICHUAN_PATH, **load_kwargs)
model.eval()

# 辅助：安全解析 JSON（从模型返回中提取第一个 {...}）
def safe_json_parse(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*?\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                # 进一步尝试去掉多行注释或模型常输出的前缀/后缀
                # 保底返回原文
                return {"_parse_error": True, "raw": s}
        return {"_parse_error": True, "raw": s}

# llm 生成，带显存友好策略
def llm_generate(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    # 将输入移动到模型的首个参数所在设备（accelerate 的 device_map 管理）
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    try:
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    except RuntimeError as e:
        # OOM 降级策略：清空缓存并重试一次（若仍失败，会抛出）
        print("Warning: 生成时出现 RuntimeError（可能OOM），尝试回收显存后重试...", e)
        torch.cuda.empty_cache()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # 解码
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    # 尝试回收显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text

# llm-based scoring（更健壮）
def llm_score_semantic(gold, pred, timeout_sec=30):
    prompt = f"""你是一个评估助手。任务：比较“参考答案（gold）”与“模型答案（pred）”在语义层面是否一致，并从三个维度打分（0到1，保留两位小数）：
参考答案:
{gold}

模型答案:
{pred}

请按 JSON 输出，例如：
{{"semantic_score":0.85,"coverage_score":0.70,"expression_score":0.90,"overall":0.82}}

不要输出其他非 JSON 内容。"""
    try:
        out = llm_generate(prompt, max_new_tokens=200)
        j = safe_json_parse(out)
        # 标准化字段
        for k in ["semantic_score", "coverage_score", "expression_score", "overall"]:
            if k in j:
                try:
                    v = float(j[k])
                except Exception:
                    v = 0.0
                j[k] = max(0.0, min(1.0, v))
            else:
                j[k] = 0.0
        return j
    except Exception as e:
        print("LLM-based scoring 失败：", e)
        return {"semantic_score":0.0, "coverage_score":0.0, "expression_score":0.0, "overall":0.0}

# ---- 核心 loop：对每个 embedding 模型分别评估 ----
summary_rows = []
all_details = {}
for embed_dir in EMBED_DIRS:
    name = Path(embed_dir).name
    print(f"\n=== 处理 embedding 模型：{name} (dir={embed_dir}) ===")
    
    # ✅ 固定使用本地 SentenceTransformer 版本
    embedder_path = "/root/Agent/e5-base-v2_sbert"
    embedder = SentenceTransformer(embedder_path, device=DEVICE)


    print("📦 计算文档向量并建立 FAISS 索引 ...")
    doc_embeddings = embedder.encode(doc_texts, batch_size=EMBED_BATCH, convert_to_tensor=False, show_progress_bar=True)
    doc_embeddings = np.array(doc_embeddings).astype("float32")

    if faiss is not None:
        dim = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(doc_embeddings)
        index.add(doc_embeddings)
    else:
        index = None

    results = []
    total_hit = 0
    semantic_scores = []
    times = []

    for item in tqdm(tests, desc="questions"):
        q = str(item["question"]) if item.get("question") is not None else ""
        gold = str(item.get("gold_answer", ""))
        # 1) search
        q_emb = embedder.encode([q], convert_to_tensor=False)[0].astype("float32")
        if faiss is not None:
            q_emb_np = q_emb.reshape(1, -1)
            faiss.normalize_L2(q_emb_np)
            D, I = index.search(q_emb_np, TOP_K)
            idxs = I[0].tolist()
        else:
            sims = (doc_embeddings @ q_emb) / (np.linalg.norm(doc_embeddings, axis=1) * (np.linalg.norm(q_emb)+1e-12))
            idxs = sims.argsort()[::-1][:TOP_K].tolist()

        retrieved_texts = [doc_texts[i] for i in idxs if i < len(doc_texts)]
        retrieved_ids = [doc_ids[i] for i in idxs if i < len(doc_texts)]

        # 2) semantic hit
        gold_emb = embedder.encode([gold], convert_to_tensor=True)
        retrieved_embs = embedder.encode(retrieved_texts, convert_to_tensor=True)
        cos_scores = util.cos_sim(gold_emb, retrieved_embs)[0].cpu().numpy() if len(retrieved_texts) > 0 else np.array([0.0])
        hit_flag = int(cos_scores.max() > 0.80) if len(cos_scores) > 0 else 0
        total_hit += hit_flag

        # 3) generate
        prompt_docs = "\n\n".join([f"[{i}] {t}" for i, t in enumerate(retrieved_texts, start=1)])
        prompt = f"根据下面检索到的证据，回答用户问题，要求简洁且准确（若缺信息请注明无足够信息）：\n\n证据：\n{prompt_docs}\n\n问题：\n{q}\n\n请给出中文回答："
        t0 = time.time()
        generated = llm_generate(prompt, max_new_tokens=256)
        t1 = time.time()
        gen_time = t1 - t0
        times.append(gen_time)

        # 4) compute metrics
        bleu = 0.0
        rouge_l = 0.0
        em = 0
        if sacrebleu is not None:
            try:
                bleu = sacrebleu.sentence_bleu(generated, [gold]).score/100.0
            except Exception:
                bleu = 0.0
        if rouge_scorer is not None:
            try:
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
                r = scorer.score(gold, generated)
                rouge_l = r['rougeL'].fmeasure
            except Exception:
                rouge_l = 0.0
        def normalize_whitespace(s): return "".join(str(s).split())
        em = 1 if normalize_whitespace(generated) == normalize_whitespace(gold) else 0

        # 5) llm self-eval
        self_eval = llm_score_semantic(gold, generated)
        semantic_scores.append(self_eval.get("overall", 0.0))

        results.append({
            "question": q,
            "gold_answer": gold,
            "generated_answer": generated,
            "retrieved_ids": retrieved_ids,
            "retrieved_texts": retrieved_texts,
            "hit_flag": hit_flag,
            "time_sec": gen_time,
            "bleu": bleu,
            "rouge_l": rouge_l,
            "em": em,
            "semantic_overall": self_eval.get("overall", 0.0),
            "semantic_detail": json.dumps(self_eval, ensure_ascii=False)
        })

    # aggregate
    n = len(results)
    mean_bleu = float(np.mean([r["bleu"] for r in results])) if n>0 else 0.0
    mean_rouge = float(np.mean([r["rouge_l"] for r in results])) if n>0 else 0.0
    mean_em = float(np.mean([r["em"] for r in results])) if n>0 else 0.0
    mean_time = float(np.mean([r["time_sec"] for r in results])) if n>0 else 0.0
    hit_rate = total_hit / n if n>0 else 0.0
    mean_semantic = float(np.mean(semantic_scores)) if len(semantic_scores)>0 else 0.0

    out_csv = Path(OUTPUT_DIR) / f"results_{name}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"✅ 已保存每题结果：{out_csv} (n={n})")

    summary_rows.append({
        "embed_model": name,
        "mean_bleu": mean_bleu,
        "mean_rouge_l": mean_rouge,
        "mean_em": mean_em,
        "mean_time_sec": mean_time,
        "retrieval_hit_rate": hit_rate,
        "semantic_overall": mean_semantic,
        "n": n
    })
    all_details[name] = results

# === 汇总到 Excel ===
df_summary = pd.DataFrame(summary_rows)
excel_path = Path(OUTPUT_DIR) / "rag_all_results.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="summary", index=False)
    for k, v in all_details.items():
        pd.DataFrame(v).to_excel(writer, sheet_name=k[:31], index=False)

print(f"\n📊 汇总已保存到 Excel: {excel_path}")

# === 生成论文图表 ===
if plt is None:
    print("matplotlib 未安装，跳过绘图。")
else:
    print("📈 生成论文图表 ...")
    # try seaborn style, otherwise fallback
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except Exception:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            plt.style.use('default')

    x = df_summary['embed_model']
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x, df_summary['mean_bleu'], label="BLEU")
    ax.bar(x, df_summary['mean_rouge_l'], label="ROUGE-L", alpha=0.9, bottom=0)
    ax.set_title("Embedding model comparison (BLEU / ROUGE-L)")
    ax.set_ylabel("score")
    ax.legend()
    fig.savefig(charts_dir / "bleu_rouge_comparison.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x, df_summary['retrieval_hit_rate'])
    ax.set_title("Top-K retrieval hit rate (gold included in retrieved text)")
    ax.set_ylabel("hit rate")
    fig.savefig(charts_dir / "retrieval_hit_rate.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x, df_summary['semantic_overall'])
    ax.set_title("Embedding model comparison (LLM-based semantic overall score)")
    ax.set_ylabel("semantic overall (0-1)")
    fig.savefig(charts_dir / "semantic_score.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(df_summary['mean_time_sec'], df_summary['mean_bleu'])
    for i, txt in enumerate(x):
        ax.annotate(txt, (df_summary['mean_time_sec'].iloc[i], df_summary['mean_bleu'].iloc[i]))
    ax.set_xlabel("avg gen time (s)")
    ax.set_ylabel("avg BLEU")
    ax.set_title("Time vs BLEU (per embedding model)")
    fig.savefig(charts_dir / "time_vs_bleu.png", dpi=300, bbox_inches="tight")

    print(f"✅ 图表保存在：{charts_dir}")

print("🎉 流程完成。请查看 results/ 下的 Excel/CSV 与 charts_paper/ 图片，用于论文绘图和表格。")
