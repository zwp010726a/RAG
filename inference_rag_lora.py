# inference_rag_lora.py
# 🎯 基于 LoRA 微调权重 + 检索增强的多轮问答推理模板

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import json

# ===============================
# 🔹 配置
# ===============================
MODEL_PATH = "/root/Agent/out_lora_qlo_v3"  # 训练好的 LoRA 模型
BASE_MODEL = "baichuan-7b"                 # 用于 tokenizer 和 base model
DEVICE = "cuda"
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 256

# 检索数据
DOCS_PATH = "./knowledge_base_text.jsonl"  # 每行: {"text": "..."}
EMBED_MODEL = "all-MiniLM-L6-v2"      # 小型 embedding 模型
TOP_K = 3                             # 检索 top-k

# ===============================
# 🔹 加载模型
# ===============================
print("🔹 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL, 
    use_fast=True, 
    trust_remote_code=True  # ✅ 允许执行自定义代码
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True  # ✅ 允许执行自定义代码
)
model.eval()


# ===============================
# 🔹 加载知识库并构建向量索引
# ===============================
print("🔹 Loading knowledge base and building FAISS index...")
texts = []
with open(DOCS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        texts.append(item["text"])

embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"✅ Knowledge base loaded, {len(texts)} documents indexed.")

# ===============================
# 🔹 多轮问答函数
# ===============================
conversation_history = []

def retrieve_context(query, top_k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(q_emb, top_k)
    context = "\n".join([texts[i] for i in ids[0]])
    return context

def chat(query, use_rag=True):
    global conversation_history
    conversation_history.append({"role": "user", "content": query})

    # 构造 prompt
    prompt_text = ""
    for turn in conversation_history:
        role = "用户" if turn["role"] == "user" else "助手"
        prompt_text += f"{role}: {turn['content']}\n"
    
    # 检索增强
    if use_rag:
        context = retrieve_context(query)
        prompt_text = f"以下是相关知识:\n{context}\n\n{prompt_text}"

    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                       padding=True, max_length=MAX_INPUT_LENGTH).to(DEVICE)

    # 生成
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 去掉 prompt 部分，只保留最新回答
    response = response.replace(prompt_text, "").strip()
    conversation_history.append({"role": "assistant", "content": response})
    return response

# ===============================
# 🔹 测试多轮问答
# ===============================
if __name__ == "__main__":
    print("\n✨ RAG + LoRA Chat Bot Ready! 输入 'exit' 结束对话.\n")
    while True:
        query = input("用户: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = chat(query)
        print("助手:", answer)
