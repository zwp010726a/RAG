import os
import json
import shutil

# ========== 配置路径 ==========
base_dir = os.path.abspath("/root/Agent/all-MiniLM-L6-v2")  # 你的模型主目录
transformer_dir = os.path.join(base_dir, "0_Transformer")
pooling_dir = os.path.join(base_dir, "1_Pooling")
modules_json = os.path.join(base_dir, "modules.json")

# ========== 检查基础模型文件 ==========
required_files = ["config.json", "pytorch_model.bin", "vocab.txt"]
missing = [f for f in required_files if not os.path.exists(os.path.join(base_dir, f))]
if missing:
    print(f"❌ 缺少必要文件: {missing}")
    print("请确保该目录下有 Transformer 模型文件 (config.json, pytorch_model.bin, vocab.txt 等)")
    exit(1)

# ========== 创建 SentenceTransformer 目录结构 ==========
os.makedirs(transformer_dir, exist_ok=True)
os.makedirs(pooling_dir, exist_ok=True)

# ========== 移动模型文件到 0_Transformer ==========
for f in os.listdir(base_dir):
    path = os.path.join(base_dir, f)
    if os.path.isfile(path) and f not in ["modules.json"]:
        shutil.move(path, transformer_dir)

# ========== 写入 Pooling 配置 ==========
pooling_config = {
    "word_embedding_dimension": 384,  # all-MiniLM-L6-v2 hidden_size
    "pooling_mode_cls_token": False,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_mean_sqrt_len_tokens": False
}
with open(os.path.join(pooling_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(pooling_config, f, ensure_ascii=False, indent=2)

# ========== 写入 modules.json ==========
modules_config = [
    {"idx": 0, "name": "0_Transformer", "path": "0_Transformer"},
    {"idx": 1, "name": "1_Pooling", "path": "1_Pooling"}
]
with open(modules_json, "w", encoding="utf-8") as f:
    json.dump(modules_config, f, ensure_ascii=False, indent=2)

# ========== 完成提示 ==========
print("✅ SentenceTransformer 本地结构修复完成！")
print("当前目录结构：")
for root, dirs, files in os.walk(base_dir):
    level = root.replace(base_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")

print("\n你现在可以直接加载：")
print("from sentence_transformers import SentenceTransformer")
print("model = SentenceTransformer('./Agent/all-MiniLM-L6-v2')")
