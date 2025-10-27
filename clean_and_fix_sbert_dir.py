import os
import json

base_dir = "/root/Agent/all-MiniLM-L6-v2"
transformer_dir = os.path.join(base_dir, "0_Transformer")
pooling_dir = os.path.join(base_dir, "1_Pooling")
modules_json = os.path.join(base_dir, "modules.json")

# ✅ 只保留关键文件（Transformer 模块需要的）
keep_files = {
    "config.json",
    "pytorch_model.bin",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json"
}

# 清理多余文件
for f in os.listdir(transformer_dir):
    if f not in keep_files:
        os.remove(os.path.join(transformer_dir, f))
        print(f"🗑️ 已删除多余文件: {f}")

# ✅ 重建 modules.json（防止旧文件残留）
modules_config = [
    {
        "idx": 0,
        "name": "0_Transformer",
        "path": "0_Transformer",
        "type": "sentence_transformers.models.Transformer"
    },
    {
        "idx": 1,
        "name": "1_Pooling",
        "path": "1_Pooling",
        "type": "sentence_transformers.models.Pooling"
    }
]
with open(modules_json, "w", encoding="utf-8") as f:
    json.dump(modules_config, f, ensure_ascii=False, indent=2)

print("✅ 清理完成，模块文件已更新")

# 打印最终目录结构（可视化检查）
for root, dirs, files in os.walk(base_dir):
    level = root.replace(base_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")

print("\n✅ 现在重新运行:")
print("from sentence_transformers import SentenceTransformer")
print("model = SentenceTransformer('./Agent/all-MiniLM-L6-v2')")
