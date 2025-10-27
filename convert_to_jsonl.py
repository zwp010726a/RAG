import json

# 原始 JSON 文件路径（包含 [ ... ] 的格式）
input_file = "input.jsonl"  
# 输出 JSONL 文件路径
output_file = "knowledge_base.jsonl"  

# 读取原始 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 写入 JSONL 文件
with open(output_file, "w", encoding="utf-8") as f_out:
    for item in data:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"转换完成，已生成 JSONL 文件：{output_file}")
