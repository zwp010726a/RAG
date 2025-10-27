# generate_dataset.py
import json, random

# 你可以改这些列表，添加更多问法或关键词
questions = [
    "博士生的学制最长年限是多少年？",
    "硕士生的学习年限是几年？最长期限多少？",
    "培养方案中必须包含哪些内容？",
    "研究生入学后多久制定个人培养计划？",
    "学位论文答辩委员会成员应具备什么资格？",
    "中期考核的主要内容包括哪些方面？",
    "延长学业年限的条件有哪些？",
    "博士生的培养方式有哪些？",
    "硕士转博士需要哪些条件？",
    "答辩成绩如何确定？",
    "学术不端检测在什么阶段进行？",
    "学位类型如何影响培养方案？",
    "博士生外国语考试需何时通过？",
    "硕士研究生的学术活动要求有哪些？",
    "课程成绩不达标如何处理？",
    "博士生申请学位需要满足哪些条件？",
    "学位授予单位与论文作者的关系如何规定？"
]

evidence_templates = [
    "华中科技大学规定，{}",
    "根据学校管理办法，{}",
    "学校文件中指出：{}",
    "在培养方案中提到：{}"
]

answer_conclusions = [
    "基于以上证据，简短回答：{}",
    "综上可知：{}",
    "据此可得：{}"
]

def make_one(q):
    # 随机选一个 evidence 句子变体
    ev = random.choice(evidence_templates)
    # 用简单替换把问句填入证据句
    ev_text = ev.format(q.replace("？",""))
    # 构造 answer 主体
    conclusion = q.replace("？","")  # 简化：就复用问句作为简答
    ans = f"以下是检索到的证据：\n[1 | doc1] {ev_text}\n\n{random.choice(answer_conclusions).format(conclusion)}"
    return {"question": q, "answer": ans}

def main(n=1000, out="gen_1000.jsonl"):
    with open(out, "w", encoding="utf-8") as f:
        for _ in range(n):
            q = random.choice(questions)
            item = make_one(q)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main(1000)
