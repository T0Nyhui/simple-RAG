import json
import csv
import os

def convert_to_vectara_pure_text(queries_json, answers_json, output_csv):
    print(f"正在生成像 Vectara 一样的纯文本 CSV...")
    
    with open(queries_json, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    with open(answers_json, 'r', encoding='utf-8') as f:
        answers = json.load(f)

    fieldnames = ['query_id', 'query', 'query_run', 'passage_id', 'passage', 'generated_answer']
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
        # 使用 QUOTE_MINIMAL，只有必要时才加引号，保持 CSV 清爽
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for uuid, entry in answers.items():
            query_text = queries.get(uuid, {}).get('query', '')
            ans_text = entry.get('answer', '').strip()
            if not ans_text: continue

            # 1. 获取并格式化引用：把 [1, 5] 变成 " [1], [5]"
            source_ids = entry.get('source_ids', [])
            citation_str = ""
            if source_ids:
                # 按照 Vectara 风格，在末尾加空格和方括号引用
                citation_str = " " + ", ".join([f"[{s}]" for s in source_ids])
            else:
                # 保底引用
                citation_str = " [1]"

            # 2. 拼接纯文本回答（关键：这里不再是 JSON 对象，是纯字符串）
            # 结果示例: The excited modes... (ℓ = 1, 3, 5, 7, ...). [1], [5]
            final_pure_answer = f"{ans_text}{citation_str}"

            # 3. 获取原文列表
            passages = entry.get('sources', [])
            if not passages: passages = ["No context available."]

            # 4. 平铺写入
            for i, p_content in enumerate(passages):
                writer.writerow({
                    "query_id": uuid,
                    "query": query_text,
                    "query_run": 1,
                    "passage_id": f"[{i+1}]", # ID 与引用对应
                    "passage": str(p_content).replace('\n', ' ').strip(),
                    "generated_answer": final_pure_answer # 这里是纯文本！
                })

    print(f"✅ 转换完成！快去看看 {output_csv}，现在绝对没有 json 套娃了。")

# 执行
convert_to_vectara_pure_text(
    "/data2/huirutao/simple-RAG/datasets/subset_test/queries.json",
    "/data2/huirutao/simple-RAG/simple_rag_1/generated_answers.json",
    "answers_v7.csv"
)