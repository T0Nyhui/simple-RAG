# with open("./final_eval_data.csv","r") as f:
#     for i in range(5):
#         print(f.readline())

# import pandas as pd

# # 读取文件
# df = pd.read_csv('./final_eval_data.csv')

# # 查看前 5 行（默认是 5）
# print(df.head())

import csv

with open('/data2/huirutao/simple-RAG/simple_rag_1/answers_v7.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    # 使用 islice 获取前 5 行
    from itertools import islice
    for row in islice(reader, 2):
        print(row)