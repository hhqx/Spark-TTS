import os
import pandas as pd
import csv  # 需导入csv模块用于指定quoting参数

# 构造包含\t和\n的DataFrame
data = {
    "id": [1, 2],
    "content": [
        "这是一段包含\t制表符的文本\n并且有换行",  # 含\t和\n
        "正常文本（无特殊字符）"
    ]
}
df = pd.DataFrame(data)

os.makedirs("outputs", exist_ok=True)

# 保存为TSV，自动处理\t和\n
df.to_csv(
    "outputs/output.tsv",
    sep='\t',  # 关键：指定为TSV格式
    quoting=csv.QUOTE_NONNUMERIC,  # 非数字字段自动加引号
    quotechar='"',  # 用双引号包裹字段
    index=False,  # 不保存索引
    encoding='utf-8'
)

df.to_csv(
    "outputs/output.csv",
    index=True,
    encoding='utf-8'
)