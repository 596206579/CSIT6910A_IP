import pandas as pd

# 读取.tsv文件，并忽略格式错误的行
df = pd.read_csv('amazon_reviews_us_Camera_v1_00.tsv', sep='\t', error_bad_lines=False)

# 只保留'star_rating'和'review_body'两列
df = df[['star_rating', 'review_body']]

# 保存到新的.tsv文件
df.to_csv('processed_amazon_reviews_us_Camera_v1_00.tsv', sep='\t', index=False)
