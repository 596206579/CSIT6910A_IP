# pip install transformers==4.31.0
import pandas as pd
import numpy as np
from transformers import TFBertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt  # 将使用matplotlib库来绘制图像

# 加载训练好的模型
model_path = "Model"  # 模型的路径，这里是Model文件夹，里面包含一个json文件和一个模型文件
model = TFBertForSequenceClassification.from_pretrained(model_path)

# 加载分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 加载新评论数据集
new_reviews_path = "New_amazon_reviews.tsv"
new_reviews_dataset = pd.read_csv(new_reviews_path, sep='\t', nrows=50000, skip_blank_lines=True)

# 分批进行推理（数据量过大将导致内存爆满）
batch_size = 100  # 每批处理的评论数量
predicted_ratings = []  # 存储所有预测的评分

for i in range(0, len(new_reviews_dataset), batch_size):
    batch_reviews = new_reviews_dataset["review_body"].iloc[i:i+batch_size].tolist()
    batch_reviews = [str(review) for review in batch_reviews]
    inputs = tokenizer(batch_reviews, padding=True, truncation=True, return_tensors="tf")
    predictions = model(inputs)
    predicted_labels = np.argmax(predictions.logits, axis=1)
    batch_ratings = predicted_labels + 1
    predicted_ratings.extend(batch_ratings)

# 将预测的评分添加到数据集中
new_reviews_dataset["predicted_rating"] = predicted_ratings

# 保存带有预测评分的新数据集
output_path = "predicted_reviews_dataset.tsv"
new_reviews_dataset.to_csv(output_path, sep='\t', index=False)

# 计算准确率并可视化准确率
actual_ratings = new_reviews_dataset["star_rating"].values
accuracy = np.mean(predicted_ratings == actual_ratings) * 100
plt.bar(['Accuracy'], [accuracy])
plt.ylim(0, 100)
plt.ylabel('Percentage')
plt.title('Model Accuracy')
plt.show()

print("推理完成，预测的评分已保存到", output_path)
print("Accuracy = ", accuracy)

