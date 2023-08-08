# pip install transformers==4.31.0
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import pandas as pd
import datetime

# 启用混合精度训练，提高训练精度
from tensorflow.python.keras.mixed_precision.policy import Policy, set_global_policy
policy = Policy('mixed_float16')
set_global_policy(policy)

# 加载预训练的BERT模型和分词器，并指定输出层有5个神经元
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 加载数据集
dataset_path = "processed_amazon_reviews_us_Camera_v1_00.tsv"
chunksize = 5000  # 修改处：分块加载数据

# 准备输入数据
def generate_examples():
    for chunk in pd.read_csv(dataset_path, sep='\t', chunksize=chunksize):
        for index, row in chunk.iterrows():
            if index > 50000:
                break
            example = InputExample(guid=None,
                                   text_a=row["review_body"],
                                   text_b=None,
                                   label=row["star_rating"]-1)  # 减1使得标签值在0到4之间
            yield example

# 将输入数据转换为模型需要的格式
def generate_features():
    for e in generate_examples():
        try:
            input_features = tokenizer.encode_plus(e.text_a,
                                                   add_special_tokens=True,
                                                   max_length=128,
                                                   truncation=True,  # 修改处：添加截断
                                                   padding='max_length',  # 修改处：更改填充方式
                                                   return_attention_mask=True)
            yield InputFeatures(input_ids=input_features["input_ids"],
                                attention_mask=input_features["attention_mask"],
                                token_type_ids=input_features["token_type_ids"],
                                label=e.label)
        except Exception as error:
            pass

# 创建TensorFlow数据集
def gen():
    for f in generate_features():
        yield ({'input_ids': f.input_ids, 'attention_mask': f.attention_mask, 'token_type_ids': f.token_type_ids}, f.label)

dataset = tf.data.Dataset.from_generator(gen,
                                         ({'input_ids': tf.int32, 'attention_mask': tf.int32, 'token_type_ids': tf.int32}, tf.int64),
                                         ({'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]), 'token_type_ids': tf.TensorShape([None])}, tf.TensorShape([])))

# 分割训练集和验证集
DATASET_SIZE = len(list(generate_features()))  # 修改处：获取特征长度
train_size = int(0.9 * DATASET_SIZE)
val_size = int(0.1 * DATASET_SIZE)
dataset = dataset.shuffle(DATASET_SIZE)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

# 设置TensorBoard回调
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 训练模型
model.fit(train_dataset.shuffle(100).batch(16),  # 修改处：减小批次大小
          epochs=2,
          validation_data=val_dataset.batch(16),  # 修改处：减小批次大小
          callbacks=[tensorboard_callback])

# 保存模型
model_save_path = "Model"
model.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")
