import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import numpy as np
import evaluate
import pandas as pd
from sklearn.model_selection import train_test_split

# 1️⃣ 데이터 로드
df = pd.read_csv("./preprocessed_twitter.csv")[['text', 'airline_sentiment']]
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['airline_sentiment'].map(label_map)

# 2️⃣ 데이터셋 분할 (Stratified Split)
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# 3️⃣ Dataset 변환 (리스트로 변환)
train_dataset = Dataset.from_dict({'text': train_df['text'].tolist(), 'label': train_df['label'].tolist()})
val_dataset = Dataset.from_dict({'text': val_df['text'].tolist(), 'label': val_df['label'].tolist()})
test_dataset = Dataset.from_dict({'text': test_df['text'].tolist(), 'label': test_df['label'].tolist()})

# 4️⃣ 토큰화 함수 정의 및 적용
model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 5️⃣ PyTorch Tensor 형식으로 변환
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 6️⃣ 모델 로드 및 GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 3
model_sentiment = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    hidden_dropout_prob=0.2,  # 기본 0.1 → 0.2로 증가
    attention_probs_dropout_prob=0.2
    )
model_sentiment.to(device)

# 7️⃣ 평가 지표 (정확도)
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)  # np.argmax 사용
    return accuracy.compute(predictions=preds, references=labels)

# 8️⃣ 훈련 설정
training_args = TrainingArguments(
    output_dir="./sentiment-bert-checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=50,
    seed=42,
    learning_rate = 2e-5,
    warmup_steps = 500,
    weight_decay = 0.01
)

# 9️⃣ Trainer 정의
trainer_sentiment = Trainer(
    model=model_sentiment,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 🔟 모델 학습
trainer_sentiment.train()

# 🔹 모델 저장 (상대 경로로 변경)
model_save_path = "./saved_model/sentiment-bert"
model_sentiment.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
