import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    pipeline
)
from datasets import Dataset
import numpy as np
import evaluate
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset


df = pd.read_csv("/Users/seokwanwoo/Desktop/ML_DL/preprocessed_twitter.csv")
df = df[['text', 'airline_sentiment']]


label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['airline_sentiment'].map(label_map)

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)


train_dataset = Dataset.from_dict({'text': train_df['text'], 'label': train_df['label']})
val_dataset   = Dataset.from_dict({'text': val_df['text'],  'label': val_df['label']})
test_dataset  = Dataset.from_dict({'text': test_df['text'], 'label': test_df['label']})


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset   = val_dataset.map(tokenize_function,   batched=True)
test_dataset  = test_dataset.map(tokenize_function,  batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch",   columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch",  columns=["input_ids", "attention_mask", "label"])


from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_labels = 3  
model_sentiment = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model_sentiment.to(device)


accuracy = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./sentiment-bert-checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch", 
    load_best_model_at_end=True, 
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=50,
    seed=42
)

trainer_sentiment = Trainer(
    model=model_sentiment,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 학습
trainer_sentiment.train()

# 저장
model_save_path = "/Users/seokwanwoo/Desktop/ML_DL/sentiment-bert-model2"
model_sentiment.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)


test_results = trainer_sentiment.evaluate(test_dataset)

print(test_results)