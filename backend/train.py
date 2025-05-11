# backend/train.py

import os
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from sklearn.model_selection import train_test_split

# --- CONFIG ---
MODEL_NAME = "gpt2"
OUTPUT_DIR = Path(__file__).parent / "models" / datetime.now().strftime("ft_%Y%m%d")
DATA_FILE  = Path(__file__).parent / "data" / "chat_history.jsonl"
BATCH_SIZE = 2
EPOCHS     = 1
BLOCK_SIZE = 512  # max tokens per example
SAVE_TOTAL_LIMIT = 2

# Tạo thư mục output
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ---
lines = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            text = obj["prompt"] + " " + obj["response"]
            lines.append(text)
        except:
            continue

if len(lines) < 2:
    print(f"Chỉ có {len(lines)} mẫu, không đủ dữ liệu để fine-tune. Bỏ qua bước train.")
    exit(0)
train_texts, val_texts = train_test_split(lines, test_size=0.1, random_state=42)

# --- Tokenizer & Model ---

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# đảm bảo có token đặc biệt nếu cần
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# --- Dataset class ---
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.examples = []
        for txt in texts:
            tokenized = tokenizer(
                txt,
                truncation=True,
                max_length=BLOCK_SIZE,
                padding="max_length",
            )
            self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = {k: torch.tensor(v) for k, v in self.examples[i].items()}
        # For causal LM, labels = input_ids
        item["labels"] = item["input_ids"].clone()
        return item

train_dataset = TextDataset(train_texts)
val_dataset   = TextDataset(val_texts)

# --- Data collator ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# --- TrainingArguments ---
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=500,
    logging_steps=100,
    save_total_limit=SAVE_TOTAL_LIMIT,
    learning_rate=5e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# --- Train ---
trainer.train()

# --- Save final model/tokenizer ---
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"Fine-tune completed. Model saved to {OUTPUT_DIR}")
