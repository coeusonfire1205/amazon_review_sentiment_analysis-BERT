import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/pp_7817.csv")

# Ensure correct text column
if "reviews.text" in df.columns:
    df.rename(columns={"reviews.text": "text"}, inplace=True)

df["text"] = df["text"].astype(str)

# -----------------------------
# 2. FIX LABELS (MOST IMPORTANT)
# -----------------------------
# If labels are strings, map them to integers
if df["label"].dtype == object:
    df["label"] = df["label"].str.lower().map({
        "negative": 0,
        "neutral": 1,
        "positive": 2
    })

# Drop invalid rows
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Keep only required columns
df = df[["text", "label"]]

print("Final Columns Used:", df.columns)
print("Label distribution:\n", df["label"].value_counts())

# -----------------------------
# 3. Train / Validation split
# -----------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -----------------------------
# 4. Convert to HuggingFace Dataset
# -----------------------------
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
val_dataset = Dataset.from_pandas(val_df, preserve_index=False)

# -----------------------------
# 5. Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

# Rename label → labels (REQUIRED)
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

# 🔥 EXPLICIT tensor columns (CRITICAL)
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# -----------------------------
# 6. Load BERT model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

# -----------------------------
# 7. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    remove_unused_columns=False
)

# -----------------------------
# 8. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# -----------------------------
# 9. Train & Save
# -----------------------------
trainer.train()

trainer.save_model("./model")
tokenizer.save_pretrained("./model")

print("✅ Training complete. Model saved to ./model")
