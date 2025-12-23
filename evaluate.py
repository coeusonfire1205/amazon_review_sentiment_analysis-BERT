import pandas as pd
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load dataset
df = pd.read_csv("data/pp_7817.csv").sample(100, random_state=42)
#fix the colomn name and taking the first only
#the column review.text and reviewText both are same making this just choosing single one
if "reviews.text" in df.columns:
    text_col = "reviews.text"
else:
    raise ValueError("No text column found")

print(f"Using text column: {text_col}")
#fixing the label column
if df["label"].dtype == object:
    df["label"] = df["label"].str.lower().map({
        "negative": 0,
        "neutral": 1,
        "positive": 2
    })
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("./model")
model.eval()

y_true = df["label"].tolist()
y_pred = []

for text in df[text_col]:
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    y_pred.append(torch.argmax(outputs.logits, dim=1).item())
print(classification_report(
    y_true,
    y_pred,
    target_names=["Negative", "Neutral", "Positive"]
))
