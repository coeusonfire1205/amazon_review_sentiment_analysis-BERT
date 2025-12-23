import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load tokenizer and trained model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("./model")

# Put model in evaluation mode
model.eval()

# Label mapping (MUST match training)
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

def predict_sentiment(text: str) -> str:
    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    # Disable gradient calculation
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return label_map[predicted_class]


# -----------------------------
# Run prediction from terminal
# -----------------------------
if __name__ == "__main__":
    review = input("Enter an Amazon review: ").strip()

    if len(review) == 0:
        print("Please enter a valid review.")
    else:
        sentiment = predict_sentiment(review)
        print("\nPredicted Sentiment:", sentiment)
