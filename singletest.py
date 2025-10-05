# Load your model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("./goemotions-distilbert/checkpoint-8142")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Test it
text = "no drama in my team"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    
    # Get top 5 predictions instead of just top 1
    top_5_probs, top_5_indices = torch.topk(probabilities, 5, dim=-1)

emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

print(f"Text: {text}")
print("\nTop 5 Predictions:")
print("=" * 40)

for i in range(5):
    emotion = emotion_labels[top_5_indices[0][i]]
    confidence = top_5_probs[0][i].item()
    print(f"{i+1}. {emotion}: {confidence:.4f} ({confidence*100:.2f}%)")