#Import from HuggingFace
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Load dataset - GoEmotions
dataset = load_dataset("go_emotions", "simplified")

# # Load tokenizer + model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=28  # 27 emotions + neutral
)

emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


#Trainable parameters in the model 
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
#print(print_number_of_trainable_model_parameters(base_model))


def perform_baseline_inference(dataset, sample_index=20):
    """Perform baseline inference on a test sample"""
    
    # Get sample from test set
    sample = dataset['test'][sample_index]
    text = sample['text']  # Adjust field name based on your dataset structure
    
    # text = "Good news everyone!"
    # print("This Here:", text)
    
    true_labels = sample['labels']  # This might be 'emotions' or 'label' depending on your dataset
    
    print(f"Sample Index: {sample_index}")
    print(f"Input Text: {text}")
    print("="*80)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Perform inference
    base_model.eval()
    with torch.no_grad():
        outputs = base_model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get top predictions
        top_k = 5  # Show top 5 emotions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
        
    # Display results
    print("True Labels:")
    if isinstance(true_labels, list):
        if isinstance(true_labels[0], int):
            # Single-label case
            print(f"  - {emotion_labels[true_labels[0]]}")
        else:
            # Multi-label case - show all true emotions
            for label_idx in true_labels:
                if isinstance(label_idx, int):
                    print(f"  - {emotion_labels[label_idx]}")
                else:
                    # Binary vector case
                    for i, val in enumerate(label_idx):
                        if val == 1:
                            print(f"  - {emotion_labels[i]}")
    else:
        # Single integer label
        print(f"  - {emotion_labels[int(true_labels)]}")
    
    
    
    print("\nBaseline Model Predictions (Top 5):")
    for i in range(top_k):
        emotion = emotion_labels[top_indices[0][i]]
        confidence = top_probs[0][i].item()
        print(f"  {i+1}. {emotion}: {confidence:.4f} ({confidence*100:.2f}%)")
    
    print("="*80)
    
    return {
        'text': text,
        'true_labels': true_labels,
        'predictions': top_indices[0].tolist(),
        'probabilities': top_probs[0].tolist()
    }

result = perform_baseline_inference(dataset, sample_index=20)
print(result)

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    
    # Fix: Extract integer from list
    tokenized['labels'] = [label[0] if isinstance(label, list) else label for label in examples['labels']]
    
    return tokenized



print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

print("Sample labels:", dataset['train'][0]['labels'])
print("Type:", type(dataset['train'][0]['labels']))
print("Fixed labels:", tokenized_datasets['train'][0]['labels'])


# Set up training argument
training_args = TrainingArguments(
    output_dir="./goemotions-distilbert",
    eval_strategy="epoch",
    save_strategy="epoch",           # Changed from "steps" to "epoch"
    logging_dir="./logs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    resume_from_checkpoint=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    report_to=None,
)

# Set up data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Define metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # For single-label classification
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }



# Create trainer
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],  # or "test"
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
print("Starting training...")
trainer.train()



trainer.save_model("./goemotions-distilbert-final")
tokenizer.save_pretrained("./goemotions-distilbert-final")
print("Model saved!")


