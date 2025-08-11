from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Check if a CUDA-enabled GPU is available and set the device accordingly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Move the model to the chosen device (GPU if available)
model.to(device)

# Input text
text = "This movie is terrible."

# Tokenize the input and convert to PyTorch tensors
inputs = tokenizer(text, return_tensors="pt")

# Move the input tensors to the chosen device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Get the model's output
outputs = model(**inputs)

# Access the logits
logits = outputs.logits

print(f"Logits: {logits}")

# Apply softmax to get probabilities
probabilities = torch.softmax(logits, dim=-1)
print(f"Probabilities: {probabilities}")

# Get the predicted class (index with the highest probability)
predicted_class_id = torch.argmax(probabilities).item()

# Class labels for this model
class_labels = {0: 'NEGATIVE', 1: 'POSITIVE'}
predicted_label = class_labels[predicted_class_id]

print(f"Predicted class ID: {predicted_class_id}")
print(f"Predicted label: {predicted_label}")
