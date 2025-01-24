from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Input text
input_text = "The future of AI is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Get model outputs
outputs = model(input_ids)

# Access logits
logits = outputs.logits
print("Logits shape:", logits.shape)  # Example: torch.Size([1, 6, 50257])

# Get probabilities for the next token
next_token_logits = logits[:, 0, :]  # Logits for the last token
probabilities = torch.softmax(next_token_logits, dim=-1)

# Get the most probable next token
predicted_token_id = torch.argmax(probabilities, dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print("Next token:", predicted_token)

