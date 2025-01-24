from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input
input_text = "The future of AI is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Get logits
outputs = model(input_ids)
logits = outputs.logits

# Modify logits (boost 'AI', penalize 'robotics')
boost_token_id = tokenizer.convert_tokens_to_ids("promising")
penalize_token_id = tokenizer.convert_tokens_to_ids("uncertain")

logits[:, :, boost_token_id] += 15.0  # Boost probability
logits[:, :, penalize_token_id] -= 5.0  # Penalize probability

eos_token_id = tokenizer.eos_token_id  # Get <|endoftext|> token ID
logits[:, :, eos_token_id] -= 5.0  # Penalize end-of-text token


# Generate text
output_ids = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
output_text = tokenizer.decode(output_ids[0])
print(output_text)


