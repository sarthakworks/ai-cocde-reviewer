import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# Model name
MODEL_NAME = "Qwen/QwQ-32B"

def download_model(model_name):
    """Download and load the model with caching."""
    print(f"Downloading model: {model_name}...")
    model_path = snapshot_download(repo_id=model_name, allow_patterns=["*.bin", "*.json", "*.txt"])
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Load model
model, tokenizer = download_model(MODEL_NAME)

def review_code(file_path):
    """Analyze the given code file using Qwen/QwQ-32B."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Prepare prompt
    prompt = f"Analyze the following code and provide a detailed review:\n\n{code}\n\nReview:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=4096)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move to GPU

    # Generate review
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300)
    
    review = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"Review for {file_path}:\n{review}\n")

def main():
    if len(sys.argv) < 2:
        print("Error: No files provided.")
        return

    changed_files = json.loads(sys.argv[1])

    for file in changed_files:
        review_code(file)

if __name__ == "__main__":
    main()
