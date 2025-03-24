import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from transformers import LlamaTokenizer, LlamaForCausalLM

def download_model_with_retries(model_name, retries=3, delay=60):
    """
    Attempt to download the model with retry logic.
    """
    for attempt in range(retries):
        try:
            # Download model
            print(f"Attempting to download {model_name} (Attempt {attempt + 1})...")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            print(f"Download failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Could not download the model.")
                raise

# Set model name
model_name = "codellama/CodeLlama-7b-hf"

# Attempt to download the model with retries
model, tokenizer = download_model_with_retries(model_name)

def review_code_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # Tokenize the input code
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)

    # Perform inference (forward pass)
    with torch.no_grad():
        outputs = model(**inputs)

    # Analyze the logits (these are the raw outputs from the model)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    print(f"logits {logits}...\n")
    print(f"predictions {predictions}...\n")

    # Decode the predictions back to readable tokens
    decoded_predictions = tokenizer.decode(predictions[0], skip_special_tokens=True)

    print(f"Reviewing {file_path}...\n")
    print(f"Decoded Predictions: {decoded_predictions}\n")
    
    # Suggest improvements based on the logits or predictions
    if logits.max() < 0:
        print("\nSuggestion: Check for potential issues, the model did not find strong correlations.")
    else:
        print("\nSuggestion: Code seems to align with typical patterns.")

def main():
    # The list of files passed from the GitHub Action
    if len(sys.argv) < 2:
        print("Error: No files provided.")
        return

    changed_files = sys.argv[1].split()

    # Review each changed file
    for file in changed_files:
        review_code_in_file(file)

if __name__ == "__main__":
    main()
