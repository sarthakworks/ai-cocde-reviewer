import sys
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM

# Load pre-trained CodeBERT model and tokenizer
model_name = "microsoft/codebert-base"  # You can change to the relevant model
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)

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

    print(f"Reviewing {file_path}...\n")
    print("Predictions (top tokens for each input token):")
    print(predictions)

    # You could also interpret the logits, but since it's a Masked Language Model, 
    # it's more suitable for filling masked tokens or predicting the next token. 
    # Here, we'll just print out the logits for now.
    print("\nLogits (Raw predictions):")
    print(logits)

    # Simple feedback based on the logits - this part can be enhanced based on your needs
    if logits.max() < 0:
        print("\nSuggestion: Check for potential issues, the model did not find strong correlations.")
    else:
        print("\nSuggestion: Code seems to align with typical patterns.")

def main():
    # The list of files passed from the GitHub Action
    changed_files = sys.argv[1].split()

    # Review each changed file
    for file in changed_files:
        review_code_in_file(file)

if __name__ == "__main__":
    main()
