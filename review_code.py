import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model name
MODEL_NAME = "Qwen/QwQ-32B"

def load_model():
    """Load Qwen-32B model and tokenizer with optimized settings."""
    print(f"Loading model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

# Load Qwen model and tokenizer
model, tokenizer = load_model()

def review_code(file_path):
    """Analyze the given code file using Qwen-32B."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Construct a structured chat prompt for better results
    messages = [
        {"role": "system", "content": "You are a senior software architect. Your task is to review code and provide improvements."},
        {"role": "user", "content": f"Please review the following code and provide improvements:\n\n{code}\n\nGive suggestions on readability, performance, and potential bugs."}
    ]

    # Convert to chat format
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize input
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024  # Limit token output for practical review
        )

    # Extract only the newly generated response
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"\n🔍 Review for {file_path}:\n{response}\n")

def main():
    """Main function to process changed files."""
    if len(sys.argv) < 2:
        print("Error: No files provided.")
        return

    changed_files = json.loads(sys.argv[1])  # Load list of changed files

    for file in changed_files:
        review_code(file)

if __name__ == "__main__":
    main()
