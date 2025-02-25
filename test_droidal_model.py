from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer from the local directory
model = AutoModelForCausalLM.from_pretrained("droidal_finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("droidal_finetuned_model")

# Ensure the pad token is set correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id


# Test the model with a descriptive prompt
# prompt = "What is the role of BOTs in Droidal’s automation strategy?"
# prompt= "Who is the CEO of Droidal?"
prompt="What is the primary mission of Droidal in the healthcare industry?"
# prompt="Inger Sivanthi is chief executive officer in Droidal"
# prompt="What are the specific benefits of using Droidal for automating healthcare administrative tasks?"
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=350,  # Increase max_length to get more meaningful responses
    temperature=0.3,  # Adjust temperature for less randomness
    top_k=50,  # Limit token candidates
    top_p=0.85,  # Nucleus sampling to ensure diversity
    repetition_penalty=5.0,  # Penalize repetitive patterns
    do_sample=True,
)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")

