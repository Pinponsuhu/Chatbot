from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(input_text):
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate response using the model
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    
    output = model.generate(input_ids, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2,pad_token_id=tokenizer.eos_token_id,attention_mask=attention_mask)
    
    # Decode the generated text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response.strip()

print("Chatbot: Hello! How can I help you today? (type 'exit' to stop)")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    response = generate_response(user_input)
    print(f"Chatbot: {response}")
