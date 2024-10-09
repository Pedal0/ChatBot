from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# Charger le modèle et le tokenizer fine-tunés
model_path = './fine_tuned_llama3'
model = LlamaForCausalLM.from_pretrained(model_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

template = """
Here is the conversation history: {context}

Question: {question}

Answer:
"""

def generate_response(context, question):
    input_text = template.format(context=context, question=question)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=500)
    response = tokenizer.decode(output, skip_special_tokens=True)
    return response

def handle_conversation():
    context = ""
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        result = generate_response(context, user_input)
        print("Bot:", result)
        context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == "__main__":
    handle_conversation()
