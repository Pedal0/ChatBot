from langchain_ollama import OllamaLLM

model = OllamaLLM(model='./fine_tuned_llama3-1B')

result = model.invoke(input="hello world")

print(result)