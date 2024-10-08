from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd

data = './data/video_games_sales.csv'
video_games_sales = pd.read_csv(data)

template = """
Answer the question below. Using the dataset {video_games_sales}.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model='llama3')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handl_conversation():
    context = ""
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        result = chain.invoke({"context": context, "question": user_input})
        print("Bot:", result)
        context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == "__main__":
    handl_conversation()