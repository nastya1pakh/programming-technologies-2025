import sys
from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
prompt = os.getenv("prompt")
temperature = os.getenv("temperature")
dialog_history = []
client = Mistral(api_key=api_key)

def get_response(text: str, client: Mistral):
    response = client.chat.complete(
        model="mistral-tiny", 
        messages=dialog_history,
        temperature = temperature
    )
    keepingHistory(response.choices[0].message.content, "assistant")
    return response.choices[0].message.content

def keepingHistory(Message:str, role:str):
    global dialog_history
    if(len(dialog_history)==6):
        dialog_history.pop(0)
    dialog_history.append({"role": role, "content": Message})

if __name__ == "__main__":
    print("Введите ваш вопрос (или 'exit' для выхода):")
    while True:
        question = input("Вы: ")
        if question.lower() == "exit":
            print("Завершение программы.")
            break
        keepingHistory(question, "user")
        answer = get_response(question, client)
        print("AI:", answer)


