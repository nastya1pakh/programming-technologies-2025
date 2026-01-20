import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
temperature = os.getenv("TEMP")
prompt = os.getenv("PROMPT")

client = Mistral(api_key=api_key)

chat_history = []

def get_response(text: str, client: Mistral):
    response = client.chat.complete(
        model="mistral-tiny", 
        messages=chat_history,
        temperature = temperature
    )
    keepingHistory(response.choices[0].message.content, "assistant")
    return response.choices[0].message.content

def keepingHistory(Message:str, role:str):
    global chat_history
    if(len(chat_history)==6):
        chat_history.pop(0)
    chat_history.append({"role": role, "content": Message})

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

