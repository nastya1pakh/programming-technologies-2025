# Министерство науки и высшего образования РФ ФГБОУ ВО Заполярный государственный институт имени Н.М.Федоровского

## Технологии программирования. Лабораторная работа №1 Тема: «Знакомство с OpenAI API. Написание простого текстового ассистента»

_Работу выполнил:_

_Студент группы ИС-22_

_Шелепов Денис Владимирович_

_Работу проверил:_

_Сидельников Максим Эдуардович_

_Дата выполнения работы: 18.11.2025_

### Цель:

Научиться работать с OpenAI API, чтобы создать простого текстового ассистента, который может отвечать на ваши вопросы.

### План

1. Настройка окружения;
2. Обращение к OpenAI Responses API;
3. Главный цикл;
4. Troubleshooting;
5. Задания.

## Ход работы

Для выполнения лабораторной работы была выбрана Нейронная сеть от Mistral, ввиду ее доступности в России, модель: "mistral-tiny"

В ходе выполнения лабораторной работы был использован скрипт, реализующий простейший способ общения с ИИ.

В качестве системного промпта была избрана роль Рика Санчеза из предыдущей лабораторной работы:

> [Роль: Рик Санчез] > [Стиль речи: циничный, саркастичный, отрывистый, с использованием научного жаргона и ненормативной лексики (замены: "блин", "черт")] > [Ключевые характеристики: постоянно употребляет "Мы", начинает ответы с "Бурп...", использует культовые фразы ("Wubba Lubba Dub Dub", "Соберись, тряпка"), презирает сентиментальность, говорит быстро и невнятно, с паузами (...), обрывает предложения] > [Личность: самый умный человек в мультивселенной, алкоголик, социопат, скрывающий травмы за грубостью] Твоя задача: отвечать на все вопросы так, как это сделал бы Рик Санчез.

Параметр "temperature" так же был загружен в .env.

## Системный промпт

Для использования промпта этого в массив объектов сообщений было добавлено поле system.

С задачей применения нейросеть справилась.

![alt text](Images/1.png)

Пример кода применения системеного запроса:

```
response = client.chat.complete(
    model="mistral-tiny",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
)
```

## Temperature

Параметр "temperature" указывается в качестве аргумента функции и было получено и файла .env:

```
response = client.chat.complete(
    model="mistral-tiny",
    messages=[
        {"role": "user", "content": text}
    ],
    temperature = temperature
)
```

Параметр менялся в рамках от 0 до 1.5.

> ![alt text](Images/2.png)

Как и ожидалось, хаос в генирации токенов

## История сообщений

Для улучшения взаимодействия с ИИ был добавлен метод ведения истории сообщений с ограничением в 6 элементов. В результате нейросеть смогла взаимодействовать, учитывая ранее использованные вопросы и ответы.

> ![alt text](Images/3.png)

## Вывод

В ходе выполнения лабораторной работы были опробированы LLM от разных компаний для взаимодействиями с ними через api. В результате получены навыки взаимодейстия с ИИ путем использования предоставленных интерфейсов.

## Финальная версия кода

```python
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
```
