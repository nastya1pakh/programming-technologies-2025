from utils.loader import dp
import logging
from aiogram.types import Message
from utils.gpt import get_response, client
from utils.database import add_message
from aiogram import F

@dp.message()
async def message_handler(message: Message) -> None:
    try:
        user_name = message.from_user.full_name

        if message.text:
            response = await get_response(message.from_user.id, message.text, client)
            await message.answer(user_name + ", " + response)

        elif message.photo:
            await message.answer("Я не работаю с изображениями")

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при получении ответа")
