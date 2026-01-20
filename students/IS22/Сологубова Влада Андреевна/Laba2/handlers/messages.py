from utils.loader import dp
import logging
from aiogram.types import Message, ContentType
from utils.gpt import get_response
from utils.database import save_message

@dp.message()
async def message_handler(message: Message) -> None:
    try:
        if message.content_type == ContentType.PHOTO:
            await message.answer("Вы отправили картинку!")
            return
        
        user_id = message.from_user.id
        username = message.from_user.first_name
        text = message.text

        response = await get_response(user_id, username, text)

        await save_message(user_id, username, text, response)

        await message.answer(f"Вот ваш ответ на вопрос, {username}. {response}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при получении ответа от модели")