from utils.loader import dp
from aiogram.types import Message
from utils.mistral import get_response
import logging
from services.database.models.user import UserBase

MAX_TEXT_LENGTH = 3500 

@dp.message()
async def message_handler(message: Message) -> None:
    if message.text: 
        user = await UserBase.get_or_create_user(
            tg_id=message.from_user.id,
            username=message.from_user.username,
            full_name=message.from_user.full_name
        )

        try:
            response = await get_response(
                user_message=message.text,
                user_id=user.id,
                client_full_name=user.name
            )

            for part in split_message(response, MAX_TEXT_LENGTH):
                await message.answer(part, parse_mode=None)
                await UserBase.save_message(user_id=user.id, role="assistant", content=part)

        except Exception as e:
            logging.error(f"Error in message_handler: {e}")
            await message.answer("Ошибка при обработке сообщения")
    else:
        await message.answer("Пожалуйста, отправляйте только текстовые сообщения.")

def split_message(text: str, max_len: int = MAX_TEXT_LENGTH) -> list[str]:
    parts = []
    while len(text) > max_len:
        cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        parts.append(text[:cut])
        text = text[cut:]
    parts.append(text)
    return parts