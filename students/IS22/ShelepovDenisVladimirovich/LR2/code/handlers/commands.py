from utils.loader import dp
import logging
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from services.database.models.user import UserBase

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        await UserBase.get_or_create_user(
            tg_id=message.from_user.id,
            username=message.from_user.username,
            full_name=message.from_user.full_name
        )
        await message.answer(f"Привет, {message.from_user.full_name}, я твой бот-ассистент! Можешь задавать мне вопросы, и я буду отвечать на них. \
            Пожалуйста, помни про свой баланс на счету аккаунта в OpenAI и не ддось меня без необходимости)")
    except Exception as e:
        logging.error(f"Error occurred: {e}")

@dp.message(Command("clear_history"))
async def reset_context(message: Message):
    user_id = (await UserBase.get_user(message.from_user.id)).id
    
    isUserHasMessages = True if(len(await UserBase.get_messages_by_user_id(user_id)) > 0) else False
    print("Is user has messages:", isUserHasMessages)
    if isUserHasMessages:
        await UserBase.del_user_messages(user_id)
        await message.answer("История диалога сброшена. Можешь начинать новый разговор!")
    else:
        await message.answer("История не была найдена. Начни новый разговор!")
