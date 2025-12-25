from utils.loader import dp
import logging
from aiogram.filters import CommandStart
from aiogram.types import Message
from config import SYSTEM_PROMPT
from utils.database import update_reset
from aiogram.filters import Command

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        await message.answer(f"Привет, {message.from_user.full_name}, я твой бот-ассистент! Можешь задавать мне вопросы, и я буду отвечать на них.")
        await message.answer(f"Сейчас задействован системный промпт:\n{SYSTEM_PROMPT}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")

@dp.message(Command("reset_context"))
async def reset_context_handler(message: Message):
    user_id = message.from_user.id

    await update_reset(user_id)

    await message.answer("Контекст диалога сброшен! Начинаем диалог заново.")