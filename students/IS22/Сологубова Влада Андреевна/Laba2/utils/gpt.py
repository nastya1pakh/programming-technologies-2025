import aiohttp
import logging
from config import SYSTEM_PROMPT
from utils.database import get_history

async def get_response(user_id: int, username: str, user_message: str) -> str:
    try:
        history = await get_history(user_id)

        prompt = SYSTEM_PROMPT + "\n\nИстория диалога:\n"

        for msg, resp in history:
            prompt += f"Пользователь: {msg}\n"
            prompt += f"Ассистент: {resp}\n"

        prompt += f"\nПользователь: {user_message}\nАссистент:"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3:4b",
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                data = await response.json()
                return data["response"]

    except Exception as e:
        logging.error(f"Error: {e}")
        return "Ошибка при обращении к локальной модели"