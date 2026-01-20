from mistralai import Mistral
from config import MISTRAL_API_KEY, PROMPT
from services.database.models.message import MessageBase
import logging

client = Mistral(api_key=MISTRAL_API_KEY)

async def get_response(
    user_message: str,
    user_id: int,
    client_full_name: str
) -> str:
    try:
        history = await MessageBase.get_messages_by_user_id(user_id)

        messages = [
            {"role": "system", "content": f"{PROMPT} {client_full_name}".strip()}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        print("Messages sent to Mistral:", messages)
        response = await client.chat.complete_async(
            model="mistral-small-2506",
            messages=messages
        )
        
        assistant_reply = response.choices[0].message.content
        
        await MessageBase.save_message(user_id, "user", user_message)
        await MessageBase.save_message(user_id, "assistant", assistant_reply)

        return assistant_reply

    except Exception as e:
        logging.error(f"Mistral API error: {e}")
        return "Извини, сейчас не могу ответить. Попробуй позже."