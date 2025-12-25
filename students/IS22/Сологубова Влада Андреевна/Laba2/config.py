from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")