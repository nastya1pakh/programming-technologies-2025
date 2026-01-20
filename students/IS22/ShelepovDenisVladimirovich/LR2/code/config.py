from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PROMPT = os.getenv("PROMPT2")
DB_NAME = os.getenv("DATABAWSE_NAME")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD")