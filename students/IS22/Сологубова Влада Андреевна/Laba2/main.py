import asyncio
import logging
import sys
from handlers import dp
from utils.loader import bot
from utils.database import create_tables

async def main():
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    create_tables()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())