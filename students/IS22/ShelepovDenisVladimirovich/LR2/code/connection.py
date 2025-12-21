from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from services.database.models.base import Base
from services.database.models.user import UserBase
from services.database.models.message import MessageBase

DB_URL = f'postgresql+asyncpg://postgres:12345@localhost:5432/postgres'
engine:AsyncEngine = create_async_engine(DB_URL, echo=True)

def create_db_and_tables() -> None:
	Base.metadata.create_all(engine)
	
if __name__ == "__main__":
    create_db_and_tables()