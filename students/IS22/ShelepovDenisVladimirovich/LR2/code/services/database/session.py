from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from connection import engine

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)