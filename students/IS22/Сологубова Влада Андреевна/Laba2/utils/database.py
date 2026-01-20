# import sqlite3

# conn = sqlite3.connect("messages.db")
# cursor = conn.cursor()

# cursor.execute("""
# CREATE TABLE IF NOT EXISTS messages (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     user_id INTEGER,
#     username TEXT,
#     message TEXT,
#     response TEXT,
#     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
# )
# """)

# cursor.execute("""
# CREATE TABLE IF NOT EXISTS session (
#     user_id INTEGER PRIMARY KEY,
#     last_reset TIMESTAMP
# )
# """)

# conn.commit()

# def save_message(user_id: int, username: str, message: str, response: str):
#     cursor.execute("""
#         INSERT INTO messages (user_id, username, message, response)
#         VALUES (?, ?, ?, ?)
#     """, (user_id, username, message, response))

#     conn.commit()

# def get_last_reset(user_id: int):
#     cursor.execute("""
#         SELECT last_reset FROM session WHERE user_id = ?
#     """, (user_id,))
    
#     row = cursor.fetchone()
#     return row[0] if row else None

# def update_reset(user_id: int):
#     cursor.execute("""
#         INSERT OR REPLACE INTO session (user_id, last_reset)
#         VALUES (?, CURRENT_TIMESTAMP)
#     """, (user_id,))
#     conn.commit()

# def get_history(user_id: int, limit: int = 3):
#     last_reset = get_last_reset(user_id)

#     if last_reset:
#         cursor.execute("""
#             SELECT message, response FROM messages
#             WHERE user_id = ? AND timestamp > ?
#             ORDER BY id DESC
#             LIMIT ?
#         """, (user_id, last_reset, limit))
#     else:
#         cursor.execute("""
#             SELECT message, response FROM messages
#             WHERE user_id = ?
#             ORDER BY id DESC
#             LIMIT ?
#         """, (user_id, limit))

#     return cursor.fetchall()[::-1] 

from sqlalchemy import Column, Integer, String, Text, DateTime, select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone
import asyncio

engine = create_async_engine('sqlite+aiosqlite:///messages.db', echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(AsyncAttrs, DeclarativeBase):
    pass

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer)
    username = Column(String)
    message = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Session(Base):
    __tablename__ = 'session'
    
    user_id = Column(Integer, primary_key=True)
    last_reset = Column(DateTime)

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def save_message(user_id: int, username: str, message: str, response: str):
    async with async_session() as session:
        async with session.begin():
            new_message = Message(
                user_id=user_id,
                username=username,
                message=message,
                response=response
            )
            session.add(new_message)

async def get_last_reset(user_id: int):
    async with async_session() as session:
        result = await session.execute(
            select(Session.last_reset).where(Session.user_id == user_id)
        )
        row = result.first()
        return row[0] if row else None

async def update_reset(user_id: int):
    async with async_session() as session:
        async with session.begin():
            existing = await session.get(Session, user_id)
            
            if existing:
                existing.last_reset = func.now()
            else:
                session.add(Session(user_id=user_id, last_reset=func.now()))

async def get_history(user_id: int, limit: int = 3):
    last_reset = await get_last_reset(user_id)
    print (last_reset)
    print (type(last_reset))
    async with async_session() as session:
        stmt = select(Message.message, Message.response).where(Message.user_id == user_id)
        
        if last_reset:
            stmt = stmt.where(Message.timestamp > last_reset)
        
        stmt = stmt.order_by(Message.id.desc()).limit(limit)
        result = await session.execute(stmt)
        rows = result.fetchall()
        
        return rows[::-1]