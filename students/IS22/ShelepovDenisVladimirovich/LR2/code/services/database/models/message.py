from services.database.models.base import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, ForeignKey, Text, select, delete
from services.database.session import async_session

class MessageBase(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    message: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(100))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    @classmethod
    async def get_messages_by_user_id(cls, user_id: int) -> list[dict]:
        async with async_session() as session:
            result = await session.execute(
                select(cls)
                .where(cls.user_id == user_id)
                .order_by(cls.id)
            )

            return [
                {"role": msg.role, "content": msg.message}
                for msg in result.scalars().all()
            ]
        
    @classmethod
    async def save_message(cls, user_id: int, role: str, content: str) -> None:
        async with async_session.begin() as session:
            session.add(
                cls(
                    user_id=user_id,
                    role=role,
                    message=content
                )
            )

    @classmethod
    async def delete_by_user_id(cls, user_id: int) -> None:
        async with async_session.begin() as session:
            await session.execute(
                delete(cls).where(cls.user_id == user_id)
            )