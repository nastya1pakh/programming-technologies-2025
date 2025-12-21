from services.database.models.base import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, select
from services.database.session import async_session

class UserBase(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    login: Mapped[str] = mapped_column(String(255), unique=True)

    @classmethod
    async def get_user(cls, tg_id: int):
        async with async_session() as session:
            result = await session.execute(
                select(cls).where(cls.login == str(tg_id))
            )
            return result.scalars().first()

    @classmethod
    async def get_or_create_user(cls, tg_id: int, username: str | None, full_name: str):
        async with async_session() as session:
            result = await session.execute(
                select(cls).where(cls.login == str(tg_id))
            )
            user = result.scalars().first()

            if not user:
                user = cls(
                    login=str(tg_id),
                    name=full_name or username or "Unknown"
                )
                session.add(user)
                await session.commit()
                await session.refresh(user)

            return user