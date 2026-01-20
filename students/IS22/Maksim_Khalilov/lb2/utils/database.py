from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import sessionmaker, declarative_base

engine = create_engine("sqlite:///messages.db")
Session = sessionmaker(bind=engine)

Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    role = Column(Text)
    content = Column(Text)


Base.metadata.create_all(engine)

def add_message(user_id: int, role: str, content: str):
    session = Session()
    msg = Message(user_id=user_id, role=role, content=content)
    session.add(msg)
    session.commit()
    session.close()

def get_messages(user_id: int, limit: int = 10):
    session = Session()
    rows = (
        session.query(Message)
        .filter(Message.user_id == user_id)
        .order_by(Message.id.desc())
        .limit(limit)
        .all()
    )
    session.close()

    return [(msg.role, msg.content) for msg in reversed(rows)]


def reset_messages(user_id: int):
    session = Session()
    session.query(Message).filter(Message.user_id == user_id).delete()
    session.commit()
    session.close()
