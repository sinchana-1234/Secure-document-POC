import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Enum as SAEnum
from app.database import Base


class Role(str, enum.Enum):
    admin = "admin"
    user = "user"


class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name       = Column(String(255), nullable=True)
    role            = Column(SAEnum(Role), nullable=False, default=Role.user)
    created_at      = Column(DateTime, default=datetime.utcnow)