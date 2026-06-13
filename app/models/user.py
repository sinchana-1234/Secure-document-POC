import enum
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum as SAEnum, Integer, String
from sqlalchemy.orm import relationship

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

    # Cascade delete: removing a User automatically deletes their Document rows.
    # This pairs with Document.owner (back_populates="owner") and the
    # ondelete="CASCADE" on the Document.owner_id FK in models/document.py.
    # Without this, deleting a user either throws a Postgres FK integrity error
    # or leaves orphaned Document rows with a dangling owner_id.
    documents = relationship(
        "Document",
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )