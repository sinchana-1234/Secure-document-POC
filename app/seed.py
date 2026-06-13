"""
Seed the first admin. Run once: `python -m app.seed`
Registration is admin-only, so we need ONE bootstrap admin before anyone can log in.
"""
from app.database import SessionLocal, Base, engine
from app.models import User, Role
from app.core.security import hash_password

DEFAULT_ADMIN_EMAIL = "admin@docintel.local"
DEFAULT_ADMIN_PASSWORD = "Admin@123"   # CHANGE after first login


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == DEFAULT_ADMIN_EMAIL).first():
            print(f"Admin already exists: {DEFAULT_ADMIN_EMAIL}")
            return
        admin = User(
            email=DEFAULT_ADMIN_EMAIL, hashed_password=hash_password(DEFAULT_ADMIN_PASSWORD),
            full_name="Platform Admin", role=Role.admin, department="IT",
        )
        db.add(admin)
        db.commit()
        print(f"Created admin: {DEFAULT_ADMIN_EMAIL} / {DEFAULT_ADMIN_PASSWORD}")
        print(">>> Change this password after first login.")
    finally:
        db.close()


if __name__ == "__main__":
    main()