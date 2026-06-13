"""
Auth routes. Login uses OAuth2PasswordRequestForm so Swagger's Authorize button works.
Only an admin may mint another admin — self-service "role":"admin" would be a hole.
The first admin comes from the seed script.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_admin
from app.core.security import hash_password, verify_password, create_access_token
from app.database import get_db
from app.models import User, Role
from app.schemas import UserCreate, UserOut, Token

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    role = Role.admin if payload.role == "admin" else Role.user
    user = User(
        email=payload.email, hashed_password=hash_password(payload.password),
        full_name=payload.full_name, role=role, 
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    token = create_access_token(subject=str(user.id), role=user.role.value)
    return Token(access_token=token, role=user.role.value, user_id=user.id)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user