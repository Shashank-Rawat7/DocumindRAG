from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user import UserCreate
from app.core.security import hash_password


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, data: UserCreate) -> User:
    new_user = User(email=data.email, hashed_password=hash_password(data.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user