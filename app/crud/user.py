from app.core import storage
from app.models.document import Document
from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user import UserCreate
from app.core.security import hash_password
from sqlalchemy.exc import IntegrityError


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, data: UserCreate) -> User | None:
    new_user = User(email=data.email, hashed_password=hash_password(data.password))
    db.add(new_user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return None
    db.refresh(new_user)
    return new_user

def delete_user(db: Session, user: User) -> None:
    stored_names = [
        name
        for (name,) in db.query(Document.stored_name)
        .filter(Document.owner_id == user.id, Document.stored_name.isnot(None))
        .all()
    ]

    db.delete(user)
    db.commit()

    for name in stored_names:
        storage.delete_file(name)