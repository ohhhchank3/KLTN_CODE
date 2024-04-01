from contextlib import contextmanager
from functools import wraps
from typing import Generator, Type

from sqlalchemy.orm import Session

# Giả định rằng SessionLocal là một class
SessionLocal: Type = ...

@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager được sử dụng để tự động lấy Session và tránh lỗi.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def with_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with session_scope() as session:
            try:
                result = f(session, *args, **kwargs)
                session.commit()
                return result
            except:
                session.rollback()
                raise

    return wrapper

@contextmanager
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db0() -> Type:
    """
    Lấy SessionLocal từ context manager
    """
    db = SessionLocal()
    return db
