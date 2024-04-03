from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String


class BaseModel:
    """
    Mô hình cơ bản
    """
    id = Column(Integer, primary_key=True, index=True, comment="ID chính")
    create_time = Column(DateTime, default=datetime.utcnow, comment="Thời gian tạo")
    update_time = Column(DateTime, default=None, onupdate=datetime.utcnow, comment="Thời gian cập nhật")
    create_by = Column(String, default=None, comment="Người tạo")
    update_by = Column(String, default=None, comment="Người cập nhật")
