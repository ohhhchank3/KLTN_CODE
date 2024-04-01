from sqlalchemy import JSON, Column, DateTime, Integer, String, func

from backend.db.base import Base


class MessageModel(Base):
    """
    Mô hình bản ghi cuộc trò chuyện
    """
    __tablename__ = 'message'
    id = Column(String(32), primary_key=True, comment='ID Bản ghi cuộc trò chuyện')
    conversation_id = Column(String(32), default=None, index=True, comment='ID Cuộc trò chuyện')
    # chat/agent_chat và các loại khác
    chat_type = Column(String(50), comment='Loại Cuộc trò chuyện')
    query = Column(String(4096), comment='Câu hỏi của người dùng')
    response = Column(String(4096), comment='Phản hồi từ mô hình')
    # Lưu trữ thông tin như ID của cơ sở kiến thức để mở rộng sau này
    meta_data = Column(JSON, default={})
    # Điểm đánh giá, điểm càng cao tức là đánh giá càng tốt
    feedback_score = Column(Integer, default=-1, comment='Điểm đánh giá từ người dùng')
    feedback_reason = Column(String(255), default="", comment='Lý do đánh giá từ người dùng')
    create_time = Column(DateTime, default=func.now(), comment='Thời gian Tạo')

    def __repr__(self):
        return f"<message(id='{self.id}', conversation_id='{self.conversation_id}', chat_type='{self.chat_type}', query='{self.query}', response='{self.response}',meta_data='{self.meta_data}',feedback_score='{self.feedback_score}',feedback_reason='{self.feedback_reason}', create_time='{self.create_time}')>"
