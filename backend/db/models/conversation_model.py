from sqlalchemy import JSON, Column, DateTime, Integer, String, func

from backend.db.base import Base


class ConversationModel(Base):
    """
    Mô hình bản ghi cuộc trò chuyện
    """
    __tablename__ = 'conversation'
    id = Column(String(32), primary_key=True, comment='ID Cuộc trò chuyện')
    name = Column(String(50), comment='Tên Cuộc trò chuyện')
    # chat/agent_chat và các loại khác
    chat_type = Column(String(50), comment='Loại Cuộc trò chuyện')
    create_time = Column(DateTime, default=func.now(), comment='Thời gian Tạo')

    def __repr__(self):
        return f"<Conversation(id='{self.id}', name='{self.name}', chat_type='{self.chat_type}', create_time='{self.create_time}')>"
