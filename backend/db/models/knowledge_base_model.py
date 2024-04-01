from sqlalchemy import Column, DateTime, Integer, String, func

from backend.db.base import Base


class KnowledgeBaseModel(Base):
    """
    Mô hình cơ sở kiến thức
    """
    __tablename__ = 'knowledge_base'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID Cơ sở kiến thức')
    kb_name = Column(String(50), comment='Tên Cơ sở kiến thức')
    kb_info = Column(String(200), comment='Thông tin Cơ sở kiến thức (dành cho Agent)')
    vs_type = Column(String(50), comment='Loại vector')
    embed_model = Column(String(50), comment='Tên mô hình nhúng')
    file_count = Column(Integer, default=0, comment='Số lượng tệp')
    create_time = Column(DateTime, default=func.now(), comment='Thời gian Tạo')

    def __repr__(self):
        return f"<KnowledgeBase(id='{self.id}', kb_name='{self.kb_name}',kb_intro='{self.kb_info}', vs_type='{self.vs_type}', embed_model='{self.embed_model}', file_count='{self.file_count}', create_time='{self.create_time}')>"
