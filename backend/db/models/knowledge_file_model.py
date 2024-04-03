from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Integer,
                        String, func)

from backend.db.base import Base


class KnowledgeFileModel(Base):
    """
    Mô hình tập tin kiến thức
    """
    __tablename__ = 'knowledge_file'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID Tập tin kiến thức')
    file_name = Column(String(255), comment='Tên tập tin')
    file_ext = Column(String(10), comment='Phần mở rộng tập tin')
    kb_name = Column(String(50), comment='Tên cơ sở kiến thức')
    document_loader_name = Column(String(50), comment='Tên trình tải tài liệu')
    text_splitter_name = Column(String(50), comment='Tên trình chia văn bản')
    file_version = Column(Integer, default=1, comment='Phiên bản tập tin')
    file_mtime = Column(Float, default=0.0, comment="Thời gian sửa đổi tập tin")
    file_size = Column(Integer, default=0, comment="Kích thước tập tin")
    custom_docs = Column(Boolean, default=False, comment="Có tùy chỉnh tài liệu")
    docs_count = Column(Integer, default=0, comment="Số lượng tài liệu được chia")
    create_time = Column(DateTime, default=func.now(), comment='Thời gian Tạo')

    def __repr__(self):
        return f"<KnowledgeFile(id='{self.id}', file_name='{self.file_name}', file_ext='{self.file_ext}', kb_name='{self.kb_name}', document_loader_name='{self.document_loader_name}', text_splitter_name='{self.text_splitter_name}', file_version='{self.file_version}', create_time='{self.create_time}')>"


class FileDocModel(Base):
    """
    Mô hình tài liệu tập tin-Vector
    """
    __tablename__ = 'file_doc'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID')
    kb_name = Column(String(50), comment='Tên cơ sở kiến thức')
    file_name = Column(String(255), comment='Tên tập tin')
    doc_id = Column(String(50), comment="ID Tài liệu vector")
    meta_data = Column(JSON, default={})

    def __repr__(self):
        return f"<FileDoc(id='{self.id}', kb_name='{self.kb_name}', file_name='{self.file_name}', doc_id='{self.doc_id}', metadata='{self.meta_data}')>"
