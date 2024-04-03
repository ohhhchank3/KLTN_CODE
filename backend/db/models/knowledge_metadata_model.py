from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Integer,
                        String, func)

from backend.db.base import Base


class SummaryChunkModel(Base):
    """
    Mô hình tóm tắt chunk, được sử dụng để lưu trữ các đoạn chunk của mỗi doc_id trong file_doc,
    Nguồn dữ liệu:
        Đầu vào của người dùng: Người dùng tải lên tập tin, có thể điền mô tả về tập tin, tạo ra doc_id trong file_doc, lưu vào summary_chunk
        Tự động chia nhỏ bởi chương trình: Từ thông tin trang trong trường meta_data của bảng file_doc, chia nhỏ theo trang, tạo văn bản tóm tắt từ prompt tự động, lưu doc_id tương ứng vào summary_chunk
    Công việc tiếp theo:
        Xây dựng cơ sở dữ liệu vector: Tạo chỉ mục cho summary_context trong bảng summary_chunk, xây dựng cơ sở dữ liệu vector, meta_data là dữ liệu siêu (doc_ids)
        Liên kết ngữ nghĩa: Dựa trên mô tả do người dùng nhập vào, tóm tắt tự động được chia nhỏ, tính toán độ tương đồng ngữ nghĩa

    """
    __tablename__ = 'summary_chunk'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID')
    kb_name = Column(String(50), comment='Tên cơ sở kiến thức')
    summary_context = Column(String(255), comment='Văn bản tóm tắt')
    summary_id = Column(String(255), comment='ID Vector tóm tắt')
    doc_ids = Column(String(1024), comment="Danh sách liên kết ID vector")
    meta_data = Column(JSON, default={})

    def __repr__(self):
        return (f"<SummaryChunk(id='{self.id}', kb_name='{self.kb_name}', summary_context='{self.summary_context}',"
                f" doc_ids='{self.doc_ids}', metadata='{self.metadata}')>")
