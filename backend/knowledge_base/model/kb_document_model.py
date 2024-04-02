
from langchain.docstore.document import Document


class DocumentWithVSId(Document):
    """
    
Tài liệu được vector hóa
    """
    id: str = None
    score: float = 3.0
