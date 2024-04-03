import uuid

from backend.db.models.conversation_model import ConversationModel
from backend.db.session import with_session


@with_session
def add_conversation_to_db(session, chat_type, name="", conversation_id=None):
    """
    Thêm bản ghi trò chuyện mới
    """
    if not conversation_id:
        conversation_id = uuid.uuid4().hex
    c = ConversationModel(id=conversation_id, chat_type=chat_type, name=name)

    session.add(c)
    return c.id
