from fastapi import Body
from backend.db.repository import feedback_message_to_db
from backend.utils import BaseResponse

from configs import log_verbose, logger


def chat_feedback(message_id: str = Body("", max_length=32, description="ID của bản ghi chat"),
                  score: int = Body(0, max=100, description="Điểm đánh giá của người dùng, tối đa 100 điểm"),
                  reason: str = Body("", description="Lý do đánh giá của người dùng, ví dụ: không đúng sự thật")
                  ):
    try:
        feedback_message_to_db(message_id, score, reason)
    except Exception as e:
        msg = f"Đã xảy ra lỗi khi phản hồi bản ghi chat: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=200, msg=f"Đã phản hồi bản ghi chat {message_id}")
