from typing import Dict, List, Tuple, Union

from langchain.prompts.chat import ChatMessagePromptTemplate
from pydantic import BaseModel, Field

from configs import log_verbose, logger


class History(BaseModel):
    """
    Lịch sử trò chuyện
    Có thể tạo từ dict, ví dụ
    h = History(**{"role":"user","content":"Xin chào"})
    Hoặc có thể chuyển thành tuple, ví dụ
    h.to_msy_tuple = ("human", "Xin chào")
    """
    role: str = Field(...)
    content: str = Field(...)

    def to_msg_tuple(self):
        return "ai" if self.role=="assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate:
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        if is_raw: # Hiện tại, tin nhắn lịch sử mặc định đều là văn bản không có input_variable.
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content

        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )

    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        if isinstance(h, (list,tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict):
            h = cls(**h)

        return h
