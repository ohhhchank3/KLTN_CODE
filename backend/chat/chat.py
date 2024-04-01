import asyncio
import json
from typing import AsyncIterable, List, Optional, Union

from fastapi import Body
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from sse_starlette.sse import EventSourceResponse

from backend.callback_handler.conversation_callback_handler import \
    ConversationCallbackHandler
from backend.chat.utils import History
from backend.db.repository import add_message_to_db
from backend.memory.conversation_db_buffer_memory import \
    ConversationBufferDBMemory
from backend.utils import get_ChatOpenAI, get_prompt_template, wrap_done
from configs.model_config import LLM_MODELS, TEMPERATURE


async def chat(query: str = Body(..., description="Người dùng nhập", examples=["Cô đơn"]),
               conversation_id: str = Body("", description="ID cuộc trò chuyện"),
               history_len: int = Body(-1, description="Số lượng tin nhắn lịch sử được lấy từ cơ sở dữ liệu"),
               history: Union[int, List[History]] = Body([],
                                                         description="Lịch sử trò chuyện. Sử dụng số nguyên để lấy từ cơ sở dữ liệu",
                                                         examples=[[
                                                             {"role": "user",
                                                              "content": "Chúng ta hãy chơi trò chơi dựa trên thành ngữ, tôi sẽ bắt đầu: Đất nước trời đất"},
                                                             {"role": "assistant", "content": "Nơi đất, ở trời"}]]
                                                         ),
               stream: bool = Body(False, description="Xuất dữ liệu theo dòng"),
               model_name: str = Body(LLM_MODELS[0], description="Tên mô hình LLM."),
               temperature: float = Body(TEMPERATURE, description="Nhiệt độ LLM", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="Số lượng token LLM tối đa"),
               prompt_name: str = Body("default", description="Tên mẫu prompt được sử dụng (được cấu hình trong configs/prompt_config.py)"),
               ):
    async def chat_iterator() -> AsyncIterable[str]:
        nonlocal history, max_tokens
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        memory = None

        # Lưu phản hồi từ LLM vào cơ sở dữ liệu tin nhắn
        message_id = add_message_to_db(chat_type="llm_chat", query=query, conversation_id=conversation_id)
        conversation_callback = ConversationCallbackHandler(conversation_id=conversation_id, message_id=message_id,
                                                            chat_type="llm_chat",
                                                            query=query)
        callbacks.append(conversation_callback)

        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

        if history:  # Ưu tiên sử dụng lịch sử trò chuyện được truyền từ front-end
            history = [History.from_data(h) for h in history]
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])
        elif conversation_id and history_len > 0:  # Lấy lịch sử trò chuyện từ cơ sở dữ liệu theo yêu cầu của front-end
            # Mẫu prompt phải chứa biến memory.memory_key khi sử dụng memory
            prompt = get_prompt_template("llm_chat", "with_history")
            chat_prompt = PromptTemplate.from_template(prompt)
            # Lấy danh sách tin nhắn dựa trên conversation_id để tạo memory
            memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                                llm=model,
                                                message_limit=history_len)
        else:
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        # Bắt đầu một nhiệm vụ chạy nền.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Sử dụng server-sent-events để xuất dữ liệu theo dòng
                yield json.dumps(
                    {"text": token, "message_id": message_id},
                    ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "message_id": message_id},
                ensure_ascii=False)

        await task

    return EventSourceResponse(chat_iterator())
