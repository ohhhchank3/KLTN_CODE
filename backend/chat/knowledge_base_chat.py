import asyncio
import json
from typing import AsyncIterable, List, Optional
from urllib.parse import urlencode

from fastapi import Body, Request
from fastapi.concurrency import run_in_threadpool
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from sse_starlette.sse import EventSourceResponse

from backend.chat.utils import History
from backend.knowledge_base.kb_doc_api import search_docs
from backend.knowledge_base.kb_service.base import KBServiceFactory
from backend.reranker.reranker import LangchainReranker
from backend.utils import (BaseResponse, embedding_device, get_ChatOpenAI,
                           get_prompt_template, wrap_done)
from configs.kb_config import (CHUNK_SIZE, OVERLAP_SIZE, SCORE_THRESHOLD,
                               VECTOR_SEARCH_TOP_K, ZH_TITLE_ENHANCE)
from configs.model_config import (LLM_MODELS, MODEL_PATH, RERANKER_MAX_LENGTH,
                                  RERANKER_MODEL, TEMPERATURE, USE_RERANKER)


async def knowledge_base_chat(query: str = Body(..., description="Đầu vào từ người dùng", examples=["Xin chào"]),
                              knowledge_base_name: str = Body(..., description="Tên của cơ sở kiến thức", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="Số lượng vector tìm kiếm tối đa"),
                              score_threshold: float = Body(
                                  SCORE_THRESHOLD,
                                  description="Ngưỡng liên quan của cơ sở kiến thức, giá trị từ 0-1, giá trị nhỏ hơn SCORE tương ứng với mức độ liên quan cao hơn, giá trị 1 tương ứng với không có lọc, khuyến nghị đặt khoảng 0.5"
                              ),
                              history: List[History] = Body(
                                  [],
                                  description="Lịch sử trò chuyện",
                                  examples=[[
                                      {"role": "user",
                                       "content": "Chúng ta hãy chơi đố vui thành ngữ, tôi sẽ đầu tiên, sinh long hổ bạo"},
                                      {"role": "assistant",
                                       "content": "Hổ đầu hổ đuôi"}]]
                              ),
                              stream: bool = Body(False, description="Xuất dữ liệu theo luồng"),
                              model_name: str = Body(LLM_MODELS[0], description="Tên mô hình LLM"),
                              temperature: float = Body(TEMPERATURE, description="Nhiệt độ mẫu của LLM", ge=0.0, le=1.0),
                              max_tokens: Optional[int] = Body(
                                  None,
                                  description="Số lượng token tối đa được sinh ra bởi LLM, mặc định là giá trị tối đa của mô hình"
                              ),
                              prompt_name: str = Body(
                                  "default",
                                  description="Tên của mẫu prompt được sử dụng (được cấu hình trong configs/prompt_config.py)"
                              ),
                              request: Request = None,
                              ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Không tìm thấy cơ sở kiến thức {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(
            query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        docs = await run_in_threadpool(search_docs,
                                       query=query,
                                       knowledge_base_name=knowledge_base_name,
                                       top_k=top_k,
                                       score_threshold=score_threshold)

        # Thêm reranker
        if USE_RERANKER:
            reranker_model_path = MODEL_PATH["reranker"].get(RERANKER_MODEL,"BAAI/bge-reranker-large")
            print("-----------------đường dẫn mô hình------------------")
            print(reranker_model_path)
            reranker_model = LangchainReranker(top_n=top_k,
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path
                                            )
            print(docs)
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=query)
            print("---------sau khi rerank------------------")
            print(docs)
        context = "\n".join([doc.page_content for doc in docs])

        if len(docs) == 0:  # Nếu không tìm thấy tài liệu liên quan, sử dụng mẫu empty
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Bắt đầu một nhiệm vụ chạy ở nền.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""Nguồn [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:  # Nếu không tìm thấy tài liệu liên quan
            source_documents.append(f"<span style='color:red'>Không tìm thấy tài liệu liên quan, trả lời dựa trên khả năng của mô hình lớn!</span>")

        if stream:
            async for token in callback.aiter():
                # Sử dụng server-sent-events để xuất dữ liệu theo luồng
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name))
