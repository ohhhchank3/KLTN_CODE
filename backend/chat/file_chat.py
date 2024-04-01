import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterable, List, Optional

from fastapi import Body, File, Form, UploadFile
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from sse_starlette.sse import EventSourceResponse

from backend.chat.utils import History
from backend.knowledge_base.kb_cache.faiss_cache import memo_faiss_pool
from backend.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from backend.knowledge_base.utils import KnowledgeFile
from backend.utils import (BaseResponse, get_ChatOpenAI, get_prompt_template,
                           get_temp_dir, run_in_thread_pool, wrap_done)
from configs import (CHUNK_SIZE, LLM_MODELS, OVERLAP_SIZE, SCORE_THRESHOLD,
                     TEMPERATURE, VECTOR_SEARCH_TOP_K, ZH_TITLE_ENHANCE)


def _parse_files_in_thread(
    files: List[UploadFile],
    dir: str,
    zh_title_enhance: bool,
    chunk_size: int,
    chunk_overlap: int,
):
    """
    Sử dụng luồng đa nhiệm để lưu các tệp được tải lên vào thư mục tương ứng.
    Trả về kết quả lưu: [success hoặc error, tên tệp, msg, docs]
    """
    def parse_file(file: UploadFile) -> dict:
        '''
        Lưu một tệp.
        '''
        try:
            filename = file.filename
            file_path = os.path.join(dir, filename)
            file_content = file.file.read()  # Đọc nội dung của tệp được tải lên

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            kb_file = KnowledgeFile(filename=filename, knowledge_base_name="temp")
            kb_file.filepath = file_path
            docs = kb_file.file2text(zh_title_enhance=zh_title_enhance,
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap)
            return True, filename, f"Đã tải lên tệp {filename} thành công", docs
        except Exception as e:
            msg = f"Tải lên tệp {filename} thất bại, lỗi: {e}"
            return False, filename, msg, []

    params = [{"file": file} for file in files]
    for result in run_in_thread_pool(parse_file, params=params):
        yield result


def upload_temp_docs(
    files: List[UploadFile] = File(..., description="Tải lên tệp, hỗ trợ tải lên nhiều tệp"),
    prev_id: str = Form(None, description="ID của cơ sở kiến thức trước đó"),
    chunk_size: int = Form(CHUNK_SIZE, description="Độ dài tối đa của một đoạn văn trong cơ sở kiến thức"),
    chunk_overlap: int = Form(OVERLAP_SIZE, description="Chiều dài trùng lắp giữa các đoạn văn kế tiếp"),
    zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="Cho phép tăng cường tiêu đề tiếng Trung"),
) -> BaseResponse:
    '''
    Lưu các tệp vào thư mục tạm và thực hiện vector hóa.
    Trả về tên thư mục tạm làm ID, cũng là ID của cơ sở kiến thức tạm thời.
    '''
    if prev_id is not None:
        memo_faiss_pool.pop(prev_id)

    failed_files = []
    documents = []
    path, id = get_temp_dir(prev_id)
    for success, file, msg, docs in _parse_files_in_thread(files=files,
                                                        dir=path,
                                                        zh_title_enhance=zh_title_enhance,
                                                        chunk_size=chunk_size,
                                                        chunk_overlap=chunk_overlap):
        if success:
            documents += docs
        else:
            failed_files.append({file: msg})

    with memo_faiss_pool.load_vector_store(id).acquire() as vs:
        vs.add_documents(documents)
    return BaseResponse(data={"id": id, "failed_files": failed_files})


async def file_chat(query: str = Body(..., description="Đầu vào từ người dùng", examples=["Xin chào"]),
                    knowledge_id: str = Body(..., description="ID của cơ sở kiến thức tạm"),
                    top_k: int = Body(VECTOR_SEARCH_TOP_K, description="Số lượng vector tìm kiếm tối đa"),
                    score_threshold: float = Body(SCORE_THRESHOLD, description="Ngưỡng liên quan của cơ sở kiến thức"),
                    history: List[History] = Body([],
                                                description="Lịch sử trò chuyện",
                                                examples=[[
                                                    {"role": "user",
                                                    "content": "Chúng ta hãy chơi đố vui thành ngữ, tôi sẽ đầu tiên, sinh long hổ bạo"},
                                                    {"role": "assistant",
                                                    "content": "Hổ đầu hổ đuôi"}]]
                                                ),
                    stream: bool = Body(False, description="Xuất dữ liệu theo luồng"),
                    model_name: str = Body(LLM_MODELS[0], description="Tên của mô hình LLM"),
                    temperature: float = Body(TEMPERATURE, description="Nhiệt độ mẫu của LLM", ge=0.0, le=1.0),
                    max_tokens: Optional[int] = Body(None, description="Số lượng token tối đa"),
                    prompt_name: str = Body("default", description="Tên của mẫu prompt"),
                ):
    if knowledge_id not in memo_faiss_pool.keys():
        return BaseResponse(code=404, msg=f"Không tìm thấy cơ sở kiến thức tạm thời {knowledge_id}, vui lòng tải lên tệp trước")

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
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
        embed_func = EmbeddingsFunAdapter()
        embeddings = await embed_func.aembed_query(query)
        with memo_faiss_pool.acquire(knowledge_id) as vs:
            docs = vs.similarity_search_with_score_by_vector(embeddings, k=top_k, score_threshold=score_threshold)
            docs = [x[0] for x in docs]

        context = "\n".join([doc.page_content for doc in docs])
        if len(docs) == 0:
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            text = f"""Nguồn [{inum + 1}] [{filename}] \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:
            source_documents.append(f"""<span style='color:red'>Không tìm thấy tài liệu liên quan, trả lời dựa trên khả năng của mô hình lớn!</span>""")

        if stream:
            async for token in callback.aiter():
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

    return EventSourceResponse(knowledge_base_chat_iterator())
