import asyncio
import json
from typing import AsyncIterable, Dict, List, Optional

from fastapi import Body
from fastapi.concurrency import run_in_threadpool
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities.bing_search import BingSearchAPIWrapper
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from markdownify import markdownify
from backend.chat.utils import History
from backend.utils import (BaseResponse, get_ChatOpenAI, get_prompt_template,
                          wrap_done)
from sse_starlette import EventSourceResponse
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

from configs import (BING_SEARCH_URL, BING_SUBSCRIPTION_KEY, LLM_MODELS,
                     METAPHOR_API_KEY, OVERLAP_SIZE, SEARCH_ENGINE_TOP_K,
                     TEMPERATURE)


def bing_search(text, result_len=SEARCH_ENGINE_TOP_K, **kwargs):
    if not (BING_SEARCH_URL and BING_SUBSCRIPTION_KEY):
        return [{"snippet": "Vui lòng đặt BING_SUBSCRIPTION_KEY và BING_SEARCH_URL trong môi trường OS",
                 "title": "Không tìm thấy thông tin môi trường",
                 "link": "https://python.langchain.com/en/latest/modules/agents/tools/examples/bing_search.html"}]
    search = BingSearchAPIWrapper(bing_subscription_key=BING_SUBSCRIPTION_KEY,
                                  bing_search_url=BING_SEARCH_URL)
    return search.results(text, result_len)


def duckduckgo_search(text, result_len=SEARCH_ENGINE_TOP_K, **kwargs):
    search = DuckDuckGoSearchAPIWrapper()
    return search.results(text, result_len)


def metaphor_search(
        text: str,
        result_len: int = SEARCH_ENGINE_TOP_K,
        split_result: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = OVERLAP_SIZE,
) -> List[Dict]:
    from metaphor_python import Metaphor

    if not METAPHOR_API_KEY:
        return []

    client = Metaphor(METAPHOR_API_KEY)
    search = client.search(text, num_results=result_len, use_autoprompt=True)
    contents = search.get_contents().contents
    for x in contents:
        x.extract = markdownify(x.extract)

    # Các nội dung được trả về từ metaphor đều là văn bản dài, cần phải cắt thành các đoạn rồi tìm kiếm
    if split_result:
        docs = [Document(page_content=x.extract,
                         metadata={"link": x.url, "title": x.title})
                for x in contents]
        text_splitter = RecursiveCharacterTextSplitter(["\n\n", "\n", ".", " "],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        splitted_docs = text_splitter.split_documents(docs)

        # Đặt các tài liệu đã cắt vào cơ sở dữ liệu vector tạm thời, sau đó chọn ra TOP_K tài liệu
        if len(splitted_docs) > result_len:
            normal = NormalizedLevenshtein()
            for x in splitted_docs:
                x.metadata["score"] = normal.similarity(text, x.page_content)
            splitted_docs.sort(key=lambda x: x.metadata["score"], reverse=True)
            splitted_docs = splitted_docs[:result_len]

        docs = [{"snippet": x.page_content,
                 "link": x.metadata["link"],
                 "title": x.metadata["title"]}
                for x in splitted_docs]
    else:
        docs = [{"snippet": x.extract,
                 "link": x.url,
                 "title": x.title}
                for x in contents]

    return docs


SEARCH_ENGINES = {"bing": bing_search,
                  "duckduckgo": duckduckgo_search,
                  "metaphor": metaphor_search,
                  }


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


async def lookup_search_engine(
        query: str,
        search_engine_name: str,
        top_k: int = SEARCH_ENGINE_TOP_K,
        split_result: bool = False,
):
    search_engine = SEARCH_ENGINES[search_engine_name]
    results = await run_in_threadpool(search_engine, query, result_len=top_k, split_result=split_result)
    docs = search_result2docs(results)
    return docs

async def search_engine_chat(query: str = Body(..., description="Đầu vào từ người dùng", examples=["Xin chào"]),
                             search_engine_name: str = Body(..., description="Tên công cụ tìm kiếm", examples=["duckduckgo"]),
                             top_k: int = Body(SEARCH_ENGINE_TOP_K, description="Số lượng kết quả tìm kiếm"),
                             history: List[History] = Body([],
                                                           description="Lịch sử trò chuyện",
                                                           examples=[[
                                                               {"role": "user",
                                                                "content": "Chúng ta chơi câu đối, tôi đầu tiên, từ ngữ là 'sống bóng'"},
                                                               {"role": "assistant",
                                                                "content": "Hình như từ 'nghèo' phù hợp"}]]
                                                           ),
                             stream: bool = Body(False, description="Xuất dữ liệu theo luồng"),
                             model_name: str = Body(LLM_MODELS[0], description="Tên mô hình LLM"),
                             temperature: float = Body(TEMPERATURE, description="Nhiệt độ lấy mẫu của LLM", ge=0.0, le=1.0),
                             max_tokens: Optional[int] = Body(None,
                                                              description="Giới hạn số lượng token LLM tạo ra, mặc định là giá trị lớn nhất của mô hình"),
                             prompt_name: str = Body("default",
                                                     description="Tên mẫu prompt sử dụng"),
                             split_result: bool = Body(False,
                                                       description="Phân tách kết quả tìm kiếm (chủ yếu dành cho công cụ tìm kiếm metaphor)")
                             ):
    if search_engine_name not in SEARCH_ENGINES.keys():
        return BaseResponse(code=404, msg=f"Không hỗ trợ công cụ tìm kiếm {search_engine_name}")

    if search_engine_name == "bing" and not BING_SUBSCRIPTION_KEY:
        return BaseResponse(code=404, msg=f"Để sử dụng công cụ tìm kiếm Bing, cần thiết lập `BING_SUBSCRIPTION_KEY`")

    history = [History.from_data(h) for h in history]

    async def search_engine_chat_iterator(query: str,
                                          search_engine_name: str,
                                          top_k: int,
                                          history: Optional[List[History]],
                                          model_name: str = LLM_MODELS[0],
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

        docs = await lookup_search_engine(query, search_engine_name, top_k, split_result=split_result)
        context = "\n".join([doc.page_content for doc in docs])

        prompt_template = get_prompt_template("search_engine_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Bắt đầu một task chạy ở nền
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = [
            f"""Nguồn [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
            for inum, doc in enumerate(docs)
        ]

        if len(source_documents) == 0:  # Không tìm thấy tài liệu liên quan (hiếm khi xảy ra)
            source_documents.append(f"""<span style='color:red'>Không tìm thấy tài liệu liên quan, trả lời dựa trên khả năng của mô hình lớn!</span>""")

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

    return EventSourceResponse(search_engine_chat_iterator(query=query,
                                                           search_engine_name=search_engine_name,
                                                           top_k=top_k,
                                                           history=history,
                                                           model_name=model_name,
                                                           prompt_name=prompt_name),
                               )
