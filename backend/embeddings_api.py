from typing import Dict, List

from fastapi import Body
from fastapi.concurrency import run_in_threadpool
from langchain.docstore.document import Document

from backend.model_workers.base import ApiEmbeddingsParams
from backend.utils import (BaseResponse, get_model_worker_config,
                           list_embed_models, list_online_embed_models)
from configs.basic_config import log_verbose, logger
from configs.model_config import EMBEDDING_MODEL

online_embed_models = list_online_embed_models()


def embed_texts(
        texts: List[str],
        embed_model: str = EMBEDDING_MODEL,
        to_query: bool = False,
) -> BaseResponse:
    '''
    Đối với các văn bản, thực hiện vector hóa. Định dạng dữ liệu trả về: BaseResponse(data=List[List[float]])
    '''
    try:
        if embed_model in list_embed_models():  # Sử dụng mô hình Embeddings địa phương
            from backend.utils import load_local_embeddings

            embeddings = load_local_embeddings(model=embed_model)
            return BaseResponse(data=embeddings.embed_documents(texts))

        if embed_model in list_online_embed_models():  # Sử dụng API trực tuyến
            config = get_model_worker_config(embed_model)
            worker_class = config.get("worker_class")
            embed_model = config.get("embed_model")
            worker = worker_class()
            if worker_class.can_embedding():
                params = ApiEmbeddingsParams(texts=texts, to_query=to_query, embed_model=embed_model)
                resp = worker.do_embeddings(params)
                return BaseResponse(**resp)

        return BaseResponse(code=500, msg=f"Mô hình được chỉ định {embed_model} không hỗ trợ chức năng Embeddings.")
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"Có lỗi xảy ra trong quá trình vector hóa văn bản: {e}")


async def aembed_texts(
    texts: List[str],
    embed_model: str = EMBEDDING_MODEL,
    to_query: bool = False,
) -> BaseResponse:
    '''
    Đối với các văn bản, thực hiện vector hóa. Định dạng dữ liệu trả về: BaseResponse(data=List[List[float]])
    '''
    try:
        if embed_model in list_embed_models(): # Sử dụng mô hình Embeddings địa phương
            from backend.utils import load_local_embeddings

            embeddings = load_local_embeddings(model=embed_model)
            return BaseResponse(data=await embeddings.aembed_documents(texts))

        if embed_model in list_online_embed_models(): # Sử dụng API trực tuyến
            return await run_in_threadpool(embed_texts,
                                           texts=texts,
                                           embed_model=embed_model,
                                           to_query=to_query)
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"Có lỗi xảy ra trong quá trình vector hóa văn bản: {e}")


def embed_texts_endpoint(
        texts: List[str] = Body(..., description="Danh sách văn bản để nhúng", examples=[["xin chào", "thế giới"]]),
        embed_model: str = Body(EMBEDDING_MODEL,
                                description=f"Mô hình nhúng được sử dụng, ngoài các mô hình nhúng địa phương, cũng hỗ trợ dịch vụ nhúng từ xa ({online_embed_models}) được cung cấp qua API."),
        to_query: bool = Body(False, description="Có sử dụng vectơ để tìm kiếm không. Một số mô hình như Minimax đã tối ưu hóa việc phân biệt vectơ để lưu trữ/tìm kiếm."),
) -> BaseResponse:
    '''
    Đối với các văn bản, thực hiện vector hóa và trả về BaseResponse(data=List[List[float]])
    '''
    return embed_texts(texts=texts, embed_model=embed_model, to_query=to_query)


def embed_documents(
        docs: List[Document],
        embed_model: str = EMBEDDING_MODEL,
        to_query: bool = False,
) -> Dict:
    """
    Vector hóa List[Document], chuyển đổi thành các tham số có thể được chấp nhận bởi VectorStore.add_embeddings
    """
    texts = [x.page_content for x in docs]
    metadatas = [x.metadata for x in docs]
    embeddings = embed_texts(texts=texts, embed_model=embed_model, to_query=to_query).data
    if embeddings is not None:
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }
