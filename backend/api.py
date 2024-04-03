import os
import sys

import nltk

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from typing import List, Literal

import uvicorn
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from backend.chat.chat import chat
from backend.chat.completion import completion
from backend.chat.feedback import chat_feedback
from backend.chat.search_engine_chat import search_engine_chat
from backend.embeddings_api import embed_texts_endpoint
from backend.llm_api import (change_llm_model, get_model_config,
                             list_config_models, list_running_models,
                             list_search_engines, stop_llm_model)
from backend.utils import (BaseResponse, FastAPI, ListResponse,
                           MakeFastAPIOffline, get_prompt_template,
                           get_server_configs)
from configs.model_config import NLTK_DATA_PATH, VERSION
from configs.server_config import OPEN_CROSS_DOMAIN

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


async def document():
    return RedirectResponse(url="/docs")


def create_app(run_mode: str = None):
    app = FastAPI(
        title="Langchain-Chatchat API Server",
        version=VERSION
    )
    MakeFastAPIOffline(app)
    # Thêm middleware CORS để cho phép tất cả các nguồn gốc
    # Trong config.py, đặt OPEN_DOMAIN=True để cho phép chuyển miền
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app, run_mode=run_mode)
    return app


def mount_app_routes(app: FastAPI, run_mode: str = None):
    app.get("/",
            response_model=BaseResponse,
            summary="Tài liệu swagger")(document)

    # Nhãn: Chat
    app.post("/chat/chat",
             tags=["Chat"],
             summary="Trò chuyện với mô hình LLM (qua LLMChain)",
             )(chat)

    app.post("/chat/search_engine_chat",
             tags=["Chat"],
             summary="Trò chuyện với công cụ tìm kiếm",
             )(search_engine_chat)

    app.post("/chat/feedback",
             tags=["Chat"],
             summary="Trả về đánh giá cuộc trò chuyện của mô hình LLM",
             )(chat_feedback)

    # Gắn các tuyến đường liên quan đến kiến thức
    mount_knowledge_routes(app)
    # Gắn các tuyến đường liên quan đến tóm tắt tên file
    mount_filename_summary_routes(app)

    # Gắn các tuyến đường liên quan đến mô hình LLM
    app.post("/llm_model/list_running_models",
             tags=["Quản lý mô hình LLM"],
             summary="Liệt kê các mô hình đã chạy",
             )(list_running_models)

    app.post("/llm_model/list_config_models",
             tags=["Quản lý mô hình LLM"],
             summary="Liệt kê các mô hình được cấu hình",
             )(list_config_models)

    app.post("/llm_model/get_model_config",
             tags=["Quản lý mô hình LLM"],
             summary="Lấy cấu hình mô hình (sau khi hợp nhất)",
             )(get_model_config)

    app.post("/llm_model/stop",
             tags=["Quản lý mô hình LLM"],
             summary="Dừng mô hình LLM đã chỉ định (Model Worker)",
             )(stop_llm_model)

    app.post("/llm_model/change",
             tags=["Quản lý mô hình LLM"],
             summary="Thay đổi mô hình LLM đã chỉ định (Model Worker)",
             )(change_llm_model)

    # Gắn các tuyến đường liên quan đến máy chủ
    app.post("/server/configs",
             tags=["Trạng thái máy chủ"],
             summary="Lấy thông tin cấu hình máy chủ",
             )(get_server_configs)

    app.post("/server/list_search_engines",
             tags=["Trạng thái máy chủ"],
             summary="Lấy danh sách các công cụ tìm kiếm hỗ trợ của máy chủ",
             )(list_search_engines)

    @app.post("/server/get_prompt_template",
             tags=["Trạng thái máy chủ"],
             summary="Lấy mẫu gợi nhớ đã cấu hình"
             )
    def get_server_prompt_template(
        type: Literal["llm_chat", "knowledge_base_chat", "search_engine_chat", "agent_chat"]=Body("llm_chat", description="Loại mẫu, có thể chọn: llm_chat, knowledge_base_chat, search_engine_chat, agent_chat"),
        name: str = Body("default", description="Tên mẫu"),
    ) -> str:
        return get_prompt_template(type=type, name=name)

    # Gắn các tuyến đường khác
    app.post("/other/completion",
             tags=["Khác"],
             summary="Yêu cầu mô hình LLM hoàn thiện (qua LLMChain)",
             )(completion)

    app.post("/other/embed_texts",
            tags=["Khác"],
            summary="Biến đổi văn bản thành vectơ, hỗ trợ mô hình local và online",
            )(embed_texts_endpoint)


def mount_knowledge_routes(app: FastAPI):
    from backend.chat.agent_chat import agent_chat
    from backend.chat.file_chat import file_chat, upload_temp_docs
    from backend.chat.knowledge_base_chat import knowledge_base_chat
    from backend.knowledge_base.kb_api import create_kb, delete_kb, list_kbs
    from backend.knowledge_base.kb_doc_api import (DocumentWithVSId,
                                                   delete_docs, download_doc,
                                                   list_files,
                                                   recreate_vector_store,
                                                   search_docs, update_docs,
                                                   update_docs_by_id,
                                                   update_info, upload_docs)

    app.post("/chat/knowledge_base_chat",
             tags=["Chat"],
             summary="Trò chuyện với kiến thức")(knowledge_base_chat)

    app.post("/chat/file_chat",
             tags=["Quản lý kiến thức"],
             summary="Trò chuyện qua tệp"
             )(file_chat)

    app.post("/chat/agent_chat",
             tags=["Chat"],
             summary="Trò chuyện với agent")(agent_chat)

    # Nhãn: Quản lý kiến thức
    app.get("/knowledge_base/list_knowledge_bases",
            tags=["Quản lý kiến thức"],
            response_model=ListResponse,
            summary="Lấy danh sách kiến thức")(list_kbs)

    app.post("/knowledge_base/create_knowledge_base",
             tags=["Quản lý kiến thức"],
             response_model=BaseResponse,
             summary="Tạo kiến thức"
             )(create_kb)

    app.post("/knowledge_base/delete_knowledge_base",
             tags=["Quản lý kiến thức"],
             response_model=BaseResponse,
             summary="Xóa kiến thức"
             )(delete_kb)

    app.get("/knowledge_base/list_files",
            tags=["Quản lý kiến thức"],
            response_model=ListResponse,
            summary="Lấy danh sách tệp trong kiến thức"
            )(list_files)

    app.post("/knowledge_base/search_docs",
             tags=["Quản lý kiến thức"],
             response_model=List[DocumentWithVSId],
             summary="Tìm kiếm trong kiến thức"
             )(search_docs)

    app.post("/knowledge_base/update_docs_by_id",
             tags=["Quản lý kiến thức"],
             response_model=BaseResponse,
             summary="Cập nhật tài liệu trực tiếp trong kiến thức"
             )(update_docs_by_id)

    app.post("/knowledge_base/upload_docs",
             tags=["Quản lý kiến thức"],
             response_model=BaseResponse,
             summary="Tải lên tệp vào kiến thức và/hoặc thực hiện biến đổi vectơ"
             )(upload_docs)

    app.post("/knowledge_base/delete_docs",
             tags=["Quản lý kiến thức"],
             response_model=BaseResponse,
             summary="Xóa tệp trong kiến thức"
             )(delete_docs)

    app.post("/knowledge_base/update_info",
             tags=["Quản lý kiến thức"],
             response_model=BaseResponse,
             summary="Cập nhật giới thiệu kiến thức"
             )(update_info)
    app.post("/knowledge_base/update_docs",
             tags=["Quản lý kiến thức"],
             response_model=BaseResponse,
             summary="Cập nhật tệp trong kiến thức"
             )(update_docs)

    app.get("/knowledge_base/download_doc",
            tags=["Quản lý kiến thức"],
            summary="Tải xuống tệp kiến thức")(download_doc)

    app.post("/knowledge_base/recreate_vector_store",
             tags=["Quản lý kiến thức"],
             summary="Tạo lại kho vectơ dựa trên tài liệu trong content, xuất thông tin tiến trình dạng stream"
             )(recreate_vector_store)

    app.post("/knowledge_base/upload_temp_docs",
             tags=["Quản lý kiến thức"],
             summary="Tải lên tệp vào thư mục tạm thời, sử dụng cho trò chuyện qua tệp."
             )(upload_temp_docs)


def mount_filename_summary_routes(app: FastAPI):
    from backend.knowledge_base.kb_summary_api import (
        recreate_summary_vector_store, summary_doc_ids_to_vector_store,
        summary_file_to_vector_store)

    app.post("/knowledge_base/kb_summary_api/summary_file_to_vector_store",
             tags=["Quản lý tóm tắt tên file"],
             summary="Tóm tắt dựa trên tên tệp"
             )(summary_file_to_vector_store)
    app.post("/knowledge_base/kb_summary_api/summary_doc_ids_to_vector_store",
             tags=["Quản lý tóm tắt tên file"],
             summary="Tóm tắt dựa trên doc_ids",
             response_model=BaseResponse,
             )(summary_doc_ids_to_vector_store)
    app.post("/knowledge_base/kb_summary_api/recreate_summary_vector_store",
             tags=["Quản lý tóm tắt tên file"],
             summary="Tạo lại tóm tắt từ các tệp trong kiến thức"
             )(recreate_summary_vector_store)



def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='langchain-ChatGLM',
                                     description='Về langchain-ChatGLM, ChatGLM dựa trên kiến thức local với langchain'
                                                 ' ｜ Trò chuyện dựa trên kiến thức local với langchain')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # Khởi tạo các thông báo
    args = parser.parse_args()
    args_dict = vars(args)

    app = create_app()

    run_api(host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
