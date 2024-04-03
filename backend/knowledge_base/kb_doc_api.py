import json
import os
import urllib
from typing import Dict, List

from fastapi import Body, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from langchain.docstore.document import Document
from pydantic import Json
from sse_starlette import EventSourceResponse

from backend.db.repository.knowledge_file_repository import get_file_detail
from backend.knowledge_base.kb_service.base import KBServiceFactory
from backend.knowledge_base.model.kb_document_model import DocumentWithVSId
from backend.knowledge_base.utils import (KnowledgeFile, files2docs_in_thread,
                                          get_file_path,
                                          list_files_from_folder,
                                          validate_kb_name)
from backend.utils import BaseResponse, ListResponse, run_in_thread_pool
from configs.basic_config import log_verbose, logger
from configs.kb_config import (CHUNK_SIZE, DEFAULT_VS_TYPE, OVERLAP_SIZE,
                               SCORE_THRESHOLD, VECTOR_SEARCH_TOP_K,
                               ZH_TITLE_ENHANCE)
from configs.model_config import EMBEDDING_MODEL


def search_docs(
        query: str = Body("", description="Từ khóa tìm kiếm", examples=["Xin chào"]),
        knowledge_base_name: str = Body(..., description="Tên cơ sở kiến thức", examples=["samples"]),
        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="Số lượng kết quả tìm kiếm tối đa"),
        score_threshold: float = Body(SCORE_THRESHOLD,
                                      description="Ngưỡng độ tương đồng của kết quả tìm kiếm trong kho kiến thức, giá trị nằm trong khoảng từ 0 đến 1, "
                                                  "giá trị càng nhỏ thể hiện mức độ tương đồng càng cao, "
                                                  "đặt giá trị 1 tương đương không lọc kết quả, khuyến nghị đặt giá trị xung quanh 0.5",
                                      ge=0, le=1),
        file_name: str = Body("", description="Tên tệp, hỗ trợ wildcard SQL"),
        metadata: dict = Body({}, description="Lọc kết quả tìm kiếm dựa trên metadata, chỉ hỗ trợ tìm kiếm trên một cấp key"),
) -> List[DocumentWithVSId]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    data = []
    if kb is not None:
        if query:
            docs = kb.search_docs(query, top_k, score_threshold)
            data = [DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")) for x in docs]
        elif file_name or metadata:
            data = kb.list_docs(file_name=file_name, metadata=metadata)
            for d in data:
                if "vector" in d.metadata:
                    del d.metadata["vector"]
    return data


def update_docs_by_id(
        knowledge_base_name: str = Body(..., description="Tên cơ sở kiến thức", examples=["samples"]),
        docs: Dict[str, Document] = Body(..., description="Nội dung tài liệu cần cập nhật, định dạng như sau: {id: Document, ...}")
) -> BaseResponse:
    '''
    Cập nhật nội dung tài liệu dựa trên ID
    '''
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=500, msg=f"Không tìm thấy cơ sở kiến thức {knowledge_base_name}")
    if kb.update_doc_by_ids(docs=docs):
        return BaseResponse(msg=f"Cập nhật tài liệu thành công")
    else:
        return BaseResponse(msg=f"Cập nhật tài liệu thất bại")


def list_files(
        knowledge_base_name: str
) -> ListResponse:
    if not validate_kb_name(knowledge_base_name):
        return ListResponse(code=403, msg="Đừng tấn công tôi", data=[])

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return ListResponse(code=404, msg=f"Không tìm thấy cơ sở kiến thức {knowledge_base_name}", data=[])
    else:
        all_doc_names = kb.list_files()
        return ListResponse(data=all_doc_names)


def _save_files_in_thread(files: List[UploadFile],
                          knowledge_base_name: str,
                          override: bool):
    """
    Sử dụng luồng để lưu các tệp đã tải lên vào thư mục tương ứng với cơ sở kiến thức.
    Trả về kết quả lưu: {"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    """

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        '''
        Lưu một tệp.
        '''
        try:
            filename = file.filename
            file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename)
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = file.file.read()  # Đọc nội dung của tệp đã tải lên
            if (os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                file_status = f"Tệp {filename} đã tồn tại."
                logger.warn(file_status)
                return dict(code=404, msg=file_status, data=data)

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"Tải lên tệp {filename} thành công", data=data)
        except Exception as e:
            msg = f"Tải lên tệp {filename} thất bại, thông báo lỗi: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return dict(code=500, msg=msg, data=data)

    params = [{"file": file, "knowledge_base_name": knowledge_base_name, "override": override} for file in files]
    for result in run_in_thread_pool(save_file, params=params):
        yield result


def upload_docs(
        files: List[UploadFile] = File(..., description="Tệp đã tải lên, hỗ trợ tải nhiều tệp"),
        knowledge_base_name: str = Form(..., description="Tên cơ sở kiến thức", examples=["samples"]),
        override: bool = Form(False, description="Ghi đè lên các tệp đã tồn tại"),
        to_vector_store: bool = Form(True, description="Có thực hiện vector hóa tệp sau khi tải lên không"),
        chunk_size: int = Form(CHUNK_SIZE, description="Độ dài tối đa của một đoạn văn bản"),
        chunk_overlap: int = Form(OVERLAP_SIZE, description="Độ dài trùng lặp giữa các đoạn văn bản"),
        zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="Có kích hoạt cải thiện tiêu đề tiếng Trung không"),
        docs: Json = Form({}, description="Cấu trúc docs tùy chỉnh, cần chuyển đổi sang chuỗi json",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        not_refresh_vs_cache: bool = Form(False, description="Không làm mới bộ lưu trữ vector (sử dụng cho FAISS)"),
) -> BaseResponse:
    """
    API: Tải lên các tệp và/hoặc thực hiện vector hóa
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Đừng tấn công tôi")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Không tìm thấy cơ sở kiến thức {knowledge_base_name}")

    failed_files = {}
    file_names = list(docs.keys())

    # Đầu tiên lưu các tệp đã tải lên vào đĩa
    for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

        if filename not in file_names:
            file_names.append(filename)

    # Vector hóa các tệp đã lưu
    if to_vector_store:
        result = update_docs(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=True,
        )
        failed_files.update(result.data["failed_files"])
        if not not_refresh_vs_cache:
            kb.save_vector_store()

    return BaseResponse(code=200, msg="Tải lên tệp và vector hóa hoàn thành", data={"failed_files": failed_files})


def delete_docs(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        file_names: List[str] = Body(..., examples=[["file_name.md", "test.txt"]]),
        delete_content: bool = Body(False),
        not_refresh_vs_cache: bool = Body(False, description="Không làm mới bộ lưu trữ vector (sử dụng cho FAISS)"),
) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Đừng tấn công tôi")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Không tìm thấy cơ sở kiến thức {knowledge_base_name}")

    failed_files = {}
    for file_name in file_names:
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"Không tìm thấy tệp {file_name}"

        try:
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb.delete_doc(kb_file, delete_content, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"Xoá tệp {file_name} thất bại, thông báo lỗi: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"Xoá tệp hoàn thành", data={"failed_files": failed_files})


def update_info(
        knowledge_base_name: str = Body(..., description="Tên cơ sở kiến thức", examples=["samples"]),
        kb_info: str = Body(..., description="Thông tin giới thiệu cơ sở kiến thức", examples=["Đây là một cơ sở kiến thức"]),
):
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Đừng tấn công tôi")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Không tìm thấy cơ sở kiến thức {knowledge_base_name}")
    kb.update_info(kb_info)

    return BaseResponse(code=200, msg=f"Cập nhật thông tin cơ sở kiến thức hoàn thành", data={"kb_info": kb_info})


def update_docs(
        knowledge_base_name: str = Body(..., description="Tên cơ sở kiến thức", examples=["samples"]),
        file_names: List[str] = Body(..., description="Tên tệp, hỗ trợ nhiều tệp", examples=[["file_name1", "text.txt"]]),
        chunk_size: int = Body(CHUNK_SIZE, description="Độ dài tối đa của một đoạn văn bản"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="Độ lặp giữa các đoạn văn bản"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="Có kích hoạt cải thiện tiêu đề tiếng Trung không"),
        override_custom_docs: bool = Body(False, description="Ghi đè lên docs tùy chỉnh trước đó"),
        docs: Json = Body({}, description="Cấu trúc docs tùy chỉnh, cần chuyển đổi sang chuỗi json",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        not_refresh_vs_cache: bool = Body(False, description="Không làm mới bộ lưu trữ vector (sử dụng cho FAISS)"),
) -> BaseResponse:
    """
    Cập nhật tài liệu trong cơ sở kiến thức
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Đừng tấn công tôi")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Không tìm thấy cơ sở kiến thức {knowledge_base_name}")

    failed_files = {}
    kb_files = []

    # Tạo danh sách tệp cần tải docs
    for file_name in file_names:
        file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        if file_detail.get("custom_docs") and not override_custom_docs:
            continue
        if file_name not in docs:
            try:
                kb_files.append(KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name))
            except Exception as e:
                msg = f"Đọc tài liệu {file_name} thất bại: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                failed_files[file_name] = msg

    # Tạo docs từ các tệp và thực hiện vector hóa
    for status, result in files2docs_in_thread(kb_files,
                                               chunk_size=chunk_size,
                                               chunk_overlap=chunk_overlap,
                                               zh_title_enhance=zh_title_enhance):
        if status:
            kb_name, file_name, new_docs = result
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=kb_name)
            kb_file.splited_docs = new_docs
            kb.update_doc(kb_file, not_refresh_vs_cache=True)
        else:
            kb_name, file_name, error = result
            msg = f"Lỗi khi thêm tệp ‘{file_name}’ vào cơ sở kiến thức ‘{kb_name}’: {error}. Đã bỏ qua."
            logger.error(msg)
            failed_files[file_name] = error

    # Vector hóa các docs tùy chỉnh
    for file_name, v in docs.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)
            kb.update_doc(kb_file, docs=v, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"Lỗi khi thêm docs tùy chỉnh vào tệp {file_name}: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"Cập nhật tài liệu hoàn thành", data={"failed_files": failed_files})
