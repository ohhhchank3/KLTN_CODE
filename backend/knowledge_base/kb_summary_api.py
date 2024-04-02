import json
from typing import List, Optional

from fastapi import Body
from sse_starlette import EventSourceResponse

from backend.knowledge_base.kb_service.base import KBServiceFactory
from backend.knowledge_base.kb_summary.base import KBSummaryService
from backend.knowledge_base.kb_summary.summary_chunk import SummaryAdapter
from backend.knowledge_base.model.kb_document_model import DocumentWithVSId
from backend.knowledge_base.utils import list_files_from_folder
from backend.utils import BaseResponse, get_ChatOpenAI, wrap_done
from configs.basic_config import log_verbose, logger
from configs.kb_config import DEFAULT_VS_TYPE, OVERLAP_SIZE
from configs.model_config import EMBEDDING_MODEL, LLM_MODELS, TEMPERATURE


def recreate_summary_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        allow_empty_kb: bool = Body(True),
        vs_type: str = Body(DEFAULT_VS_TYPE),
        embed_model: str = Body(EMBEDDING_MODEL),
        file_description: str = Body(''),
        model_name: str = Body(LLM_MODELS[0], description="Tên mô hình LLM."),
        temperature: float = Body(TEMPERATURE, description="Nhiệt độ mẫu LLM", ge=0.0, le=1.0),
        max_tokens: Optional[int] = Body(None, description="Giới hạn số lượng token được tạo ra bởi LLM, mặc định là None cho giá trị tối đa của mô hình"),
):
    """
    Tạo lại vector tóm tắt cho một tệp trong cơ sở kiến thức
    :param max_tokens:
    :param model_name:
    :param temperature:
    :param file_description:
    :param knowledge_base_name:
    :param allow_empty_kb:
    :param vs_type:
    :param embed_model:
    :return:
    """

    def output():

        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"Không tìm thấy cơ sở kiến thức ‘{knowledge_base_name}’"}
        else:
            # Tạo lại cơ sở kiến thức
            kb_summary = KBSummaryService(knowledge_base_name, embed_model)
            kb_summary.drop_kb_summary()
            kb_summary.create_kb_summary()

            llm = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reduce_llm = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Bộ chuyển đổi tóm tắt văn bản
            summary = SummaryAdapter.form_summary(llm=llm,
                                                  reduce_llm=reduce_llm,
                                                  overlap_size=OVERLAP_SIZE)
            files = list_files_from_folder(knowledge_base_name)

            i = 0
            for i, file_name in enumerate(files):

                doc_infos = kb.list_docs(file_name=file_name)
                docs = summary.summarize(file_description=file_description,
                                         docs=doc_infos)

                status_kb_summary = kb_summary.add_kb_summary(summary_combine_docs=docs)
                if status_kb_summary:
                    logger.info(f"({i + 1} / {len(files)}): {file_name} Hoàn thành tóm tắt")
                    yield json.dumps({
                        "code": 200,
                        "msg": f"({i + 1} / {len(files)}): {file_name}",
                        "total": len(files),
                        "finished": i + 1,
                        "doc": file_name,
                    }, ensure_ascii=False)
                else:

                    msg = f"Lỗi khi tóm tắt tệp '{file_name}' trong cơ sở kiến thức '{knowledge_base_name}'. Đã bỏ qua."
                    logger.error(msg)
                    yield json.dumps({
                        "code": 500,
                        "msg": msg,
                    })
                i += 1

    return EventSourceResponse(output())


def summary_file_to_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        file_name: str = Body(..., examples=["test.pdf"]),
        allow_empty_kb: bool = Body(True),
        vs_type: str = Body(DEFAULT_VS_TYPE),
        embed_model: str = Body(EMBEDDING_MODEL),
        file_description: str = Body(''),
        model_name: str = Body(LLM_MODELS[0], description="Tên mô hình LLM."),
        temperature: float = Body(TEMPERATURE, description="Nhiệt độ mẫu LLM", ge=0.0, le=1.0),
        max_tokens: Optional[int] = Body(None, description="Giới hạn số lượng token được tạo ra bởi LLM, mặc định là None cho giá trị tối đa của mô hình"),
):
    """
    Tóm tắt một tệp trong cơ sở kiến thức thành vector
    :param model_name:
    :param max_tokens:
    :param temperature:
    :param file_description:
    :param file_name:
    :param knowledge_base_name:
    :param allow_empty_kb:
    :param vs_type:
    :param embed_model:
    :return:
    """

    def output():
        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"Không tìm thấy cơ sở kiến thức '{knowledge_base_name}'"}
        else:
            # Tạo lại cơ sở kiến thức
            kb_summary = KBSummaryService(knowledge_base_name, embed_model)
            kb_summary.create_kb_summary()

            llm = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reduce_llm = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Bộ chuyển đổi tóm tắt văn bản
            summary = SummaryAdapter.form_summary(llm=llm,
                                                  reduce_llm=reduce_llm,
                                                  overlap_size=OVERLAP_SIZE)

            doc_infos = kb.list_docs(file_name=file_name)
            docs = summary.summarize(file_description=file_description,
                                     docs=doc_infos)

            status_kb_summary = kb_summary.add_kb_summary(summary_combine_docs=docs)
            if status_kb_summary:
                logger.info(f" {file_name} Hoàn thành tóm tắt")
                yield json.dumps({
                    "code": 200,
                    "msg": f"{file_name} Hoàn thành tóm tắt",
                    "doc": file_name,
                }, ensure_ascii=False)
            else:

                msg = f"Lỗi khi tóm tắt tệp '{file_name}' trong cơ sở kiến thức '{knowledge_base_name}'. Đã bỏ qua."
                logger.error(msg)
                yield json.dumps({
                    "code": 500,
                    "msg": msg,
                })

    return EventSourceResponse(output())


def summary_doc_ids_to_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        doc_ids: List = Body([], examples=[["uuid"]]),
        vs_type: str = Body(DEFAULT_VS_TYPE),
        embed_model: str = Body(EMBEDDING_MODEL),
        file_description: str = Body(''),
        model_name: str = Body(LLM_MODELS[0], description="Tên mô hình LLM."),
        temperature: float = Body(TEMPERATURE, description="Nhiệt độ mẫu LLM", ge=0.0, le=1.0),
        max_tokens: Optional[int] = Body(None, description="Giới hạn số lượng token được tạo ra bởi LLM, mặc định là None cho giá trị tối đa của mô hình"),
) -> BaseResponse:
    """
    Tóm tắt các tài liệu trong cơ sở kiến thức dựa trên doc_ids
    :param knowledge_base_name:
    :param doc_ids:
    :param model_name:
    :param max_tokens:
    :param temperature:
    :param file_description:
    :param vs_type:
    :param embed_model:
    :return:
    """
    kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
    if not kb.exists():
        return BaseResponse(code=404, msg=f"Không tìm thấy cơ sở kiến thức {knowledge_base_name}", data={})
    else:
        llm = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reduce_llm = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Bộ chuyển đổi tóm tắt văn bản
        summary = SummaryAdapter.form_summary(llm=llm,
                                              reduce_llm=reduce_llm,
                                              overlap_size=OVERLAP_SIZE)

        doc_infos = kb.get_doc_by_ids(ids=doc_ids)
        # Chuyển đổi doc_infos thành các đối tượng được bao gói DocumentWithVSId
        doc_info_with_ids = [DocumentWithVSId(**doc.dict(), id=with_id) for with_id, doc in zip(doc_ids, doc_infos)]

        docs = summary.summarize(file_description=file_description,
                                 docs=doc_info_with_ids)

        # Chuyển đổi docs thành dict
        resp_summarize = [{**doc.dict()} for doc in docs]

        return BaseResponse(code=200, msg="Hoàn thành tóm tắt", data={"summarize": resp_summarize})
