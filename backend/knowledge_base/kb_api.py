import urllib

from fastapi import Body

from backend.db.repository.knowledge_base_repository import list_kbs_from_db
from backend.knowledge_base.kb_service.base import KBServiceFactory
from backend.knowledge_base.utils import validate_kb_name
from backend.utils import BaseResponse, ListResponse
from configs.basic_config import log_verbose, logger
from configs.model_config import EMBEDDING_MODEL


def list_kbs():
    # Lấy danh sách Cơ sở Kiến thức
    return ListResponse(data=list_kbs_from_db())


def create_kb(knowledge_base_name: str = Body(..., examples=["samples"]),
              vector_store_type: str = Body("faiss"),
              embed_model: str = Body(EMBEDDING_MODEL),
              ) -> BaseResponse:
    # Tạo Cơ sở kiến thức đã chọn
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Đừng tấn công tôi")
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="Tên cơ sở kiến thức không thể trống, vui lòng điền lại tên cơ sở kiến thức")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is not None:
        return BaseResponse(code=404, msg=f"Đã tồn tại cơ sở kiến thức cùng tên {knowledge_base_name}")

    kb = KBServiceFactory.get_service(knowledge_base_name, vector_store_type, embed_model)
    try:
        kb.create_kb()
    except Exception as e:
        msg = f"Lỗi khi tạo cơ sở kiến thức: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=200, msg=f"Đã thêm cơ sở kiến thức {knowledge_base_name}")


def delete_kb(
        knowledge_base_name: str = Body(..., examples=["samples"])
) -> BaseResponse:
    # Xóa cơ sở kiến thức đã chọn
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Đừng tấn công tôi")
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)

    if kb is None:
        return BaseResponse(code=404, msg=f"Không tìm thấy cơ sở kiến thức {knowledge_base_name}")

    try:
        status = kb.clear_vs()
        status = kb.drop_kb()
        if status:
            return BaseResponse(code=200, msg=f"Đã xóa cơ sở kiến thức {knowledge_base_name} thành công")
    except Exception as e:
        msg = f"Lỗi không mong muốn khi xóa cơ sở kiến thức: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"Xóa cơ sở kiến thức {knowledge_base_name} thất bại")
