from typing import List

from fastapi import Body

from backend.utils import (BaseResponse, fschat_controller_address,
                           get_httpx_client, get_model_worker_config,
                           list_config_llm_models)
from configs.basic_config import log_verbose, logger
from configs.model_config import LLM_MODELS
from configs.server_config import HTTPX_DEFAULT_TIMEOUT


def list_running_models(
    controller_address: str = Body(None, description="Địa chỉ máy chủ của Fastchat controller", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="Tham số này không được sử dụng, chỉ để giữ chỗ"),
) -> BaseResponse:
    '''
    Lấy danh sách các mô hình đã được tải và cấu hình của chúng từ fastchat controller
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(controller_address + "/list_models")
            models = r.json()["models"]
            data = {m: get_model_config(m).data for m in models}
            return BaseResponse(data=data)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            data={},
            msg=f"Không thể lấy danh sách các mô hình từ controller: {controller_address}. Lỗi: {e}"
        )


def list_config_models(
    types: List[str] = Body(["local", "online"], description="Các loại cấu hình mô hình, ví dụ: local, online, worker"),
    placeholder: str = Body(None, description="Chỉ để giữ chỗ, không có tác dụng thực tế")
) -> BaseResponse:
    '''
    Lấy danh sách các mô hình từ cấu hình trong configs
    '''
    data = {}
    for type, models in list_config_llm_models().items():
        if type in types:
            data[type] = {m: get_model_config(m).data for m in models}
    return BaseResponse(data=data)


def get_model_config(
    model_name: str = Body(description="Tên của mô hình LLM trong cấu hình"),
    placeholder: str = Body(None, description="Chỉ để giữ chỗ, không có tác dụng thực tế")
) -> BaseResponse:
    '''
    Lấy cấu hình của mô hình LLM (đã kết hợp)
    '''
    config = {}
    # Xóa thông tin nhạy cảm từ cấu hình ONLINE_MODEL
    for k, v in get_model_worker_config(model_name=model_name).items():
        if not (k == "worker_class"
            or "key" in k.lower()
            or "secret" in k.lower()
            or k.lower().endswith("id")):
            config[k] = v

    return BaseResponse(data=config)


def stop_llm_model(
    model_name: str = Body(..., description="Tên của mô hình LLM cần dừng", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Địa chỉ máy chủ của Fastchat controller", examples=[fschat_controller_address()])
) -> BaseResponse:
    '''
    Gửi yêu cầu dừng một mô hình LLM đến fastchat controller.
    Lưu ý: Do cách thức triển khai của Fastchat, thực tế là dừng model_worker chứa mô hình LLM.
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name},
            )
            return r.json()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            msg=f"Không thể dừng mô hình LLM {model_name} từ controller: {controller_address}. Lỗi: {e}"
        )


def change_llm_model(
    model_name: str = Body(..., description="Mô hình hiện tại đang chạy", examples=[LLM_MODELS[0]]),
    new_model_name: str = Body(..., description="Mô hình mới để chuyển sang", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Địa chỉ máy chủ của Fastchat controller", examples=[fschat_controller_address()])
):
    '''
    Gửi yêu cầu chuyển đổi một mô hình LLM đến fastchat controller.
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name, "new_model_name": new_model_name},
                timeout=HTTPX_DEFAULT_TIMEOUT, # Chờ worker_model mới
            )
            return r.json()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            msg=f"Không thể chuyển đổi mô hình LLM từ controller: {controller_address}. Lỗi: {e}"
        )


def list_search_engines() -> BaseResponse:
    from backend.chat.search_engine_chat import SEARCH_ENGINES

    return BaseResponse(data=list(SEARCH_ENGINES))
