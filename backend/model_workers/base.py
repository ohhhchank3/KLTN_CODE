import fastchat.constants
from fastchat.conversation import Conversation

from configs import LOG_PATH, TEMPERATURE

fastchat.constants.LOGDIR = LOG_PATH
import asyncio
import json
import sys
import uuid
from typing import Dict, List, Optional

import fastchat
from fastchat.serve.base_model_worker import BaseModelWorker
from pydantic import BaseModel, root_validator

from backend.utils import get_model_worker_config

__all__ = ["ApiModelWorker", "ApiChatParams", "ApiCompletionParams", "ApiEmbeddingsParams"]


class ApiConfigParams(BaseModel):
    '''
    Trình dự đoán cấu hình API trực tuyến, các giá trị không cung cấp sẽ tự động đọc từ model_config.ONLINE_LLM_MODEL.
    '''
    api_base_url: Optional[str] = None
    api_proxy: Optional[str] = None
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    group_id: Optional[str] = None  # cho minimax
    is_pro: bool = False  # cho minimax

    APPID: Optional[str] = None  # cho xinghuo
    APISecret: Optional[str] = None  # cho xinghuo
    is_v2: bool = False  # cho xinghuo

    worker_name: Optional[str] = None

    class Config:
        extra = "allow"

    @root_validator(pre=True)
    def validate_config(cls, v: Dict) -> Dict:
        if config := get_model_worker_config(v.get("worker_name")):
            for n in cls.__fields__:
                if n in config:
                    v[n] = config[n]
        return v

    def load_config(self, worker_name: str):
        self.worker_name = worker_name
        if config := get_model_worker_config(worker_name):
            for n in self.__fields__:
                if n in config:
                    setattr(self, n, config[n])
        return self


class ApiModelParams(ApiConfigParams):
    '''
    Cấu hình mô hình
    '''
    version: Optional[str] = None
    version_url: Optional[str] = None
    api_version: Optional[str] = None  # cho azure
    deployment_name: Optional[str] = None  # cho azure
    resource_name: Optional[str] = None  # cho azure

    temperature: float = TEMPERATURE
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0


class ApiChatParams(ApiModelParams):
    '''
    Tham số yêu cầu chat
    '''
    messages: List[Dict[str, str]]
    system_message: Optional[str] = None  # cho minimax
    role_meta: Dict = {}  # cho minimax


class ApiCompletionParams(ApiModelParams):
    prompt: str


class ApiEmbeddingsParams(ApiConfigParams):
    texts: List[str]
    embed_model: Optional[str] = None
    to_query: bool = False  # cho minimax


class ApiModelWorker(BaseModelWorker):
    DEFAULT_EMBED_MODEL: str = None  # None nghĩa là không hỗ trợ nhúng

    def __init__(
        self,
        model_names: List[str],
        controller_addr: str = None,
        worker_addr: str = None,
        context_len: int = 2048,
        no_register: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("worker_id", uuid.uuid4().hex[:8])
        kwargs.setdefault("model_path", "")
        kwargs.setdefault("limit_worker_concurrency", 5)
        super().__init__(model_names=model_names,
                         controller_addr=controller_addr,
                         worker_addr=worker_addr,
                         **kwargs)
        import sys

        import fastchat.serve.base_model_worker
        self.logger = fastchat.serve.base_model_worker.logger
        # Khôi phục đầu ra tiêu chuẩn bị bị ghi đè bởi fastchat
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        self.context_len = context_len
        self.semaphore = asyncio.Semaphore(self.limit_worker_concurrency)
        self.version = None

        if not no_register and self.controller_addr:
            self.init_heart_beat()

    def count_token(self, params):
        prompt = params["prompt"]
        return {"count": len(str(prompt)), "error_code": 0}

    def generate_stream_gate(self, params: Dict):
        self.call_ct += 1

        try:
            prompt = params["prompt"]
            if self._is_chat(prompt):
                messages = self.prompt_to_messages(prompt)
                messages = self.validate_messages(messages)
            else:  # Sử dụng chức năng chat để giả lập tiếp tục viết, không hỗ trợ tin nhắn lịch sử
                messages = [{"role": self.user_role, "content": f"please continue writing from here: {prompt}"}]

            p = ApiChatParams(
                messages=messages,
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                max_tokens=params.get("max_new_tokens"),
                version=self.version,
            )
            for resp in self.do_chat(p):
                yield self._jsonify(resp)
        except Exception as e:
            yield self._jsonify({"error_code": 500, "text": f"{self.model_names[0]} requesting API encountered an error: {e}"})

    def generate_gate(self, params):
        try:
            for x in self.generate_stream_gate(params):
                ...
            return json.loads(x[:-1].decode())
        except Exception as e:
            return {"error_code": 500, "text": str(e)}

    # Các phương thức cần người dùng tự định nghĩa

    def do_chat(self, params: ApiChatParams) -> Dict:
        '''
        Thực hiện chức năng Chat, mặc định sử dụng hàm chat trong mô-đun.
        Yêu cầu kết quả trả về: {"error_code": int, "text": str}
        '''
        return {"error_code": 500, "text": f"{self.model_names[0]} has not implemented chat functionality"}

    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        '''
        Thực hiện chức năng Nhúng, mặc định sử dụng hàm embed_documents trong mô-đun.
        Yêu cầu kết quả trả về: {"code": int, "data": List[List[float]], "msg": str}
        '''
        return {"code": 500, "msg": f"{self.model_names[0]} has not implemented embeddings functionality"}

    # Các phương thức hỗ trợ

    def _jsonify(self, data: Dict) -> str:
        '''
        Chuyển kết quả trả về từ hàm chat sang định dạng fastchat openai-api-server
        '''
        return json.dumps(data, ensure_ascii=False).encode() + b"\0"

    def _is_chat(self, prompt: str) -> bool:
        '''
        Kiểm tra xem prompt có phải là kết hợp của tin nhắn chat không
        '''
        key = f"{self.conv.sep}{self.user_role}:"
        return key in prompt

    def prompt_to_messages(self, prompt: str) -> List[Dict]:
        '''
        Chia prompt thành các tin nhắn.
        '''
        result = []
        user_role = self.user_role
        ai_role = self.ai_role
        user_start = user_role + ":"
        ai_start = ai_role + ":"
        for msg in prompt.split(self.conv.sep)[1:-1]:
            if msg.startswith(user_start):
                if content := msg[len(user_start):].strip():
                    result.append({"role": user_role, "content": content})
            elif msg.startswith(ai_start):
                if content := msg[len(ai_start):].strip():
                    result.append({"role": ai_role, "content": content})
            else:
                raise RuntimeError(f"unknown role in msg: {msg}")
        return result

    @classmethod
    def can_embedding(cls):
        return cls.DEFAULT_EMBED_MODEL is not None
