import sys

from configs.model_config import LLM_DEVICE

HTTPX_DEFAULT_TIMEOUT = 300.0
OPEN_CROSS_DOMAIN = False

DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"

# webui.py server
WEBUI_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 9999,
}

# api.py server
API_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 7865,
}

# fastchat openai_api server
FSCHAT_OPENAI_API = {
    "host": DEFAULT_BIND_HOST,
    "port": 20000,
}


FSCHAT_MODEL_WORKERS = {

    "default": {
        "host": DEFAULT_BIND_HOST,
        "port": 20002,
        "device": LLM_DEVICE,
        "infer_turbo": False,

    },
    "MixSUra-SFT-AWQ": {
        "device": "cpu",
    },
    "openai-api": {
        "port": 21001,
    },
    "gemini-api": {
        "port": 21002,
    },
}

FSCHAT_CONTROLLER = {
    "host": DEFAULT_BIND_HOST,
    "port": 20001,
    "dispatch_method": "shortest_queue",
}