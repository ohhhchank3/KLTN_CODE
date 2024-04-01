import os

MODEL_ROOT_PATH = ""

# 选用的 Embedding 名称
EMBEDDING_MODEL = "vietnamese-sbert"
EMBEDDING_DEVICE = "auto"
RERANKER_MODEL = "bge-reranker-v2-gemma"
USE_RERANKER = False
RERANKER_MAX_LENGTH = 1024
EMBEDDING_KEYWORD_FILE = "keywords.txt"
EMBEDDING_MODEL_OUTPUT_PATH = "output"


LLM_MODELS = ["MixSUra-SFT-AWQ", "gemini-api", "openai-api"]
Agent_MODEL = None


LLM_DEVICE = "auto"

HISTORY_LEN = 3

MAX_TOKENS = 2048

TEMPERATURE = 0.7

ONLINE_LLM_MODEL = {
    "openai-api": {
        "model_name": "gpt-4",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "",
        "openai_proxy": "",
    },
    # Gemini API https://makersuite.google.com/app/apikey
    "gemini-api": {
        "api_key": "",
        "provider": "GeminiWorker",
    }

}


MODEL_PATH = {
    "embed_model": {
        "vietnamese-sbert": "keepitreal/vietnamese-sbert",
        "embedding-001": "your Google API key",
        "text-embedding-ada-002": "your OPENAI_API_KEY",
    },

    "llm_model": {
        "MixSUra-SFT-AWQ": "ura-hcmut/MixSUra-SFT-AWQ",

    },

    "reranker": {
        "bge-reranker-large": "BAAI/bge-reranker-large",
        "bge-reranker-base": "BAAI/bge-reranker-base",
    }
}


NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

VLLM_MODEL_DICT = {
    "MixSUra-SFT-AWQ": "ura-hcmut/MixSUra-SFT-AWQ",
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-chat-hf",

}

SUPPORT_AGENT_MODEL = [
    "openai-api", 
    "gemini-api", 
    "MixSUra-SFT-AWQ",
]