import os

DEFAULT_KNOWLEDGE_BASE = "samples"
DEFAULT_VS_TYPE = "faiss"

CACHED_VS_NUM = 1
CACHED_MEMO_VS_NUM = 10


CHUNK_SIZE = 2500


OVERLAP_SIZE = 100

# 知识库匹配向量数量
VECTOR_SEARCH_TOP_K = 5


SCORE_THRESHOLD = 1.0

DEFAULT_SEARCH_ENGINE = "duckduckgo"

SEARCH_ENGINE_TOP_K = 3


BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"

BING_SUBSCRIPTION_KEY = ""

METAPHOR_API_KEY = ""

# https://www.seniverse.com/
SENIVERSE_API_KEY = ""

ZH_TITLE_ENHANCE = False

PDF_OCR_THRESHOLD = (0.6, 0.6)

KB_INFO = {
    "知识库名称": "知识库介绍",
    "samples": "关于本项目issue的解答",
}


KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
if not os.path.exists(KB_ROOT_PATH):
    os.mkdir(KB_ROOT_PATH)
DB_ROOT_PATH = os.path.join(KB_ROOT_PATH, "info.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_ROOT_PATH}"

# 可选向量库类型及对应配置
kbs_config = {
    "faiss": {
    },
    "milvus": {
        "host": "127.0.0.1",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": False,
    },
    "zilliz": {
        "host": "in01-a7ce524e41e3935.ali-cn-hangzhou.vectordb.zilliz.com.cn",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": True,
        },
    "pg": {
        "connection_uri": "postgresql://postgres:postgres@127.0.0.1:5432/langchain_chatchat",
    },

    "es": {
        "host": "127.0.0.1",
        "port": "9200",
        "index_name": "test_index",
        "user": "",
        "password": ""
    },
    "milvus_kwargs":{
        "search_params":{"metric_type": "L2"},
        "index_params":{"metric_type": "L2","index_type": "HNSW"} 
    },
    "chromadb": {}
}

text_splitter_dict = {
    "VietNamRecursiveTextSplitter": {
        "source": "huggingface", 
        "tokenizer_name_or_path": "",
    },
    "SpacyTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": "gpt2",
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "cl100k_base",
    },
    "MarkdownHeaderTextSplitter": {
        "headers_to_split_on":
            [
                ("#", "head1"),
                ("##", "head2"),
                ("###", "head3"),
                ("####", "head4"),
            ]
    },
}


TEXT_SPLITTER_NAME = "VietNamRecursiveTextSplitter"
EMBEDDING_KEYWORD_FILE = "embedding_keywords.txt"