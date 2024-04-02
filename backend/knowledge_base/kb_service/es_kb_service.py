import os
import shutil
from typing import List

from elasticsearch import BadRequestError, Elasticsearch
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.elasticsearch import ElasticsearchStore

from backend.knowledge_base.kb_service.base import KBService, SupportedVSType
from backend.knowledge_base.utils import KnowledgeFile
from backend.utils import load_local_embeddings
from configs.basic_config import logger
from configs.kb_config import CACHED_VS_NUM, KB_ROOT_PATH, kbs_config
from configs.model_config import EMBEDDING_DEVICE, EMBEDDING_MODEL


class ESKBService(KBService):

    def do_init(self):
        self.kb_path = self.get_kb_path(self.kb_name)
        self.index_name = os.path.split(self.kb_path)[-1]
        self.IP = kbs_config[self.vs_type()]['host']
        self.PORT = kbs_config[self.vs_type()]['port']
        self.user = kbs_config[self.vs_type()].get("user",'')
        self.password = kbs_config[self.vs_type()].get("password",'')
        self.dims_length = kbs_config[self.vs_type()].get("dims_length",None)
        self.embeddings_model = load_local_embeddings(self.embed_model, EMBEDDING_DEVICE)
        try:
            if self.user != "" and self.password != "":
                self.es_client_python =  Elasticsearch(f"http://{self.IP}:{self.PORT}",
                basic_auth=(self.user,self.password))
            else:
                logger.warning("ES chưa cấu hình tên người dùng và mật khẩu")
                self.es_client_python = Elasticsearch(f"http://{self.IP}:{self.PORT}")
        except ConnectionError:
            logger.error("Kết nối đến Elasticsearch thất bại!")
            raise ConnectionError
        except Exception as e:
            logger.error(f"Lỗi xảy ra: {e}")
            raise e
        try:
            mappings = {
                "properties": {
                    "dense_vector": {
                        "type": "dense_vector",
                        "dims": self.dims_length,
                        "index": True
                    }
                }
            }
            self.es_client_python.indices.create(index=self.index_name, mappings=mappings)
        except BadRequestError as e:
            logger.error("Tạo chỉ mục thất bại, thử lại")
            logger.error(e)

        try:
            if self.user != "" and self.password != "":
                self.db_init = ElasticsearchStore(
                es_url=f"http://{self.IP}:{self.PORT}",
                index_name=self.index_name,
                query_field="context",
                vector_query_field="dense_vector",
                embedding=self.embeddings_model,
                es_user=self.user,
                es_password=self.password
            )
            else:
                logger.warning("ES chưa cấu hình tên người dùng và mật khẩu")
                self.db_init = ElasticsearchStore(
                    es_url=f"http://{self.IP}:{self.PORT}",
                    index_name=self.index_name,
                    query_field="context",
                    vector_query_field="dense_vector",
                    embedding=self.embeddings_model,
                )
        except ConnectionError:
            print("### Khởi tạo Elasticsearch thất bại!")
            logger.error("### Khởi tạo Elasticsearch thất bại!")
            raise ConnectionError
        except Exception as e:
            logger.error(f"Lỗi xảy ra: {e}")
            raise e
        try:
            self.db_init._create_index_if_not_exists(
                                                     index_name=self.index_name,
                                                     dims_length=self.dims_length
                                                     )
        except Exception as e:
            logger.error("Tạo chỉ mục thất bại...")
            logger.error(e)

    @staticmethod
    def get_kb_path(knowledge_base_name: str):
        return os.path.join(KB_ROOT_PATH, knowledge_base_name)

    @staticmethod
    def get_vs_path(knowledge_base_name: str):
        return os.path.join(ESKBService.get_kb_path(knowledge_base_name), "vector_store")

    def do_create_kb(self):
        if os.path.exists(self.doc_path):
            if not os.path.exists(os.path.join(self.kb_path, "vector_store")):
                os.makedirs(os.path.join(self.kb_path, "vector_store"))
            else:
                logger.warning("Thư mục `vector_store` đã tồn tại.")

    def vs_type(self) -> str:
        return SupportedVSType.ES

    def _load_es(self, docs, embed_model):
        try:
            if self.user != "" and self.password != "":
                self.db = ElasticsearchStore.from_documents(
                        documents=docs,
                        embedding=embed_model,
                        es_url= f"http://{self.IP}:{self.PORT}",
                        index_name=self.index_name,
                        distance_strategy="COSINE",
                        query_field="context",
                        vector_query_field="dense_vector",
                        verify_certs=False,
                        es_user=self.user,
                        es_password=self.password
                    )
            else:
                self.db = ElasticsearchStore.from_documents(
                        documents=docs,
                        embedding=embed_model,
                        es_url= f"http://{self.IP}:{self.PORT}",
                        index_name=self.index_name,
                        distance_strategy="COSINE",
                        query_field="context",
                        vector_query_field="dense_vector",
                        verify_certs=False)
        except ConnectionError as ce:
            print(ce)
            print("Kết nối đến Elasticsearch thất bại!")
            logger.error("Kết nối đến Elasticsearch thất bại!")
        except Exception as e:
            logger.error(f"Lỗi xảy ra: {e}")
            print(e)

    def do_search(self, query:str, top_k: int, score_threshold: float):
        docs = self.db_init.similarity_search_with_score(query=query, k=top_k)
        return docs

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        results = []
        for doc_id in ids:
            try:
                response = self.es_client_python.get(index=self.index_name, id=doc_id)
                source = response["_source"]
                text = source.get("context", "")
                metadata = source.get("metadata", {})
                results.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                logger.error(f"Lỗi khi truy xuất tài liệu từ Elasticsearch! {e}")
        return results

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        for doc_id in ids:
            try:
                self.es_client_python.delete(index=self.index_name, id=doc_id, refresh=True)
            except Exception as e:
                logger.error(f"Lỗi xóa tài liệu từ Elasticsearch! {e}")

    def do_delete_doc(self, kb_file, **kwargs):
        if self.es_client_python.indices.exists(index=self.index_name):
            query = {
                "query": {
                    "term": {
                        "metadata.source.keyword": self.get_relative_source_path(kb_file.filepath)
                    }
                }
            }
            search_results = self.es_client_python.search(body=query, size=50)
            delete_list = [hit["_id"] for hit in search_results['hits']['hits']]
            if len(delete_list) == 0:
                return None
            else:
                for doc_id in delete_list:
                    try:
                        self.es_client_python.delete(index=self.index_name, id=doc_id, refresh=True)
                    except Exception as e:
                        logger.error(f"Lỗi xóa tài liệu từ Elasticsearch! {e}")

    def do_add_doc(self, docs: List[Document], **kwargs):
        self._load_es(docs=docs, embed_model=self.embeddings_model)
        file_path = docs[0].metadata.get("source")
        query = {
            "query": {
                "term": {
                    "metadata.source.keyword": file_path
                },
                "term": {
                    "_index": self.index_name
                }
            }
        }
        search_results = self.es_client_python.search(body=query, size=50)
        if len(search_results["hits"]["hits"]) == 0:
            raise ValueError("Số lượng phục hồi phần tử bằng 0")
        info_docs = [{"id":hit["_id"], "metadata": hit["_source"]["metadata"]} for hit in search_results["hits"]["hits"]]
        return info_docs

    def do_clear_vs(self):
        if self.es_client_python.indices.exists(index=self.kb_name):
            self.es_client_python.indices.delete(index=self.kb_name)

    def do_drop_kb(self):
        if os.path.exists(self.kb_path):
            shutil.rmtree(self.kb_path)


if __name__ == '__main__':
    esKBService = ESKBService("test")
    esKBService.add_doc(KnowledgeFile(filename="README.md", knowledge_base_name="test"))
    print(esKBService.search_docs("Làm thế nào để bắt đầu dịch vụ API"))
