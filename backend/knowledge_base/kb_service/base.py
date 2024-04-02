import operator
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

from backend.db.repository.knowledge_base_repository import (add_kb_to_db,
                                                             delete_kb_from_db,
                                                             get_kb_detail,
                                                             kb_exists,
                                                             list_kbs_from_db,
                                                             load_kb_from_db)
from backend.db.repository.knowledge_file_repository import (
    add_file_to_db, count_files_from_db, delete_file_from_db,
    delete_files_from_db, file_exists_in_db, get_file_detail,
    list_docs_from_db, list_files_from_db)
from backend.embeddings_api import aembed_texts, embed_documents, embed_texts
from backend.knowledge_base.model.kb_document_model import DocumentWithVSId
from backend.knowledge_base.utils import (KnowledgeFile, get_doc_path,
                                          get_kb_path, list_files_from_folder,
                                          list_kbs_from_folder)
from configs.kb_config import (KB_INFO, SCORE_THRESHOLD, VECTOR_SEARCH_TOP_K,
                               kbs_config)
from configs.model_config import EMBEDDING_MODEL, MODEL_PATH


def normalize(embeddings: List[List[float]]) -> np.ndarray:
    '''
    Alternative to sklearn.preprocessing.normalize (using L2), avoiding installing scipy, scikit-learn
    '''
    norm = np.linalg.norm(embeddings, axis=1)
    norm = np.reshape(norm, (norm.shape[0], 1))
    norm = np.tile(norm, (1, len(embeddings[0])))
    return np.divide(embeddings, norm)


class SupportedVSType:
    FAISS = 'faiss'
    MILVUS = 'milvus'
    DEFAULT = 'default'
    ZILLIZ = 'zilliz'
    PG = 'pg'
    ES = 'es'
    CHROMADB = 'chromadb'


class KBService(ABC):

    def __init__(self,
                 knowledge_base_name: str,
                 embed_model: str = EMBEDDING_MODEL,
                 ):
        self.kb_name = knowledge_base_name
        self.kb_info = KB_INFO.get(knowledge_base_name, f"Knowledge base about {knowledge_base_name}")
        self.embed_model = embed_model
        self.kb_path = get_kb_path(self.kb_name)
        self.doc_path = get_doc_path(self.kb_name)
        self.do_init()

    def __repr__(self) -> str:
        return f"{self.kb_name} @ {self.embed_model}"

    def save_vector_store(self):
        '''
        Save the vector store: FAISS saves to disk, milvus saves to the database. PGVector is not supported for now.
        '''
        pass

    def create_kb(self):
        """
        Create a knowledge base
        """
        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path)
        self.do_create_kb()
        status = add_kb_to_db(self.kb_name, self.kb_info, self.vs_type(), self.embed_model)
        return status

    def clear_vs(self):
        """
        Delete all content from the vector store
        """
        self.do_clear_vs()
        status = delete_files_from_db(self.kb_name)
        return status

    def drop_kb(self):
        """
        Delete the knowledge base
        """
        self.do_drop_kb()
        status = delete_kb_from_db(self.kb_name)
        return status

    def _docs_to_embeddings(self, docs: List[Document]) -> Dict:
        '''
        Convert List[Document] to parameters acceptable by VectorStore.add_embeddings
        '''
        return embed_documents(docs=docs, embed_model=self.embed_model, to_query=False)

    def add_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        Add files to the knowledge base
        If docs are specified, do not vectorize the text again and mark the corresponding database entries as custom_docs=True.
        """
        if docs:
            custom_docs = True
            for doc in docs:
                doc.metadata.setdefault("source", kb_file.filename)
        else:
            docs = kb_file.file2text()
            custom_docs = False

        if docs:
            # Change metadata["source"] to a relative path
            for doc in docs:
                try:
                    source = doc.metadata.get("source", "")
                    if os.path.isabs(source):
                        rel_path = Path(source).relative_to(self.doc_path)
                        doc.metadata["source"] = str(rel_path.as_posix().strip("/"))
                except Exception as e:
                    print(f"Cannot convert absolute path ({source}) to relative path. Error is: {e}")
            self.delete_doc(kb_file)
            doc_infos = self.do_add_doc(docs, **kwargs)
            status = add_file_to_db(kb_file,
                                    custom_docs=custom_docs,
                                    docs_count=len(docs),
                                    doc_infos=doc_infos)
        else:
            status = False
        return status

    def delete_doc(self, kb_file: KnowledgeFile, delete_content: bool = False, **kwargs):
        """
        Delete files from the knowledge base
        """
        self.do_delete_doc(kb_file, **kwargs)
        status = delete_file_from_db(kb_file)
        if delete_content and os.path.exists(kb_file.filepath):
            os.remove(kb_file.filepath)
        return status

    def update_info(self, kb_info: str):
        """
        Update the knowledge base introduction
        """
        self.kb_info = kb_info
        status = add_kb_to_db(self.kb_name, self.kb_info, self.vs_type(), self.embed_model)
        return status

    def update_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        Update the vector store using content from the file
        If docs are specified, use custom docs and mark the corresponding database entry as custom_docs=True
        """
        if os.path.exists(kb_file.filepath):
            self.delete_doc(kb_file, **kwargs)
            return self.add_doc(kb_file, docs=docs, **kwargs)

    def exist_doc(self, file_name: str):
        return file_exists_in_db(KnowledgeFile(knowledge_base_name=self.kb_name,
                                               filename=file_name))

    def list_files(self):
        return list_files_from_db(self.kb_name)

    def count_files(self):
        return count_files_from_db(self.kb_name)

    def search_docs(self,
                    query: str,
                    top_k: int = VECTOR_SEARCH_TOP_K,
                    score_threshold: float = SCORE_THRESHOLD,
                    ) -> List[Document]:
        docs = self.do_search(query, top_k, score_threshold)
        return docs

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        return []

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        raise NotImplementedError

    def update_doc_by_ids(self, docs: Dict[str, Document]) -> bool:
        '''
        Input parameter: {doc_id: Document, ...}
        If the value corresponding to doc_id is None, or its page_content is empty, delete the document
        '''
        self.del_doc_by_ids(list(docs.keys()))
        docs = []
        ids = []
        for k, v in docs.items():
            if not v or not v.page_content.strip():
                continue
            ids.append(k)
            docs.append(v)
        self.do_add_doc(docs=docs, ids=ids)
        return True

    def list_docs(self, file_name: str = None, metadata: Dict = {}) -> List[DocumentWithVSId]:
        '''
        Retrieve Documents by file_name or metadata
        '''
        doc_infos = list_docs_from_db(kb_name=self.kb_name, file_name=file_name, metadata=metadata)
        docs = []
        for x in doc_infos:
            doc_info = self.get_doc_by_ids([x["id"]])[0]
            if doc_info is not None:
                # Handle non-empty case
                doc_with_id = DocumentWithVSId(**doc_info.dict(), id=x["id"])
                docs.append(doc_with_id)
            else:
                # Handle empty case
                # You can choose to skip the current iteration or perform other actions
                pass
        return docs

    def get_relative_source_path(self, filepath: str):
        '''
        Convert file path to a relative path to ensure consistency in queries
        '''
        relative_path = filepath
        if os.path.isabs(relative_path):
            try:
                relative_path = Path(filepath).relative_to(self.doc_path)
            except Exception as e:
                print(f"Cannot convert absolute path {filepath} to relative path. Error is: {e}")

        relative_path = str(relative_path.as_posix().strip("/"))
        return relative_path

    @abstractmethod
    def do_create_kb(self):
        """
        Create knowledge base logic in subclasses
        """
        pass

    @staticmethod
    def list_kbs_type():
        return list(kbs_config.keys())

    @classmethod
    def list_kbs(cls):
        return list_kbs_from_db()

    def exists(self, kb_name: str = None):
        kb_name = kb_name or self.kb_name
        return kb_exists(kb_name)

    @abstractmethod
    def vs_type(self) -> str:
        pass

    @abstractmethod
    def do_init(self):
        pass

    @abstractmethod
    def do_drop_kb(self):
        """
        Delete knowledge base logic in subclasses
        """
        pass

    @abstractmethod
    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float,
                  ) -> List[Tuple[Document, float]]:
        """
        Search knowledge base logic in subclasses
        """
        pass

    @abstractmethod
    def do_add_doc(self,
                   docs: List[Document],
                   **kwargs,
                   ) -> List[Dict]:
        """
        Add documents to the knowledge base logic in subclasses
        """
        pass

    @abstractmethod
    def do_delete_doc(self,
                      kb_file: KnowledgeFile):
        """
        Delete documents from the knowledge base logic in subclasses
        """
        pass

    @abstractmethod
    def do_clear_vs(self):
        """
        Delete all vectors from the knowledge base logic in subclasses
        """
        pass


class KBServiceFactory:

    @staticmethod
    def get_service(kb_name: str,
                    vector_store_type: Union[str, SupportedVSType],
                    embed_model: str = EMBEDDING_MODEL,
                    ) -> KBService:
        if isinstance(vector_store_type, str):
            vector_store_type = getattr(SupportedVSType, vector_store_type.upper())
        if SupportedVSType.FAISS == vector_store_type:
            from backend.knowledge_base.kb_service.faiss_kb_service import \
                FaissKBService
            return FaissKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.PG == vector_store_type:
            from backend.knowledge_base.kb_service.pg_kb_service import \
                PGKBService
            return PGKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.MILVUS == vector_store_type:
            from backend.knowledge_base.kb_service.milvus_kb_service import \
                MilvusKBService
            return MilvusKBService(kb_name,embed_model=embed_model)
        elif SupportedVSType.ZILLIZ == vector_store_type:
            from backend.knowledge_base.kb_service.zilliz_kb_service import \
                ZillizKBService
            return ZillizKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.DEFAULT == vector_store_type:
            from backend.knowledge_base.kb_service.milvus_kb_service import \
                MilvusKBService
            return MilvusKBService(kb_name,
                                   embed_model=embed_model)  # other milvus parameters are set in model_config.kbs_config
        elif SupportedVSType.ES == vector_store_type:
            from backend.knowledge_base.kb_service.es_kb_service import \
                ESKBService
            return ESKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.CHROMADB == vector_store_type:
            from backend.knowledge_base.kb_service.chromadb_kb_service import \
                ChromaKBService
            return ChromaKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.DEFAULT == vector_store_type:  # kb_exists of default kbservice is False, to make validation easier.
            from backend.knowledge_base.kb_service.default_kb_service import \
                DefaultKBService
            return DefaultKBService(kb_name)

    @staticmethod
    def get_service_by_name(kb_name: str) -> KBService:
        _, vs_type, embed_model = load_kb_from_db(kb_name)
        if _ is None:  # kb not in db, just return None
            return None
        return KBServiceFactory.get_service(kb_name, vs_type, embed_model)

    @staticmethod
    def get_default():
        return KBServiceFactory.get_service("default", SupportedVSType.DEFAULT)


class EmbeddingsFunAdapter(Embeddings):
    def __init__(self, embed_model: str = EMBEDDING_MODEL):
        self.embed_model = embed_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = embed_texts(texts=texts, embed_model=self.embed_model, to_query=False).data
        return normalize(embeddings).tolist()

    def embed_query(self, text: str) -> List[float]:
        embeddings = embed_texts(texts=[text], embed_model=self.embed_model, to_query=True).data
        query_embed = embeddings[0]
        query_embed_2d = np.reshape(query_embed, (1, -1))  
        normalized_query_embed = normalize(query_embed_2d)
        return normalized_query_embed[0].tolist() 

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = (await aembed_texts(texts=texts, embed_model=self.embed_model, to_query=False)).data
        return normalize(embeddings).tolist()

    async def aembed_query(self, text: str) -> List[float]:
        embeddings = (await aembed_texts(texts=[text], embed_model=self.embed_model, to_query=True)).data
        query_embed = embeddings[0]
        query_embed_2d = np.reshape(query_embed, (1, -1))  
        normalized_query_embed = normalize(query_embed_2d)
        return normalized_query_embed[0].tolist() 


def score_threshold_process(score_threshold, k, docs):
    if score_threshold is not None:
        cmp = (
            operator.le
        )
        docs = [
            (doc, similarity)
            for doc, similarity in docs
            if cmp(similarity, score_threshold)
        ]
    return docs[:k]
