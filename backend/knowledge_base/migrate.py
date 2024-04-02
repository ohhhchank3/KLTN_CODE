import os
from typing import List, Literal

from dateutil.parser import parse

from backend.db.base import Base, engine
from backend.db.models.conversation_model import ConversationModel
from backend.db.models.message_model import MessageModel
from backend.db.repository.knowledge_file_repository import \
    add_file_to_db  # ensure Models are imported
from backend.db.repository.knowledge_metadata_repository import \
    add_summary_to_db
from backend.db.session import session_scope
from backend.knowledge_base.kb_service.base import KBServiceFactory
from backend.knowledge_base.utils import (KnowledgeFile, files2docs_in_thread,
                                          get_file_path,
                                          list_files_from_folder,
                                          list_kbs_from_folder)
from configs.basic_config import log_verbose, logger
from configs.kb_config import (CHUNK_SIZE, DEFAULT_VS_TYPE, OVERLAP_SIZE,
                               ZH_TITLE_ENHANCE)
from configs.model_config import EMBEDDING_MODEL


def create_tables():
    Base.metadata.create_all(bind=engine)


def reset_tables():
    Base.metadata.drop_all(bind=engine)
    create_tables()


def import_from_db(
        sqlite_path: str = None,
        # csv_path: str = None,
) -> bool:
    """
    Trong trường hợp không có thay đổi nào trong cơ sở kiến thức và bộ lưu trữ vector, nhập dữ liệu từ cơ sở dữ liệu sao lưu vào info.db.
    Được sử dụng khi nâng cấp phiên bản, cấu trúc info.db thay đổi, nhưng không cần tạo lại vector hóa.
    Hãy đảm bảo rằng tên bảng trong cơ sở dữ liệu backup và cơ sở dữ liệu cần nhập có tên giống nhau,
    cũng như tên các trường cần nhập cũng phải giống nhau.
    Hiện tại chỉ hỗ trợ sqlite.
    """
    import sqlite3 as sql
    from pprint import pprint

    models = list(Base.registry.mappers)

    try:
        con = sql.connect(sqlite_path)
        con.row_factory = sql.Row
        cur = con.cursor()
        tables = [x["name"] for x in cur.execute("select name from sqlite_master where type='table'").fetchall()]
        for model in models:
            table = model.local_table.fullname
            if table not in tables:
                continue
            print(f"processing table: {table}")
            with session_scope() as session:
                for row in cur.execute(f"select * from {table}").fetchall():
                    data = {k: row[k] for k in row.keys() if k in model.columns}
                    if "create_time" in data:
                        data["create_time"] = parse(data["create_time"])
                    pprint(data)
                    session.add(model.class_(**data))
        con.close()
        return True
    except Exception as e:
        print(f"Không thể đọc cơ sở dữ liệu sao lưu: {sqlite_path}. Lỗi: {e}")
        return False


def file_to_kbfile(kb_name: str, files: List[str]) -> List[KnowledgeFile]:
    kb_files = []
    for file in files:
        try:
            kb_file = KnowledgeFile(filename=file, knowledge_base_name=kb_name)
            kb_files.append(kb_file)
        except Exception as e:
            msg = f"{e}, đã bỏ qua"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
    return kb_files


def folder2db(
        kb_names: List[str],
        mode: Literal["recreate_vs", "update_in_db", "increment"],
        vs_type: Literal["faiss", "milvus", "pg", "chromadb"] = DEFAULT_VS_TYPE,
        embed_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
):
    """
    Sử dụng các tệp đã tồn tại trong thư mục cục bộ để điền dữ liệu vào cơ sở dữ liệu và/hoặc bộ lưu trữ vector.
    Thiết lập tham số `mode` thành:
        recreate_vs: tạo lại tất cả các vector hóa và điền thông tin vào cơ sở dữ liệu bằng các tệp đã tồn tại trong thư mục cục bộ
        update_in_db: cập nhật vector hóa và thông tin cơ sở dữ liệu bằng các tệp cục bộ chỉ tồn tại trong cơ sở dữ liệu
        increment: tạo vector hóa và thông tin cơ sở dữ liệu cho các tệp cục bộ chỉ tồn tại trong cơ sở dữ liệu
    """

    def files2vs(kb_name: str, kb_files: List[KnowledgeFile]):
        for success, result in files2docs_in_thread(kb_files,
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    zh_title_enhance=zh_title_enhance):
            if success:
                _, filename, docs = result
                print(f"Đang thêm {kb_name}/{filename} vào bộ lưu trữ vector, tổng cộng {len(docs)} tài liệu")
                kb_file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
                kb_file.splited_docs = docs
                kb.add_doc(kb_file=kb_file, not_refresh_vs_cache=True)
            else:
                print(result)

    kb_names = kb_names or list_kbs_from_folder()
    for kb_name in kb_names:
        kb = KBServiceFactory.get_service(kb_name, vs_type, embed_model)
        if not kb.exists():
            kb.create_kb()

        # Xóa bộ lưu trữ vector, tạo lại từ các tệp cục bộ
        if mode == "recreate_vs":
            kb.clear_vs()
            kb.create_kb()
            kb_files = file_to_kbfile(kb_name, list_files_from_folder(kb_name))
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        elif mode == "update_in_db":
            files = kb.list_files()
            kb_files = file_to_kbfile(kb_name, files)
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        elif mode == "increment":
            db_files = kb.list_files()
            folder_files = list_files_from_folder(kb_name)
            files = list(set(folder_files) - set(db_files))
            kb_files = file_to_kbfile(kb_name, files)
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        else:
            print(f"Chế độ di chuyển không được hỗ trợ: {mode}")


def prune_db_docs(kb_names: List[str]):
    """
    Xóa các tài liệu trong cơ sở dữ liệu không tồn tại trong thư mục cục bộ.
    Được sử dụng để xóa các tài liệu trong cơ sở dữ liệu sau khi người dùng xóa một số tệp tài liệu trong trình duyệt tệp.
    """
    for kb_name in kb_names:
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is not None:
            files_in_db = kb.list_files()
            files_in_folder = list_files_from_folder(kb_name)
            files = list(set(files_in_db) - set(files_in_folder))
            kb_files = file_to_kbfile(kb_name, files)
            for kb_file in kb_files:
                kb.delete_doc(kb_file, not_refresh_vs_cache=True)
                print(f"Xóa tài liệu thành công cho tệp: {kb_name}/{kb_file.filename}")


def prune_folder_files(kb_names: List[str]):
    """
    Xóa các tệp tài liệu trong thư mục cục bộ không tồn tại trong cơ sở dữ liệu.
    Được sử dụng để giải phóng không gian đĩa cục bộ bằng cách xóa các tệp tài liệu không sử dụng.
    """
    for kb_name in kb_names:
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is not None:
            files_in_db = kb.list_files()
            files_in_folder = list_files_from_folder(kb_name)
            files = list(set(files_in_folder) - set(files_in_db))
            for file in files:
                os.remove(get_file_path(kb_name, file))
                print(f"Xóa tệp thành công: {kb_name}/{file}")
