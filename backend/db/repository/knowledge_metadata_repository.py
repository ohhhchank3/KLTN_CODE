from typing import Dict, List

from backend.db.models.knowledge_metadata_model import SummaryChunkModel
from backend.db.session import with_session


@with_session
def list_summary_from_db(session,
                         kb_name: str,
                         metadata: Dict = {},
                         ) -> List[Dict]:
    '''
    Liệt kê các đoạn tóm tắt chunk từ cơ sở kiến thức.
    Trả về dạng: [{"id": str, "summary_context": str, "doc_ids": str}, ...]
    '''
    docs = session.query(SummaryChunkModel).filter(SummaryChunkModel.kb_name.ilike(kb_name))

    for k, v in metadata.items():
        docs = docs.filter(SummaryChunkModel.meta_data[k].astext == str(v))

    return [{"id": x.id,
             "summary_context": x.summary_context,
             "summary_id": x.summary_id,
             "doc_ids": x.doc_ids,
             "metadata": x.metadata} for x in docs.all()]


@with_session
def delete_summary_from_db(session,
                           kb_name: str
                           ) -> List[Dict]:
    '''
    Xóa các đoạn tóm tắt chunk từ cơ sở kiến thức và trả về các đoạn tóm tắt đã bị xóa.
    Trả về dạng: [{"id": str, "summary_context": str, "doc_ids": str}, ...]
    '''
    docs = list_summary_from_db(kb_name=kb_name)
    query = session.query(SummaryChunkModel).filter(SummaryChunkModel.kb_name.ilike(kb_name))
    query.delete(synchronize_session=False)
    session.commit()
    return docs


@with_session
def add_summary_to_db(session,
                      kb_name: str,
                      summary_infos: List[Dict]):
    '''
    Thêm thông tin tóm tắt vào cơ sở dữ liệu.
    summary_infos có dạng: [{"summary_context": str, "doc_ids": str}, ...]
    '''
    for summary in summary_infos:
        obj = SummaryChunkModel(
            kb_name=kb_name,
            summary_context=summary["summary_context"],
            summary_id=summary["summary_id"],
            doc_ids=summary["doc_ids"],
            meta_data=summary["metadata"],
        )
        session.add(obj)

    session.commit()
    return True


@with_session
def count_summary_from_db(session, kb_name: str) -> int:
    return session.query(SummaryChunkModel).filter(SummaryChunkModel.kb_name.ilike(kb_name)).count()
