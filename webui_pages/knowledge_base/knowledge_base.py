import json
import os
import time
from typing import Dict, List, Literal, Tuple

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder

from backend.knowledge_base.kb_service.base import (get_kb_details,
                                                    get_kb_file_details)
from backend.knowledge_base.utils import LOADER_DICT, get_file_path
from backend.utils import list_embed_models, list_online_embed_models
from configs.kb_config import (CHUNK_SIZE, DEFAULT_VS_TYPE, OVERLAP_SIZE,
                               ZH_TITLE_ENHANCE, kbs_config)
from configs.model_config import EMBEDDING_MODEL
from webui_pages.utils import *

cell_renderer = JsCode("""function(params) {if(params.value==true){return '✓'}else{return '×'}}""")

def config_aggrid(
    df: pd.DataFrame,
    columns: Dict[Tuple[str, str], Dict] = {},
    selection_mode: Literal["single", "multiple", "disabled"] = "single",
    use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    gb.configure_pagination(
        enabled=True,
        paginationAutoPageSize=False,
        paginationPageSize=10
    )
    return gb

def file_exists(kb: str, selected_rows: List) -> Tuple[str, str]:
    """
    Kiểm tra xem một tệp tài liệu có tồn tại trong thư mục cơ sở kiến thức cục bộ không.
    Trả về tên và đường dẫn của tệp nếu tồn tại.
    """
    if selected_rows:
        file_name = selected_rows[0]["file_name"]
        file_path = get_file_path(kb, file_name)
        if os.path.isfile(file_path):
            return file_name, file_path
    return "", ""

def knowledge_base_page(api: ApiRequest, is_lite: bool = None):
    try:
        kb_list = {x["kb_name"]: x for x in get_kb_details()}
    except Exception as e:
        st.error("Lỗi khi lấy thông tin cơ sở kiến thức. Vui lòng kiểm tra xem đã hoàn thành các bước khởi tạo hoặc di chuyển theo `README.md` chưa, hoặc có lỗi kết nối cơ sở dữ liệu không.")
        st.stop()
    kb_names = list(kb_list.keys())

    if "selected_kb_name" in st.session_state and st.session_state["selected_kb_name"] in kb_names:
        selected_kb_index = kb_names.index(st.session_state["selected_kb_name"])
    else:
        selected_kb_index = 0

    if "selected_kb_info" not in st.session_state:
        st.session_state["selected_kb_info"] = ""

    def format_selected_kb(kb_name: str) -> str:
        if kb := kb_list.get(kb_name):
            return f"{kb_name} ({kb['vs_type']} @ {kb['embed_model']})"
        else:
            return kb_name

    selected_kb = st.selectbox(
        "Chọn hoặc Tạo mới cơ sở kiến thức:",
        kb_names + ["Tạo mới cơ sở kiến thức"],
        format_func=format_selected_kb,
        index=selected_kb_index
    )

    if selected_kb == "Tạo mới cơ sở kiến thức":
        with st.form("Tạo mới cơ sở kiến thức"):

            kb_name = st.text_input(
                "Tên cơ sở kiến thức mới",
                placeholder="Tên cơ sở kiến thức mới, không hỗ trợ đặt tên bằng tiếng Trung",
                key="kb_name",
            )
            kb_info = st.text_input(
                "Giới thiệu cơ sở kiến thức",
                placeholder="Giới thiệu cơ sở kiến thức, để giúp Agent tìm kiếm dễ dàng hơn",
                key="kb_info",
            )

            cols = st.columns(2)

            vs_types = list(kbs_config.keys())
            vs_type = cols[0].selectbox(
                "Loại lưu trữ vector",
                vs_types,
                index=vs_types.index(DEFAULT_VS_TYPE),
                key="vs_type",
            )

            if is_lite:
                embed_models = list_online_embed_models()
            else:
                embed_models = list_embed_models() + list_online_embed_models()

            embed_model = cols[1].selectbox(
                "Mô hình Embedding",
                embed_models,
                index=embed_models.index(EMBEDDING_MODEL),
                key="embed_model",
            )

            submit_create_kb = st.form_submit_button(
                "Tạo mới",
                use_container_width=True,
            )

        if submit_create_kb:
            if not kb_name or not kb_name.strip():
                st.error(f"Tên cơ sở kiến thức không thể trống!")
            elif kb_name in kb_list:
                st.error(f"Cơ sở kiến thức có tên {kb_name} đã tồn tại!")
            else:
                ret = api.create_knowledge_base(
                    knowledge_base_name=kb_name,
                    vector_store_type=vs_type,
                    embed_model=embed_model,
                )
                st.toast(ret.get("msg", " "))
                st.session_state["selected_kb_name"] = kb_name
                st.session_state["selected_kb_info"] = kb_info
                st.rerun()

    elif selected_kb:
        kb = selected_kb
        st.session_state["selected_kb_info"] = kb_list[kb]['kb_info']
        # Tải lên tệp
        files = st.file_uploader("Tải lên tệp kiến thức:",
                                 [i for ls in LOADER_DICT.values() for i in ls],
                                 accept_multiple_files=True,
                                 )
        kb_info = st.text_area("Nhập giới thiệu cơ sở kiến thức:", value=st.session_state["selected_kb_info"], max_chars=None,
                               key=None,
                               help=None, on_change=None, args=None, kwargs=None)

        if kb_info != st.session_state["selected_kb_info"]:
            st.session_state["selected_kb_info"] = kb_info
            api.update_kb_info(kb, kb_info)

        # Cấu hình xử lý tệp
        with st.expander(
                "Cấu hình xử lý tệp",
                expanded=True,
        ):
            cols = st.columns(3)
            chunk_size = cols[0].number_input("Kích thước tối đa của mỗi phần:", 1, 1000, CHUNK_SIZE)
            chunk_overlap = cols[1].number_input("Kích thước trùng lắp giữa các phần:", 0, chunk_size, OVERLAP_SIZE)
            cols[2].write("")
            cols[2].write("")
            zh_title_enhance = cols[2].checkbox("Kích hoạt tăng cường tiêu đề tiếng Trung", ZH_TITLE_ENHANCE)

        if st.button(
                "Thêm tệp vào cơ sở kiến thức",
                disabled=len(files) == 0,
        ):
            ret = api.upload_kb_docs(files,
                                     knowledge_base_name=kb,
                                     override=True,
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap,
                                     zh_title_enhance=zh_title_enhance)
            if msg := check_success_msg(ret):
                st.toast(msg, icon="✔")
            elif msg := check_error_msg(ret):
                st.toast(msg, icon="✖")

        st.divider()

        # Chi tiết cơ sở kiến thức
        doc_details = pd.DataFrame(get_kb_file_details(kb))
        selected_rows = []
        if not len(doc_details):
            st.info(f"Cơ sở kiến thức `{kb}` hiện chưa có tệp nào")
        else:
            st.write(f"Cơ sở kiến thức `{kb}` hiện có các tệp sau:")
            st.info("Cơ sở kiến thức bao gồm tệp nguồn và lưu trữ vector, vui lòng chọn tệp để thao tác")
            doc_details.drop(columns=["kb_name"], inplace=True)
            doc_details = doc_details[[
                "No", "file_name", "document_loader", "text_splitter", "docs_count", "in_folder", "in_db",
            ]]
            doc_details["in_folder"] = doc_details["in_folder"].replace(True, "✓").replace(False, "×")
            doc_details["in_db"] = doc_details["in_db"].replace(True, "✓").replace(False, "×")
            gb = config_aggrid(
                doc_details,
                {
                    ("No", "STT"): {},
                    ("file_name", "Tên tệp"): {},
                    ("document_loader", "Trình tải tài liệu"): {},
                    ("docs_count", "Số lượng tài liệu"): {},
                    ("text_splitter", "Trình tách từ"): {},
                    ("in_folder", "Tệp nguồn"): {"cellRenderer": cell_renderer},
                    ("in_db", "Lưu trữ vector"): {"cellRenderer": cell_renderer},
                },
                "multiple",
            )

            doc_grid = AgGrid(
                doc_details,
                gb.build(),
                columns_auto_size_mode="FIT_CONTENTS",
                theme="alpine",
                custom_css={
                    "#gridToolBar": {"display": "none"},
                },
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False
            )

            selected_rows = doc_grid.get("selected_rows", [])

            cols = st.columns(4)
            file_name, file_path = file_exists(kb, selected_rows)
            if file_path:
                with open(file_path, "rb") as fp:
                    cols[0].download_button(
                        "Tải xuống tệp đã chọn",
                        fp,
                        file_name=file_name,
                        use_container_width=True, )
            else:
                cols[0].download_button(
                    "Tải xuống tệp đã chọn",
                    "",
                    disabled=True,
                    use_container_width=True, )

            st.write()
            # Tách từ tệp và lưu trữ vào cơ sở kiến thức
            if cols[1].button(
                    "Lưu trữ lại vào lưu trữ vector" if selected_rows and (
                            pd.DataFrame(selected_rows)["in_db"]).any() else "Lưu trữ vào lưu trữ vector",
                    disabled=not file_exists(kb, selected_rows)[0],
                    use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.update_kb_docs(kb,
                                   file_names=file_names,
                                   chunk_size=chunk_size,
                                   chunk_overlap=chunk_overlap,
                                   zh_title_enhance=zh_title_enhance)
                st.rerun()

            # Xóa khỏi lưu trữ vector, nhưng không xóa tệp
            if cols[2].button(
                    "Xóa khỏi lưu trữ vector",
                    disabled=not (selected_rows and selected_rows[0]["in_db"]),
                    use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.delete_kb_docs(kb, file_names=file_names)
                st.rerun()

            if cols[3].button(
                    "Xóa khỏi cơ sở kiến thức",
                    type="primary",
                    use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.delete_kb_docs(kb, file_names=file_names, delete_content=True)
                st.rerun()

        st.divider()

        cols = st.columns(3)

        if cols[0].button(
                "Tạo lại lưu trữ vector dựa trên tệp nguồn",
                help="Không cần tải lên tệp, sao chép tài liệu vào thư mục nội dung của cơ sở kiến thức tương ứng từ các phương tiện khác, nhấn nút này để tạo lại cơ sở kiến thức.",
                use_container_width=True,
                type="primary",
        ):
            with st.spinner("Đang tạo lại lưu trữ vector, vui lòng đợi trong khi tiến trình diễn ra."):
                empty = st.empty()
                empty.progress(0.0, "")
                for d in api.recreate_vector_store(kb,
                                                   chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   zh_title_enhance=zh_title_enhance):
                    if msg := check_error_msg(d):
                        st.toast(msg)
                    else:
                        empty.progress(d["finished"] / d["total"], d["msg"])
                st.rerun()

        if cols[2].button(
                "Xóa cơ sở kiến thức",
                use_container_width=True,
        ):
            ret = api.delete_knowledge_base(kb)
            st.toast(ret.get("msg", " "))
            time.sleep(1)
            st.rerun()

        with st.sidebar:
            keyword = st.text_input("Từ khóa tìm kiếm")
            top_k = st.slider("Số kết quả tìm kiếm", 1, 100, 3)

        st.write("Danh sách tài liệu trong tệp. Nhấp đúp để chỉnh sửa, nhập Y để xóa hàng tương ứng.")
        docs = []
        df = pd.DataFrame([], columns=["seq", "id", "content", "source"])
        if selected_rows:
            file_name = selected_rows[0]["file_name"]
            docs = api.search_kb_docs(knowledge_base_name=selected_kb, file_name=file_name)
        

            data = [
                {"seq": i + 1, "id": x["id"], "page_content": x["page_content"], "source": x["metadata"].get("source"),
                 "type": x["type"],
                 "metadata": json.dumps(x["metadata"], ensure_ascii=False),
                 "to_del": "",
                 
                 } for i, x in enumerate(docs)]
            
            df = pd.DataFrame(data)

            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_columns(["id", "source", "type", "metadata"], hide=True)
            gb.configure_column("seq", "STT", width=50)
            gb.configure_column("page_content", "Nội dung", editable=True, autoHeight=True, wrapText=True, flex=1,
                                cellEditor="agLargeTextCellEditor", cellEditorPopup=True)
            gb.configure_column("to_del", "Xóa", editable=True, width=50, wrapHeaderText=True,
                                cellEditor="agCheckboxCellEditor", cellRender="agCheckboxCellRenderer")
            gb.configure_selection()
            edit_docs = AgGrid(df, gb.build())

            if st.button("Lưu thay đổi"):
                origin_docs = {
                    x["id"]: {"page_content": x["page_content"], "type": x["type"], "metadata": x["metadata"]} for x in
                    docs}
                changed_docs = []
                for index, row in edit_docs.data.iterrows():
                    origin_doc = origin_docs[row["id"]]
                    if row["page_content"] != origin_doc["page_content"]:
                        if row["to_del"] not in ["Y", "y", 1]:
                            changed_docs.append({
                                "page_content": row["page_content"],
                                "type": row["type"],
                                "metadata": json.loads(row["metadata"]),
                            })

                if changed_docs:
                    if api.update_kb_docs(knowledge_base_name=selected_kb,
                                          file_names=[file_name],
                                          docs={file_name: changed_docs}):
                        st.toast("Cập nhật tài liệu thành công")
                    else:
                        st.toast("Cập nhật tài liệu thất bại")
