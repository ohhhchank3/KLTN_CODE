import os
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List

import streamlit as st
from streamlit_chatbox import *
from streamlit_modal import Modal

from backend.knowledge_base.utils import LOADER_DICT
from configs.kb_config import DEFAULT_KNOWLEDGE_BASE, DEFAULT_SEARCH_ENGINE
from configs.model_config import (HISTORY_LEN, LLM_MODELS, SUPPORT_AGENT_MODEL,
                                  TEMPERATURE)
from configs.prompt_config import PROMPT_TEMPLATES
from webui_pages.utils import *

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "chatchat_icon_blue_square_v2.png"
    )
)


def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    '''
    Trả về lịch sử tin nhắn.
    content_in_expander điều khiển liệu có trả về nội dung trong expander hay không, thường được chọn khi xuất ra, không cần thiết khi truyền vào lịch sử của LLM
    '''

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)


@st.cache_data
def upload_temp_docs(files, _api: ApiRequest) -> str:
    '''
    Tải lên các tệp vào thư mục tạm thời, dùng cho đối thoại tệp
    Trả về ID tạm thời của thư viện vector
    '''
    return _api.upload_temp_docs(files).get("data", {}).get("id")


def parse_command(text: str, modal: Modal) -> bool:
    '''
    Kiểm tra xem người dùng có nhập lệnh tùy chỉnh không, hiện tại hỗ trợ:
    /new {session_name}. Nếu không cung cấp tên, mặc định là “Hội thoại X”
    /del {session_name}. Nếu không cung cấp tên, trong trường hợp có nhiều hơn 1 hội thoại, xóa hội thoại hiện tại.
    /clear {session_name}. Nếu không cung cấp tên, xóa lịch sử hội thoại hiện tại.
    /help. Xem trợ giúp lệnh
    Trả về True nếu nhập là lệnh, ngược lại Trả về False
    '''
    if m := re.match(r"/([^\s]+)\s*(.*)", text):
        cmd, name = m.groups()
        name = name.strip()
        conv_names = chat_box.get_chat_names()
        if cmd == "help":
            modal.open()
        elif cmd == "new":
            if not name:
                i = 1
                while True:
                    name = f"Hội thoại {i}"
                    if name not in conv_names:
                        break
                    i += 1
            if name in st.session_state["conversation_ids"]:
                st.error(f"Tên hội thoại “{name}” đã tồn tại")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"][name] = uuid.uuid4().hex
                st.session_state["cur_conv_name"] = name
        elif cmd == "del":
            name = name or st.session_state.get("cur_conv_name")
            if len(conv_names) == 1:
                st.error("Đây là hội thoại cuối cùng, không thể xóa")
                time.sleep(1)
            elif not name or name not in st.session_state["conversation_ids"]:
                st.error(f"Tên hội thoại không hợp lệ: “{name}”")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"].pop(name, None)
                chat_box.del_chat_name(name)
                st.session_state["cur_conv_name"] = ""
        elif cmd == "clear":
            chat_box.reset_history(name=name or None)
        return True
    return False


def dialogue_page(api: ApiRequest, is_lite: bool = False):
    st.session_state.setdefault("conversation_ids", {})
    st.session_state["conversation_ids"].setdefault(chat_box.cur_chat_name, uuid.uuid4().hex)
    st.session_state.setdefault("file_chat_id", None)
    default_model = api.get_default_llm_model()[0]

    if not chat_box.chat_inited:
        st.toast(
            f"Chào mừng đến với [`Langchain-Chatchat`](https://github.com/chatchat-space/Langchain-Chatchat) ! \n\n"
            f"Đang chạy trên mô hình `{default_model}`, bạn có thể bắt đầu đặt câu hỏi ngay bây giờ."
        )
        chat_box.init_session()

    # Hiển thị thông báo về lệnh tùy chỉnh
    modal = Modal("Lệnh tùy chỉnh", key="cmd_help", max_width="500")
    if modal.is_open():
        with modal.container():
            cmds = [x for x in parse_command.__doc__.split("\n") if x.strip().startswith("/")]
            st.write("\n\n".join(cmds))

    with st.sidebar:
        # Chọn hội thoại
        conv_names = list(st.session_state["conversation_ids"].keys())
        index = 0
        if st.session_state.get("cur_conv_name") in conv_names:
            index = conv_names.index(st.session_state.get("cur_conv_name"))
        conversation_name = st.selectbox("Hội thoại hiện tại:", conv_names, index=index)
        chat_box.use_chat_name(conversation_name)
        conversation_id = st.session_state["conversation_ids"][conversation_name]

        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"Đã chuyển sang chế độ {mode}."
            if mode == "Trò chuyện từ kiến thức":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} Hiện đang sử dụng kiến thức từ `{cur_kb}`."
            st.toast(text)

        dialogue_modes = ["Trò chuyện LLM",
                          "Trò chuyện từ kiến thức",
                          "Trò chuyện từ tệp",
                          "Trò chuyện từ trình tìm kiếm",
                          "Trò chuyện Agent tùy chỉnh",
                          ]
        dialogue_mode = st.selectbox("Chọn chế độ trò chuyện:",
                                     dialogue_modes,
                                     index=0,
                                     on_change=on_mode_change,
                                     key="dialogue_mode",
                                     )

        def on_llm_change():
            if llm_model:
                config = api.get_model_config(llm_model)
                if not config.get("online_api"):  # Chỉ model_worker địa phương mới có thể chuyển đổi mô hình
                    st.session_state["prev_llm_model"] = llm_model
                st.session_state["cur_llm_model"] = st.session_state.llm_model

        def llm_model_format_func(x):
            if x in running_models:
                return f"{x} (Đang chạy)"
            return x

        running_models = list(api.list_running_models())
        available_models = []
        config_models = api.list_config_models()
        if not is_lite:
            for k, v in config_models.get("local", {}).items():
                if (v.get("model_path_exists")
                        and k not in running_models):
                    available_models.append(k)
        for k, v in config_models.get("online", {}).items():
            if not v.get("provider") and k not in running_models and k in LLM_MODELS:
                available_models.append(k)
        llm_models = running_models + available_models
        cur_llm_model = st.session_state.get("cur_llm_model", default_model)
        if cur_llm_model in llm_models:
            index = llm_models.index(cur_llm_model)
        else:
            index = 0
        llm_model = st.selectbox("Chọn mô hình LLM:",
                                 llm_models,
                                 index,
                                 format_func=llm_model_format_func,
                                 on_change=on_llm_change,
                                 key="llm_model",
                                 )
        if (st.session_state.get("prev_llm_model") != llm_model
                and not is_lite
                and not llm_model in config_models.get("online", {})
                and not llm_model in config_models.get("langchain", {})
                and llm_model not in running_models):
            with st.spinner(f"Đang tải mô hình: {llm_model}, vui lòng đợi không thao tác hoặc làm mới trang"):
                prev_model = st.session_state.get("prev_llm_model")
                r = api.change_llm_model(prev_model, llm_model)
                if msg := check_error_msg(r):
                    st.error(msg)
                elif msg := check_success_msg(r):
                    st.success(msg)
                    st.session_state["prev_llm_model"] = llm_model

        index_prompt = {
            "Trò chuyện LLM": "llm_chat",
            "Trò chuyện Agent tùy chỉnh": "agent_chat",
            "Trò chuyện từ trình tìm kiếm": "search_engine_chat",
            "Trò chuyện từ kiến thức": "knowledge_base_chat",
            "Trò chuyện từ tệp": "knowledge_base_chat",
        }
        prompt_templates_kb_list = list(PROMPT_TEMPLATES[index_prompt[dialogue_mode]].keys())
        prompt_template_name = prompt_templates_kb_list[0]
        if "prompt_template_select" not in st.session_state:
            st.session_state.prompt_template_select = prompt_templates_kb_list[0]

        def prompt_change():
            text = f"Đã chuyển sang mẫu {prompt_template_name}."
            st.toast(text)

        prompt_template_select = st.selectbox(
            "Chọn mẫu Prompt:",
            prompt_templates_kb_list,
            index=0,
            on_change=prompt_change,
            key="prompt_template_select",
        )
        prompt_template_name = st.session_state.prompt_template_select
        temperature = st.slider("Nhiệt độ:", 0.0, 2.0, TEMPERATURE, 0.05)
        history_len = st.number_input("Số lượt trò chuyện trong lịch sử:", 0, 20, HISTORY_LEN)

        def on_kb_change():
            st.toast(f"Đã chọn kiến thức: {st.session_state.selected_kb}")

        if dialogue_mode == "Trò chuyện từ kiến thức":
            with st.expander("Cấu hình kiến thức", True):
                kb_list = api.list_knowledge_bases()
                index = 0
                if DEFAULT_KNOWLEDGE_BASE in kb_list:
                    index = kb_list.index(DEFAULT_KNOWLEDGE_BASE)
                selected_kb = st.selectbox(
                    "Chọn kiến thức:",
                    kb_list,
                    index=index,
                    on_change=on_kb_change,
                    key="selected_kb",
                )
                kb_top_k = st.number_input("Số lượng kiến thức xuất hiện:", 1, 20, VECTOR_SEARCH_TOP_K)

                ## Bge model có thể vượt quá 1
                score_threshold = st.slider("Ngưỡng điểm khớp kiến thức:", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01)
        elif dialogue_mode == "Trò chuyện từ tệp":
            with st.expander("Cấu hình trò chuyện từ tệp", True):
                files = st.file_uploader("Tải lên tệp kiến thức:",
                                         [i for ls in LOADER_DICT.values() for i in ls],
                                         accept_multiple_files=True,
                                         )
                kb_top_k = st.number_input("Số lượng kiến thức xuất hiện:", 1, 20, VECTOR_SEARCH_TOP_K)

                ## Bge model có thể vượt quá 1
                score_threshold = st.slider("Ngưỡng điểm khớp kiến thức:", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01)
                if st.button("Bắt đầu tải lên", disabled=len(files) == 0):
                    st.session_state["file_chat_id"] = upload_temp_docs(files, api)
        elif dialogue_mode == "Trò chuyện từ trình tìm kiếm":
            search_engine_list = api.list_search_engines()
            if DEFAULT_SEARCH_ENGINE in search_engine_list:
                index = search_engine_list.index(DEFAULT_SEARCH_ENGINE)
            else:
                index = search_engine_list.index("duckduckgo") if "duckduckgo" in search_engine_list else 0
            with st.expander("Cấu hình trình tìm kiếm", True):
                search_engine = st.selectbox(
                    label="Chọn trình tìm kiếm",
                    options=search_engine_list,
                    index=index,
                )
                se_top_k = st.number_input("Số lượng kết quả trả về:", 1, 20, SEARCH_ENGINE_TOP_K)

    # Hiển thị các tin nhắn từ lịch sử trên ứng dụng khi chạy lại
    chat_box.output_messages()

    chat_input_placeholder = "Vui lòng nhập nội dung trò chuyện, nhấn Shift+Enter để xuống dòng. Nhập /help để xem các lệnh tùy chỉnh."

    def on_feedback(
            feedback,
            message_id: str = "",
            history_index: int = -1,
    ):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)
        api.chat_feedback(message_id=message_id,
                          score=score_int,
                          reason=reason)
        st.session_state["need_rerun"] = True

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "Hãy nhập lý do cho điểm của bạn",
    }

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        if parse_command(text=prompt, modal=modal):  # Người dùng nhập lệnh tùy chỉnh
            st.rerun()
        else:
            history = get_messages_history(history_len)
            chat_box.user_say(prompt)
            if dialogue_mode == "Trò chuyện LLM":
                chat_box.ai_say("Đang suy nghĩ...")
                text = ""
                message_id = ""
                r = api.chat_chat(prompt,
                                  history=history,
                                  conversation_id=conversation_id,
                                  model=llm_model,
                                  prompt_name=prompt_template_name,
                                  temperature=temperature)
                for t in r:
                    if error_msg := check_error_msg(t):  # Kiểm tra xem có lỗi không
                        st.error(error_msg)
                        break
                    text += t.get("text", "")
                    chat_box.update_msg(text)
                    message_id = t.get("message_id", "")

                metadata = {
                    "message_id": message_id,
                }
                chat_box.update_msg(text, streaming=False, metadata=metadata)  # Cập nhật chuỗi cuối cùng, không có con trỏ
                chat_box.show_feedback(**feedback_kwargs,
                                       key=message_id,
                                       on_submit=on_feedback,
                                       kwargs={"message_id": message_id, "history_index": len(chat_box.history) - 1})

            elif dialogue_mode == "Trò chuyện Agent tùy chỉnh":
                if not any(agent in llm_model for agent in SUPPORT_AGENT_MODEL):
                    chat_box.ai_say([
                        f"Đang suy nghĩ... \n\n <span style='color:red'>Mô hình này chưa được điều chỉnh cho Agent, vui lòng chọn một mô hình khác để có trải nghiệm tốt hơn!</span>\n\n\n",
                        Markdown("...", in_expander=True, title="Quá trình suy nghĩ", state="complete"),

                    ])
                else:
                    chat_box.ai_say([
                        f"Đang suy nghĩ...",
                        Markdown("...", in_expander=True, title="Quá trình suy nghĩ", state="complete"),

                    ])
                text = ""
                ans = ""
                for d in api.agent_chat(prompt,
                                        history=history,
                                        model=llm_model,
                                        prompt_name=prompt_template_name,
                                        temperature=temperature,
                                        ):
                    try:
                        d = json.loads(d)
                    except:
                        pass
                    if error_msg := check_error_msg(d):  # Kiểm tra xem có lỗi không
                        st.error(error_msg)
                    if chunk := d.get("answer"):
                        text += chunk
                        chat_box.update_msg(text, element_index=1)
                    if chunk := d.get("final_answer"):
                        ans += chunk
                        chat_box.update_msg(ans, element_index=0)
                    if chunk := d.get("tools"):
                        text += "\n\n".join(d.get("tools", []))
                        chat_box.update_msg(text, element_index=1)
                chat_box.update_msg(ans, element_index=0, streaming=False)
                chat_box.update_msg(text, element_index=1, streaming=False)
            elif dialogue_mode == "Trò chuyện từ kiến thức":
                chat_box.ai_say([
                    f"Đang tìm kiếm trong kiến thức `{selected_kb}` ...",
                    Markdown("...", in_expander=True, title="Kết quả phù hợp từ kiến thức", state="complete"),
                ])
                text = ""
                for d in api.knowledge_base_chat(prompt,
                                                 knowledge_base_name=selected_kb,
                                                 top_k=kb_top_k,
                                                 score_threshold=score_threshold,
                                                 history=history,
                                                 model=llm_model,
                                                 prompt_name=prompt_template_name,
                                                 temperature=temperature):
                    if error_msg := check_error_msg(d):  # Kiểm tra xem có lỗi không
                        st.error(error_msg)
                    elif chunk := d.get("answer"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)
                chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
            elif dialogue_mode == "Trò chuyện từ tệp":
                if st.session_state["file_chat_id"] is None:
                    st.error("Vui lòng tải lên tệp trước khi trò chuyện")
                    st.stop()
                chat_box.ai_say([
                    f"Đang tìm kiếm trong tệp `{st.session_state['file_chat_id']}` ...",
                    Markdown("...", in_expander=True, title="Kết quả từ tệp", state="complete"),
                ])
                text = ""
                for d in api.file_chat(prompt,
                                       knowledge_id=st.session_state["file_chat_id"],
                                       top_k=kb_top_k,
                                       score_threshold=score_threshold,
                                       history=history,
                                       model=llm_model,
                                       prompt_name=prompt_template_name,
                                       temperature=temperature):
                    if error_msg := check_error_msg(d):  # Kiểm tra xem có lỗi không
                        st.error(error_msg)
                    elif chunk := d.get("answer"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)
                chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
            elif dialogue_mode == "Trò chuyện từ trình tìm kiếm":
                chat_box.ai_say([
                    f"Đang tìm kiếm trên `{search_engine}` ...",
                    Markdown("...", in_expander=True, title="Kết quả từ trình tìm kiếm", state="complete"),
                ])
                text = ""
                for d in api.search_engine_chat(prompt,
                                                search_engine_name=search_engine,
                                                top_k=se_top_k,
                                                history=history,
                                                model=llm_model,
                                                prompt_name=prompt_template_name,
                                                temperature=temperature,
                                                split_sent=True,
                                                ):
                    if error_msg := check_error_msg(d):  # Kiểm tra xem có lỗi không
                        st.error(error_msg)
                    elif chunk := d.get("answer"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)
                chat_box.update_msg("\n\n".join(d.get("urls", [])), element_index=1, streaming=False)

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

    now = datetime.now()
    with st.sidebar:

        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()
            st.rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )
