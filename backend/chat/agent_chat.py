import asyncio
import json
from typing import AsyncIterable, List, Optional

from fastapi import Body
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from sse_starlette.sse import EventSourceResponse

from backend.agent import model_container
from backend.agent.callbacks import CustomAsyncIteratorCallbackHandler, Status
from backend.agent.custom_agent.ChatGLM3Agent import initialize_glm3_agent
from backend.agent.custom_template import (CustomOutputParser,
                                           CustomPromptTemplate)
from backend.agent.tools_select import tool_names, tools
from backend.chat.utils import History
from backend.knowledge_base.kb_service.base import get_kb_details
from backend.utils import get_ChatOpenAI, get_prompt_template, wrap_done
from configs.model_config import (HISTORY_LEN, LLM_MODELS, TEMPERATURE,
                                  Agent_MODEL)


async def agent_chat(query: str = Body(..., description="Người dùng nhập", examples=["Đánh một bài thơ"]),
                     history: List[History] = Body([],
                                                   description="Lịch sử trò chuyện",
                                                   examples=[[
                                                       {"role": "user", "content": "Hãy sử dụng công cụ kiểm tra thời tiết hôm nay ở Bắc Kinh"},
                                                       {"role": "assistant",
                                                        "content": "Sử dụng công cụ dự báo thời tiết để biết hôm nay ở Bắc Kinh có mây, 10-14 độ C, gió Đông Bắc cấp 2, dễ bị cảm cúm"}]]
                                                   ),
                     stream: bool = Body(False, description="Xuất dữ liệu theo dòng"),
                     model_name: str = Body(LLM_MODELS[0], description="Tên mô hình LLM."),
                     temperature: float = Body(TEMPERATURE, description="Nhiệt độ LLM", ge=0.0, le=1.0),
                     max_tokens: Optional[int] = Body(None, description="Số lượng token LLM tối đa"),
                     prompt_name: str = Body("default",
                                             description="Tên mẫu prompt được sử dụng (được cấu hình trong configs/prompt_config.py)"),
                     ):
    history = [History.from_data(h) for h in history]

    async def agent_chat_iterator(
            query: str,
            history: Optional[List[History]],
            model_name: str = LLM_MODELS[0],
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = CustomAsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

        kb_list = {x["kb_name"]: x for x in get_kb_details()}
        model_container.DATABASE = {name: details['kb_info'] for name, details in kb_list.items()}

        if Agent_MODEL:
            model_agent = get_ChatOpenAI(
                model_name=Agent_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=[callback],
            )
            model_container.MODEL = model_agent
        else:
            model_container.MODEL = model

        prompt_template = get_prompt_template("agent_chat", prompt_name)
        prompt_template_agent = CustomPromptTemplate(
            template=prompt_template,
            tools=tools,
            input_variables=["input", "intermediate_steps", "history"]
        )
        output_parser = CustomOutputParser()
        llm_chain = LLMChain(llm=model, prompt=prompt_template_agent)
        memory = ConversationBufferWindowMemory(k=HISTORY_LEN * 2)
        for message in history:
            if message.role == 'user':
                memory.chat_memory.add_user_message(message.content)
            else:
                memory.chat_memory.add_ai_message(message.content)
        if "chatglm3" in model_container.MODEL.model_name or "zhipu-api" in model_container.MODEL.model_name:
            agent_executor = initialize_glm3_agent(
                llm=model,
                tools=tools,
                callback_manager=None,
                prompt=prompt_template,
                input_variables=["input", "intermediate_steps", "history"],
                memory=memory,
                verbose=True,
            )
        else:
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nQuan sát:", "Quan sát"],
                allowed_tools=tool_names,
            )
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                                tools=tools,
                                                                verbose=True,
                                                                memory=memory,
                                                                )
        while True:
            try:
                task = asyncio.create_task(wrap_done(
                    agent_executor.acall(query, callbacks=[callback], include_run_info=True),
                    callback.done))
                break
            except:
                pass

        if stream:
            async for chunk in callback.aiter():
                tools_use = []
                # Sử dụng server-sent-events để xuất dữ liệu theo dòng
                data = json.loads(chunk)
                if data["status"] == Status.start or data["status"] == Status.complete:
                    continue
                elif data["status"] == Status.error:
                    tools_use.append("\n```\n")
                    tools_use.append("Tên công cụ: " + data["tool_name"])
                    tools_use.append("Tình trạng công cụ: " + "Gọi thất bại")
                    tools_use.append("Lỗi: " + data["error"])
                    tools_use.append("Thử lại")
                    tools_use.append("\n```\n")
                    yield json.dumps({"tools": tools_use}, ensure_ascii=False)
                elif data["status"] == Status.tool_finish:
                    tools_use.append("\n```\n")
                    tools_use.append("Tên công cụ: " + data["tool_name"])
                    tools_use.append("Tình trạng công cụ: " + "Gọi thành công")
                    tools_use.append("Nhập công cụ: " + data["input_str"])
                    tools_use.append("Xuất công cụ: " + data["output_str"])
                    tools_use.append("\n```\n")
                    yield json.dumps({"tools": tools_use}, ensure_ascii=False)
                elif data["status"] == Status.agent_finish:
                    yield json.dumps({"final_answer": data["final_answer"]}, ensure_ascii=False)
                else:
                    yield json.dumps({"answer": data["llm_token"]}, ensure_ascii=False)


        else:
            answer = ""
            final_answer = ""
            async for chunk in callback.aiter():
                data = json.loads(chunk)
                if data["status"] == Status.start or data["status"] == Status.complete:
                    continue
                if data["status"] == Status.error:
                    answer += "\n```\n"
                    answer += "Tên công cụ: " + data["tool_name"] + "\n"
                    answer += "Tình trạng công cụ: " + "Gọi thất bại" + "\n"
                    answer += "Lỗi: " + data["error"] + "\n"
                    answer += "\n```\n"
                if data["status"] == Status.tool_finish:
                    answer += "\n```\n"
                    answer += "Tên công cụ: " + data["tool_name"] + "\n"
                    answer += "Tình trạng công cụ: " + "Gọi thành công" + "\n"
                    answer += "Nhập công cụ: " + data["input_str"] + "\n"
                    answer += "Xuất công cụ: " + data["output_str"] + "\n"
                    answer += "\n```\n"
                if data["status"] == Status.agent_finish:
                    final_answer = data["final_answer"]
                else:
                    answer += data["llm_token"]

            yield json.dumps({"answer": answer, "final_answer": final_answer}, ensure_ascii=False)
        await task

    return EventSourceResponse(agent_chat_iterator(query=query,
                                                   history=history,
                                                   model_name=model_name,
                                                   prompt_name=prompt_name),
                               )
