import asyncio
from typing import AsyncIterable, Optional

from fastapi import Body
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from backend.utils import get_OpenAI, get_prompt_template, wrap_done
from sse_starlette.sse import EventSourceResponse

from configs import LLM_MODELS, TEMPERATURE


async def completion(query: str = Body(..., description="Người dùng nhập vào", examples=["Thí dụ về input"]),
                     stream: bool = Body(False, description="Xuất kết quả theo luồng"),
                     echo: bool = Body(False, description="Trả về cả input"),
                     model_name: str = Body(LLM_MODELS[0], description="Tên của mô hình LLM"),
                     temperature: float = Body(TEMPERATURE, description="Nhiệt độ mẫu của LLM", ge=0.0, le=1.0),
                     max_tokens: Optional[int] = Body(1024, description="Số lượng token tối đa",),
                     prompt_name: str = Body("default", description="Tên của mẫu prompt")):

    async def completion_iterator(query: str,
                                  model_name: str = LLM_MODELS[0],
                                  prompt_name: str = prompt_name,
                                  echo: bool = echo,
                                  ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_OpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
            echo=echo
        )

        prompt_template = get_prompt_template("completion", prompt_name)
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(prompt=prompt, llm=model)

        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                yield token
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer

        await task

    return EventSourceResponse(completion_iterator(query=query,
                                                 model_name=model_name,
                                                 prompt_name=prompt_name),
                             )
