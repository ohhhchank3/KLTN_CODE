from __future__ import annotations

import json
import os
import re
import sys
import warnings
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import (AsyncCallbackManagerForChainRun,
                                         CallbackManagerForChainRun)
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import Extra, root_validator
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import asyncio

from pydantic import BaseModel, Field

from backend.agent import model_contain
from backend.chat.knowledge_base_chat import knowledge_base_chat
from configs import MAX_TOKENS, SCORE_THRESHOLD, VECTOR_SEARCH_TOP_K


async def search_knowledge_base_iter(database: str, query: str):
    response = await knowledge_base_chat(query=query,
                                         knowledge_base_name=database,
                                         model_name=model_contain.MODEL.model_name,
                                         temperature=0.01,
                                         history=[],
                                         top_k=VECTOR_SEARCH_TOP_K,
                                         max_tokens=MAX_TOKENS,
                                         prompt_name="knowledge_base_chat",
                                         score_threshold=SCORE_THRESHOLD,
                                         stream=False)

    contents = ""
    async for data in response.body_iterator:  # Đây là chuỗi JSON
        data = json.loads(data)
        contents += data["answer"]
        docs = data["docs"]
    return contents


_PROMPT_TEMPLATE = """
Người dùng sẽ đặt một câu hỏi mà bạn cần tìm trong cơ sở kiến thức, bạn cần hiểu và phân tích câu hỏi và tìm kiếm nội dung liên quan trong cơ sở kiến thức.

Đối với mỗi cơ sở kiến thức, nội dung bạn xuất ra phải là một chuỗi trên một dòng, chứa tên cơ sở kiến thức và nội dung tìm kiếm, cách nhau bởi dấu phẩy, không có chữ cái và ký tự đặc biệt không cần thiết khác, ví dụ như không có dấu phẩy tiếng Việt.

Ví dụ:

robotic, tỷ lệ nam nữ trong robot là bao nhiêu
bigdata, tình hình việc làm trong lĩnh vực dữ liệu lớn như thế nào

Dưới đây là các cơ sở kiến thức bạn có thể truy cập, phần sau dấu hai chấm là chức năng của chúng, bạn nên tham khảo chúng để giúp quá trình suy nghĩ của bạn

{database_names}

Định dạng câu trả lời của bạn nên tuân thủ theo nội dung dưới đây, hãy chú ý rằng các đánh dấu như ```text cần được bao gồm, đây là cách để trích xuất câu trả lời.
Không in ra dấu phẩy tiếng Việt, không in ra dấu ngoặc kép.

Question: ${{câu hỏi của người dùng}}

```text
${{tên cơ sở kiến thức, câu hỏi tìm kiếm, không có ký tự đặc biệt ngoài dấu phẩy, không in dấu phẩy tiếng Việt, không in dấu ngoặc kép}}
```output
Kết quả tìm kiếm từ cơ sở dữ liệu

Bắt đầu tìm kiếm
Câu hỏi: {question}
"""
PROMPT = PromptTemplate(
    input_variables=["question", "database_names"],
    template=_PROMPT_TEMPLATE,
)


class LLMKnowledgeChain(LLMChain):
    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    prompt: BasePromptTemplate = PROMPT
    database_names: Dict[str, str] = model_contain.DATABASE
    input_key: str = "question"
    output_key: str = "answer"

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Khởi tạo trực tiếp một LLMKnowledgeChain với một llm đã bị loại bỏ. "
                "Vui lòng khởi tạo với đối số llm_chain hoặc sử dụng phương thức from_llm."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                prompt = values.get("prompt", PROMPT)
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _evaluate_expression(self, dataset, query) -> str:
        try:
            output = asyncio.run(search_knowledge_base_iter(dataset, query))
        except Exception as e:
            output = "Thông tin nhập không đúng hoặc không tìm thấy cơ sở kiến thức"
            return output
        return output

    def _process_llm_result(
            self,
            llm_output: str,
            llm_input: str,
            run_manager: CallbackManagerForChainRun
    ) -> Dict[str, str]:

        run_manager.on_text(llm_output, color="green", verbose=self.verbose)

        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            database = text_match.group(1).strip()
            output = self._evaluate_expression(database, llm_input)
            run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            return {self.output_key: f"Định dạng nhập không đúng: {llm_output}"}
        return {self.output_key: answer}

    async def _aprocess_llm_result(
            self,
            llm_output: str,
            run_manager: AsyncCallbackManagerForChainRun,
    ) -> Dict[str, str]:
        await run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            await run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            await run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            raise ValueError(f"Định dạng không xác định từ LLM: {llm_output}")
        return {self.output_key: answer}

    def _call(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(inputs[self.input_key])
        data_formatted_str = ',\n'.join([f' "{k}":"{v}"' for k, v in self.database_names.items()])
        llm_output = self.llm_chain.predict(
            database_names=data_formatted_str,
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return self._process_llm_result(llm_output, inputs[self.input_key], _run_manager)

    async def _acall(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(inputs[self.input_key])
        data_formatted_str = ',\n'.join([f' "{k}":"{v}"' for k, v in self.database_names.items()])
        llm_output = await self.llm_chain.apredict(
            database_names=data_formatted_str,
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return await self._aprocess_llm_result(llm_output, inputs[self.input_key], _run_manager)

    @property
    def _chain_type(self) -> str:
        return "llm_knowledge_chain"

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: BasePromptTemplate = PROMPT,
            **kwargs: Any,
    ) -> LLMKnowledgeChain:
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)


def search_knowledgebase_once(query: str):
    model = model_contain.MODEL
    llm_knowledge = LLMKnowledgeChain.from_llm(model, verbose=True, prompt=PROMPT)
    ans = llm_knowledge.run(query)
    return ans


class KnowledgeSearchInput(BaseModel):
    location: str = Field(description="Câu hỏi cần tìm kiếm")


if __name__ == "__main__":
    result = search_knowledgebase_once("Tỉ lệ nam nữ trong lĩnh vực dữ liệu lớn")
    print(result)
