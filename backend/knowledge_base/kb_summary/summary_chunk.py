import asyncio
import sys
from typing import List, Optional

from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import (
    MapReduceDocumentsChain, ReduceDocumentsChain)
from langchain.docstore.document import Document
from langchain.output_parsers.regex import RegexParser
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel

from backend.knowledge_base.model.kb_document_model import DocumentWithVSId
from configs.basic_config import logger


class SummaryAdapter:
    _OVERLAP_SIZE: int
    token_max: int
    _separator: str = "\n\n"
    chain: MapReduceDocumentsChain

    def __init__(self, overlap_size: int, token_max: int,
                 chain: MapReduceDocumentsChain):
        self._OVERLAP_SIZE = overlap_size
        self.chain = chain
        self.token_max = token_max

    @classmethod
    def form_summary(cls,
                     llm: BaseLanguageModel,
                     reduce_llm: BaseLanguageModel,
                     overlap_size: int,
                     token_max: int = 1300):
        """
        Lấy một phiên bản
        :param reduce_llm: llm được sử dụng để kết hợp tóm tắt
        :param llm: llm được sử dụng để tạo tóm tắt
        :param overlap_size: kích thước phần chồng chéo
        :param token_max: Số lượng chunk tối đa, mỗi chunk có độ dài nhỏ hơn độ dài token_max,
               tóm tắt lớn hơn token_max sẽ gây ra lỗi khi tạo tóm tắt lần đầu
        :return:
        """

        # Điều này điều khiển cách mỗi tài liệu sẽ được định dạng. Cụ thể là,
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )

        # Phần này nên nhận biến đầu vào là `document_variable_name`
        prompt_template = (
            "Thực hiện các công việc dựa trên văn bản. Thông tin công việc như sau" 
            "{task_briefing}" 
            "Nội dung văn bản như sau: "
            "\r\n"
            "{context}"
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["task_briefing", "context"]
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # Bây giờ chúng ta xác định cách kết hợp các tóm tắt này
        reduce_prompt = PromptTemplate.from_template(
            "Kết hợp các tóm tắt này: {context}"
        )
        reduce_llm_chain = LLMChain(llm=reduce_llm, prompt=reduce_prompt)

        document_variable_name = "context"
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        )
        reduce_documents_chain = ReduceDocumentsChain(
            token_max=token_max,
            combine_documents_chain=combine_documents_chain,
        )
        chain = MapReduceDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name=document_variable_name,
            reduce_documents_chain=reduce_documents_chain,
            # Trả về các bước trung gian
            return_intermediate_steps=True
        )
        return cls(overlap_size=overlap_size,
                   chain=chain,
                   token_max=token_max)

    def summarize(self,
                  file_description: str,
                  docs: List[DocumentWithVSId] = []
                  ) -> List[Document]:

        if sys.version_info < (3, 10):
            loop = asyncio.get_event_loop()
        else:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()

            asyncio.set_event_loop(loop)
        # Gọi các hàm coroutines đồng bộ
        return loop.run_until_complete(self.asummarize(file_description=file_description,
                                                       docs=docs))

    async def asummarize(self,
                         file_description: str,
                         docs: List[DocumentWithVSId] = []) -> List[Document]:

        logger.info("Bắt đầu tóm tắt")
        """
        Quá trình này chia thành hai phần:
        1. Xử lý từng tài liệu để có được tóm tắt cho từng tài liệu
         map_results = self.llm_chain.apply(
            # FYI - Đây là quá trình song song nên nó rất nhanh.
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        2. Kết hợp các tóm tắt của từng tài liệu để có được tóm tắt cuối cùng, return_intermediate_steps=True, trả về các bước trung gian
        result, extra_return_dict = self.reduce_documents_chain.combine_docs(
            result_docs, token_max=token_max, callbacks=callbacks, **kwargs
        )
        """
        summary_combine, summary_intermediate_steps = self.chain.combine_docs(docs=docs,
                                                                              task_briefing="Mô tả sự tiếp cận và đồng nhất giữa các phương pháp khác nhau, "
                                                                                            "để giúp độc giả hiểu rõ mối quan hệ giữa chúng.")
        print(summary_combine)
        print(summary_intermediate_steps)

        # if len(summary_combine) == 0:
        #     # Nếu rỗng, tạo lại, giảm số lượng đi một nửa
        #     result_docs = [
        #         Document(page_content=question_result_key, metadata=docs[i].metadata)
        #         # Sử dụng metadata từ các tài liệu và kết quả văn bản từ `results`
        #         for i, question_result_key in enumerate(
        #             summary_intermediate_steps["intermediate_steps"][
        #             :len(summary_intermediate_steps["intermediate_steps"]) // 2
        #             ])
        #     ]
        #     summary_combine, summary_intermediate_steps = self.chain.reduce_documents_chain.combine_docs(
        #         result_docs, token_max=self.token_max
        #     )
        logger.info("Kết thúc tóm tắt")
        doc_ids = ",".join([doc.id for doc in docs])
        _metadata = {
            "file_description": file_description,
            "summary_intermediate_steps": summary_intermediate_steps,
            "doc_ids": doc_ids
        }
        summary_combine_doc = Document(page_content=summary_combine, metadata=_metadata)

        return [summary_combine_doc]

    def _drop_overlap(self, docs: List[DocumentWithVSId]) -> List[str]:
        """
         # Loại bỏ các phần lặp lại của câu trong page_content của tài liệu
        :param docs:
        :param separator:
        :return:
        """
        merge_docs = []

        pre_doc = None
        for doc in docs:
            # Thêm trực tiếp tài liệu đầu tiên
            if len(merge_docs) == 0:
                pre_doc = doc.page_content
                merge_docs.append(doc.page_content)
                continue

            # Loại bỏ các phần lặp lại giữa kết thúc của tài liệu trước và bắt đầu của tài liệu tiếp theo
            # Lặp giảm dần kích thước của pre_doc, mỗi lần lặp loại bỏ một ký tự đầu tiên,
            # Kiểm tra phần trùng lặp, cho đến khi kích thước của pre_doc nhỏ hơn _OVERLAP_SIZE // 2 - 2 * len(separator)
            for i in range(len(pre_doc), self._OVERLAP_SIZE // 2 - 2 * len(self._separator), -1):
                # Mỗi lần lặp loại bỏ một ký tự đầu tiên
                pre_doc = pre_doc[1:]
                if doc.page_content[:len(pre_doc)] == pre_doc:
                    # Loại bỏ phần bắt đầu trùng lặp của tài liệu tiếp theo
                    merge_docs.append(doc.page_content[len(pre_doc):])
                    break

            pre_doc = doc.page_content

        return merge_docs

    def _join_docs(self, docs: List[str]) -> Optional[str]:
        text = self._separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text


if __name__ == '__main__':

    docs = [

        'Người mơ có vai trò đặc biệt, có nghĩa là giấc mơ là dự báo tương lai. Do đó, nội dung của giấc mơ',

        'Nội dung mơ phong phú và ấn tượng đặc biệt để lại ấn tượng đặc biệt cho người mơ, làm cho họ khó có thể tưởng tượng',

        'làm cho họ khó có thể tưởng tượng ra một tập hợp hệ thống hóa một cách thống nhất, cần phải dựa vào giá trị và đáng tin cậy riêng biệt của họ để thực hiện các loại phân hoá và tổng hợp khác nhau. Do đó, các nhà triết học cổ điển đã đánh giá cao giấc mơ một cách hoàn toàn',

    ]
    _OVERLAP_SIZE = 1
    separator: str = "\n\n"
    merge_docs = []
    # Loại bỏ các phần lặp lại của câu trong page_content của tài liệu,
    # Lặp giảm dần kích thước của pre_doc, mỗi lần lặp loại bỏ một ký tự đầu tiên,
    # Kiểm tra phần trùng lặp, cho đến khi kích thước của pre_doc nhỏ hơn _OVERLAP_SIZE-2len(separator)
    pre_doc = None
    for doc in docs:
        # Thêm trực tiếp tài liệu đầu tiên
        if len(merge_docs) == 0:
            pre_doc = doc
            merge_docs.append(doc)
            continue

        # Loại bỏ các phần lặp lại giữa kết thúc của tài liệu trước và bắt đầu của tài liệu tiếp theo
        # Lặp giảm dần kích thước của pre_doc, mỗi lần lặp loại bỏ một ký tự đầu tiên,
        # Kiểm tra phần trùng lặp, cho đến khi kích thước của pre_doc nhỏ hơn _OVERLAP_SIZE-2len(separator)
        for i in range(len(pre_doc), _OVERLAP_SIZE - 2 * len(separator), -1):
            # Mỗi lần lặp loại bỏ một ký tự đầu tiên
            pre_doc = pre_doc[1:]
            if doc[:len(pre_doc)] == pre_doc:
                # Loại bỏ phần bắt đầu trùng lặp của tài liệu tiếp theo
                page_content = doc[len(pre_doc):]
                merge_docs.append(page_content)

                pre_doc = doc
                break

    # Kết hợp các câu trong merge_docs thành một tài liệu
    text = separator.join(merge_docs)
    text = text.strip()

    print(text)
