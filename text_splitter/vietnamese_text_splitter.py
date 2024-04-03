from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, sentence_size: int = 250, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    def split_text1(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list

    def split_text(self, text: str) -> List[str]:
    # Loại bỏ khoảng trắng và các ký tự thừa
        text = re.sub(r'\s+', ' ', text.strip())

    # Các quy tắc chia câu
        sentence_delimiters = r'(?<=[.!?。！？\?])\s+(?=[^a-zA-Z0-9])'
        text = re.sub(sentence_delimiters, '\n', text)

    # Cắt các câu quá dài thành các đoạn nhỏ hơn
        sentences = text.split('\n')
        max_sentence_length = self.sentence_size
        result = []
        current_chunk = ''

        for sentence in sentences:
           if len(current_chunk) + len(sentence) <= max_sentence_length:
            current_chunk += sentence + '\n'
           else:
            result.append(current_chunk.strip())
            current_chunk = sentence + '\n'

        if current_chunk:
            result.append(current_chunk.strip())

        return result

