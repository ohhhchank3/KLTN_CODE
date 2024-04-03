import logging
import re
from typing import Any, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class VietnameseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "[.!?。！？]",
            "\.\s|\!\s|\?\s",
            ";\s",
            ",\s",
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip() != ""]


if __name__ == "__main__":
    text_splitter = VietnameseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=50,
        chunk_overlap=0
    )
    ls = [
        """Báo cáo tình hình Thương mại Quốc tế của Trung Quốc (75 trang). Trong 10 tháng đầu năm, kim ngạch thương mại tổng cộng đạt 19.5 nghìn tỷ nhân dân tệ, tăng 25.1% so với cùng kỳ, vượt mức tăng trưởng tổng thể 2.9 điểm phần trăm, chiếm tỷ trọng 61.7% trong tổng kim ngạch thương mại. Trong đó, kim ngạch xuất khẩu hàng thường 10.6 nghìn tỷ nhân dân tệ, tăng 25.3%, chiếm tỷ trọng 60.9% trong kim ngạch xuất khẩu tổng cộng, tăng 1.5 điểm phần trăm; kim ngạch nhập khẩu 8.9 nghìn tỷ nhân dân tệ, tăng 24.9%, chiếm tỷ trọng 62.7% trong kim ngạch nhập khẩu tổng cộng, tăng 1.8 điểm phần trăm. Kim ngạch thương mại gia công 6.8 nghìn tỷ nhân dân tệ, tăng 11.8%, chiếm tỷ trọng 21.5% trong kim ngạch thương mại tổng cộng, giảm 2.0 điểm phần trăm. Trong đó, kim ngạch xuất khẩu tăng 10.4%, chiếm tỷ trọng 24.3%, giảm 2.6 điểm phần trăm; kim ngạch nhập khẩu tăng 14.2%, chiếm tỷ trọng 18.0%, giảm 1.2 điểm phần trăm. Ngoài ra, kim ngạch nhập xuất qua cảng hàng không kho bãi đạt 3.96 nghìn tỷ nhân dân tệ, tăng 27.9%. Trong đó, kim ngạch xuất khẩu 1.47 nghìn tỷ nhân dân tệ, tăng 38.9%; kim ngạch nhập khẩu 2.49 nghìn tỷ nhân dân tệ, tăng 22.2%. Trong 3 quý đầu năm, thương mại dịch vụ của Trung Quốc tiếp tục duy trì tăng trưởng nhanh chóng. Tổng kim ngạch thương mại dịch vụ đạt 37834.3 tỷ nhân dân tệ, tăng 11.6%; trong đó, kim ngạch xuất khẩu dịch vụ đạt 17820.9 tỷ nhân dân tệ, tăng 27.3%; kim ngạch nhập khẩu đạt 20013.4 tỷ nhân dân tệ, tăng 0.5%, đây là lần đầu tiên kể từ dịch bệnh mà tốc độ tăng trưởng nhập khẩu dịch vụ đã chuyển từ âm sang dương. Sự chênh lệch tăng trưởng giữa kim ngạch xuất khẩu và nhập khẩu dịch vụ lên đến 26.8 điểm phần trăm, giúp giảm thiểu thâm hụt thương mại dịch vụ xuống còn 2192.5 tỷ nhân dân tệ. Cấu trúc thương mại dịch vụ ngày càng cải thiện, kim ngạch dịch vụ tập trung vào các lĩnh vực có nền tảng kiến thức cao đạt 16917.7 tỷ nhân dân tệ, tăng 13.3%, chiếm tỷ trọng 44.7% trong tổng kim ngạch thương mại dịch vụ, tăng 0.7 điểm phần trăm. Thứ hai, phân tích và triển vọng môi trường thương mại quốc tế của Trung Quốc. Biến động thất thường của dịch bệnh trên toàn cầu, tình hình phục hồi kinh tế chia rẽ, giá hàng hóa chính tăng, nguồn năng lượng khan hiếm, vận chuyển khan hiếm và sự lan rộng điều chỉnh chính sách từ các nền kinh tế phát triển khác nhau đã gây ra rủi ro kết hợp và gia tăng. Đồng thời, cũng cần nhận ra rằng, xu hướng tốt của nền kinh tế Trung Quốc không thay đổi, sức mạnh và linh hoạt của các doanh nghiệp xuất nhập khẩu ngày càng tăng, các mô hình kinh doanh và kỹ thuật mới đang phát triển nhanh chóng, và bước chuyển đổi sáng tạo đang diễn ra nhanh chóng. Chuỗi cung ứng công nghiệp đối mặt với thách thức. Hoa Kỳ, Châu Âu và các nước khác đang nhanh chóng ban hành kế hoạch trở về sản xuất, tăng cường quy mô cung ứng chuỗi cung ứng công nghiệp, điều chỉnh chuỗi cung ứng công nghiệp toàn cầu, và chuỗi cung ứng kép toàn cầu đang trải qua một vòng đổi mới, xu hướng hóa địa phương, gần bờ, địa phương và ngắn hóa ngắn rõ ràng hơn. Áp lực tại chuỗi cung ứng công nghiệp toàn cầu do thiếu vắc xin, "thiếu chip" trong sản xuất, giới hạn vận chuyển, giá cước vận chuyển cao, v.v., đang ngày càng tăng. Lạm phát toàn cầu duy trì ở mức cao. Sự tăng giá năng lượng đã tạo ra áp lực lớn cho lạm phát ở các nền kinh tế lớn, tăng thêm sự không chắc chắn trong việc phục hồi kinh tế toàn cầu. Báo cáo về triển vọng thị trường hàng hóa của Ngân hàng Thế giới vào tháng 10 năm nay chỉ ra rằng, giá năng lượng đã tăng hơn 80% trong năm 2021 và dự kiến sẽ tăng nhẹ trong năm 2022. IMF chỉ ra rằng, rủi ro lạm phát toàn cầu đang tăng, tương lai của lạm phát có sự không chắc chắn lớn."""
    ]
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            print(chunk)
