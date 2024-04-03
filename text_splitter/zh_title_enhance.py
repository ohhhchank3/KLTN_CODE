import re

from langchain.docstore.document import Document


def under_non_alpha_ratio(text: str, threshold: float = 0.5):
    """Kiểm tra xem tỷ lệ ký tự không phải chữ cái trong đoạn văn bản có vượt quá ngưỡng
    cho phép hay không. Điều này giúp ngăn chặn việc nhận diện văn bản như "-----------BREAK---------"
    là tiêu đề hoặc văn bản nội dung. Tỷ lệ này không tính khoảng trắng.

    Parameters
    ----------
    text
        Chuỗi văn bản đầu vào cần kiểm tra
    threshold
        Ngưỡng tỷ lệ ký tự không phải chữ cái. Nếu tỷ lệ này vượt quá ngưỡng, hàm sẽ trả về False
    """
    if len(text) == 0:
        return False

    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    total_count = len([char for char in text if char.strip()])
    try:
        ratio = alpha_count / total_count
        return ratio < threshold
    except:
        return False


def is_possible_title(
    text: str,
    title_max_word_length: int = 20,
    non_alpha_threshold: float = 0.5,
) -> bool:
    """Kiểm tra xem văn bản có phù hợp là tiêu đề hay không.

    Parameters
    ----------
    text
        Chuỗi văn bản đầu vào cần kiểm tra
    title_max_word_length
        Số lượng từ tối đa mà một tiêu đề có thể chứa
    non_alpha_threshold
        Số lượng ký tự chữ cái tối thiểu cần thiết để văn bản được coi là tiêu đề
    """

    # Nếu độ dài văn bản bằng 0, chắc chắn không phải tiêu đề
    if len(text) == 0:
        print("Không phải tiêu đề. Văn bản trống.")
        return False

    # Nếu văn bản kết thúc bằng dấu câu, không phải tiêu đề
    ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
    ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
    if ENDS_IN_PUNCT_RE.search(text) is not None:
        return False

    # Độ dài văn bản không được vượt quá giới hạn cho phép, mặc định là 20 từ
    if len(text) > title_max_word_length:
        return False

    # Tỷ lệ ký tự số không được quá cao, nếu vượt quá ngưỡng, không phải tiêu đề
    if under_non_alpha_ratio(text, threshold=non_alpha_threshold):
        return False

    # Ngăn chặn việc nhận diện những câu mở đầu như "Chúng tôi xin gửi đến Quý khách hàng," là tiêu đề
    if text.endswith((",", ".", "，", "。")):
        return False

    if text.isnumeric():
        print(f"Không phải tiêu đề. Văn bản toàn số:\n\n{text}")  # type: ignore
        return False

    # Ký tự đầu tiên trong văn bản phải là số, mặc định là 5 ký tự
    if len(text) < 5:
        text_5 = text
    else:
        text_5 = text[:5]
    alpha_in_text_5 = sum(list(map(lambda x: x.isnumeric(), list(text_5))))
    if not alpha_in_text_5:
        return False

    return True


def zh_title_enhance(docs: Document) -> Document:
    title = None
    if len(docs) > 0:
        for doc in docs:
            if is_possible_title(doc.page_content):
                doc.metadata['category'] = 'vn_Title'
                title = doc.page_content
            elif title:
                doc.page_content = f"Phần tiếp theo có liên quan đến ({title}). {doc.page_content}"
        return docs
    else:
        print("Tài liệu không tồn tại")
