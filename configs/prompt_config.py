PROMPT_TEMPLATES = {
    "llm_chat": {
        "default":
            '{{ input }}',

        "with_history":
            'Dưới đây là một cuộc trò chuyện thân thiện giữa con người và trí tuệ nhân tạo. '
            'Trí tuệ nhân tạo này thích nói chuyện và cung cấp nhiều thông tin cụ thể từ bối cảnh của mình. '
            'Nếu trí tuệ nhân tạo không biết câu trả lời cho một câu hỏi, nó sẽ thật thà nói rằng nó không biết.\n\n'
            'Trò chuyện hiện tại:\n'
            '{history}\n'
            'Con người: {input}\n'
            'Trí tuệ nhân tạo:',

        "py":
            'Bạn là một trợ lý mã thông minh, hãy viết cho tôi một đoạn mã Python đơn giản. \n'
            '{{ input }}',
    },


    "knowledge_base_chat": {
        "default":
            '<Lệnh> Dựa trên thông tin đã biết, hãy trả lời câu hỏi một cách ngắn gọn và chuyên nghiệp. '
            'Nếu không thể lấy câu trả lời từ đó, hãy nói “Không thể trả lời câu hỏi dựa trên thông tin đã biết”,'
            ' không được thêm thông tin giả mạo vào câu trả lời. Câu trả lời phải sử dụng tiếng Việt. </Lệnh>\n'
            '<Thông tin đã biết>{{ context }}</Thông tin đã biết>\n'
            '<Câu hỏi>{{ question }}</Câu hỏi>\n',

        "text":
            '<Lệnh> Dựa trên thông tin đã biết, hãy trả lời câu hỏi một cách ngắn gọn và chuyên nghiệp. '
            'Nếu không thể lấy câu trả lời từ đó, hãy nói “Không thể trả lời câu hỏi dựa trên thông tin đã biết”, '
            'câu trả lời phải sử dụng tiếng Việt. </Lệnh>\n'
            '<Thông tin đã biết>{{ context }}</Thông tin đã biết>\n'
            '<Câu hỏi>{{ question }}</Câu hỏi>\n',

        "empty":  # Khi không tìm thấy dữ liệu trong cơ sở kiến thức
            'Xin vui lòng trả lời câu hỏi của tôi:\n'
            '{{ question }}\n\n',
    },


    "search_engine_chat": {
        "default":
            '<Lệnh> Đây là những thông tin mà tôi tìm thấy trên internet, hãy trích xuất và tổ chức một cách ngắn gọn để trả lời câu hỏi. '
            'Nếu không thể lấy câu trả lời từ đó, hãy nói “Không thể tìm thấy thông tin để trả lời câu hỏi”. </Lệnh>\n'
            '<Thông tin đã biết>{{ context }}</Thông tin đã biết>\n'
            '<Câu hỏi>{{ question }}</Câu hỏi>\n',

        "search":
            '<Lệnh> Dựa trên thông tin đã biết, hãy trả lời câu hỏi một cách ngắn gọn và chuyên nghiệp. '
            'Nếu không thể lấy câu trả lời từ đó, hãy nói “Không thể trả lời câu hỏi dựa trên thông tin đã biết”, '
            'câu trả lời phải sử dụng tiếng Việt. </Lệnh>\n'
            '<Thông tin đã biết>{{ context }}</Thông tin đã biết>\n'
            '<Câu hỏi>{{ question }}</Câu hỏi>\n',
    },


    "agent_chat": {
        "default":
            'Hãy trả lời những câu hỏi sau một cách tốt nhất có thể. Nếu cần, bạn có thể sử dụng một số công cụ một cách thích hợp. '
            'Bạn có truy cập vào các công cụ sau đây:\n\n'
            '{tools}\n\n'
            'Sử dụng định dạng sau:\n'
            'Câu hỏi: câu hỏi đầu vào mà bạn phải trả lời\n'
            'Suy nghĩ: luôn luôn cân nhắc về những gì nên làm và những công cụ nào nên sử dụng.\n'
            'Hành động: hành động cần thực hiện, nên là một trong [{tool_names}]\n'
            'Đầu vào hành động: đầu vào cho hành động\n'
            'Quan sát: kết quả của hành động\n'
            '... (phần Suy nghĩ/Hành động/Đầu vào hành động/Quan sát có thể lặp lại từ không đến nhiều lần)\n'
            'Suy nghĩ: Bây giờ tôi biết câu trả lời cuối cùng\n'
            'Câu trả lời cuối cùng: câu trả lời cuối cùng cho câu hỏi đầu vào\n'
            'Bắt đầu!\n\n'
            'Lịch sử: {history}\n\n'
            'Câu hỏi: {input}\n\n'
            'Suy nghĩ: {agent_scratchpad}',

        "ChatGLM3":
            'Bạn có thể trả lời bằng cách sử dụng các công cụ, hoặc trả lời trực tiếp bằng kiến thức của bạn mà không sử dụng các công cụ. '
            'Trả lời người dùng một cách hữu ích và chính xác nhất có thể.\n'
            'Bạn có truy cập vào các công cụ sau đây:\n'
            '{tools}\n'
            'Sử dụng một chuỗi json để chỉ định một công cụ bằng cách cung cấp một hành động (tên công cụ) '
            'và một đầu vào hành động (đầu vào công cụ).\n'
            'Các giá trị "hành động" hợp lệ: "Câu trả lời cuối cùng" hoặc [{tool_names}]'
            'Chỉ cung cấp MỘT hành động duy nhất cho $JSON_BLOB, như sau:\n\n'
            '```\n'
            '{{{{\n'
            '  "action": $TOOL_NAME,\n'
            '  "action_input": $INPUT\n'
            '}}}}\n'
            '```\n\n'
            'Tuân thủ định dạng sau:\n\n'
            'Câu hỏi: câu hỏi đầu vào cần trả lời\n'
            'Suy nghĩ: xem xét các bước trước và sau\n'
            'Hành động:\n'
            '```\n'
            '$JSON_BLOB\n'
            '```\n'
            'Quan sát: kết quả của hành động\n'
            '... (lặp lại Suy nghĩ/Hành động/Quan sát N lần)\n'
            'Suy nghĩ: Tôi đã biết cách trả lời\n'
            'Hành động:\n'
            '```\n'
            '{{{{\n'
            '  "action": "Câu trả lời cuối cùng",\n'
            '  "action_input": "Câu trả lời cuối cùng cho người dùng"\n'
            '}}}}\n'
            'Bắt đầu! Nhớ luôn trả lời với một chuỗi json hợp lệ của một hành động duy nhất. Sử dụng công cụ nếu cần. '
            'Trả lời trực tiếp nếu phù hợp. Định dạng Hành động:```$JSON_BLOB```sau đó Quan sát:. \n'
            'Lịch sử: {history}\n\n'
            'Câu hỏi: {input}\n\n'
            'Suy nghĩ: {agent_scratchpad}',
    }
}
