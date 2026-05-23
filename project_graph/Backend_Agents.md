# Module: Backend Agents (AI Specialized Agents)

## 1. Mô tả chi tiết (Detailed Description)
Module này đóng vai trò là lớp Agent chuyên biệt (**AI Agent Layer**), quản lý các tác vụ xử lý thông minh nâng cao đòi hỏi khả năng lập luận phân tích tài chính hoặc xử lý ngôn ngữ tự nhiên phức tạp. Module được định nghĩa trong lớp `FinancialAgent`, đóng vai trò trung gian phối hợp giữa các prompt hệ thống chuyên môn cao và mô hình ngôn ngữ lớn chính (**Core LLM**).

Hiện tại, `FinancialAgent` chịu trách nhiệm thực thi 2 nhóm nghiệp vụ cốt lõi:
1.  **Phối hợp Sinh câu trả lời**: Đóng vai trò là đầu mối tiếp nhận câu hỏi của người dùng (đã được làm sạch và mở rộng bối cảnh RAG) cùng lịch sử hội thoại, truyền trực tiếp tới mô hình LLM chính để sinh ra phản hồi phân tích tài chính và chứng khoán sâu sắc.
2.  **Viết lại câu hỏi truy vấn (Context-Aware Query Rewriting)**:
    Đây là một cơ chế cực kỳ quan trọng giúp chatbot giải quyết bài toán mất ngữ cảnh khi người dùng giao tiếp tự nhiên. Khi người dùng nói *"Vinamilk là gì?"* tiếp theo là *"Cổ phiếu của nó có tốt không?"*, cụm từ *"của nó"* sẽ khiến các hệ thống tìm kiếm vector hoặc LLM bị bối rối nếu tách rời. 
    Agent này sẽ:
    - Trích xuất **4 tin nhắn gần nhất** (tương đương 2 lượt hội thoại) để làm ngữ cảnh và tiết kiệm token/VRAM.
    - Cấu trúc một prompt chỉ thị đặc biệt yêu cầu mô hình LLM đóng vai trò là một chuyên gia ngôn ngữ học viết lại câu truy vấn.
    - Ép buộc mô hình phản hồi dưới cấu trúc **JSON duy nhất** dạng `{"rewritten_query": "..."}`.
    - Xử lý bóc tách các thẻ markdown (như ````json ... ````), parse JSON an toàn và trích xuất chuỗi câu hỏi đã được làm rõ đại từ (thay thế *"nó"* thành *"công ty Vinamilk"*). Nếu xảy ra lỗi parse, hệ thống tự động fallback về câu hỏi gốc để đảm bảo luồng hội thoại không bị lỗi.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Đại diện giao tiếp LLM**: Nhận yêu cầu và kết nối trực tiếp với thực thể `LLMManager` để chạy suy luận.
-   **Làm rõ đại từ nhân xưng**: Phân tích lịch sử trò chuyện gần nhất, xác định chủ đề được đề cập ngay trước câu hỏi hiện tại, và thay thế các từ viết tắt hoặc đại từ mơ hồ (`nó`, `cái đó`, `họ`, v.v.) bằng tên chủ thể cụ thể.
-   **Kiểm soát định dạng đầu ra**: Áp dụng các luật định dạng nghiêm ngặt để ép LLM trả về JSON thô, xử lý hậu kỳ (regex trích xuất văn bản giữa các thẻ code block) để làm sạch chuỗi trước khi đưa vào parse JSON.
-   **Hệ thống dự phòng an toàn (Fallback)**: Đảm bảo nếu LLM sinh lỗi hoặc trả về chuỗi JSON rỗng, Agent sẽ trả về nguyên văn câu hỏi ban đầu của người dùng để chuỗi RAG tiếp tục hoạt động.

## 3. Đầu vào (Inputs)
-   `user_input` (chuỗi ký tự): Câu hỏi hiện tại của người dùng.
-   `history` (danh sách tin nhắn dạng Role-Content): Nhật ký hội thoại của phiên hiện hành.
-   `system_prompt` (tùy chọn - chuỗi ký tự): Prompts hệ thống chuyên biệt để ghi đè hệ thống mặc định (ví dụ: dùng cho Stock Analyst Agent).

## 4. Đầu ra (Outputs)
-   **Khi chat**: Một generator truyền trực tiếp luồng stream các chunk text từ LLM.
-   **Khi viết lại câu hỏi**: Một chuỗi ký tự duy nhất (`rewritten_query`) đã được chuẩn hóa và bổ sung đầy đủ ngữ cảnh (ví dụ: *"Cổ phiếu của công ty Vinamilk có tốt không?"* thay vì *"Cổ phiếu của nó có tốt không?"*).

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [financial_agent.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/agents/financial_agent.py)
-   **Đường dẫn tương đối**: `./backend/src/agents/financial_agent.py`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_LLM.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_LLM.md) (`./Backend_LLM.md`): Lấy instance kết nối LLM hoạt động qua `get_llm()` và gọi hàm `generate_response()` để sinh văn bản.
    -   [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Sử dụng đối tượng `logger` ghi nhận các dòng log so sánh câu thô và câu viết lại, cũng như log lỗi parse JSON.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Pipeline.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Pipeline.md) (`./Backend_Pipeline.md`): Chat Pipeline gọi `rewrite_query` ngay ở đầu quy trình xử lý và gọi `process_chat` để sinh câu trả lời RAG cuối cùng hoặc câu từ chối an toàn.
