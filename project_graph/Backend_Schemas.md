# Module: Backend Schemas (Pydantic API Data Schemas)

## 1. Mô tả chi tiết (Detailed Description)
Module này tập hợp các lược đồ dữ liệu (**Pydantic Data Schemas**) phục vụ việc xác thực (validation), ép kiểu (type coercion) và tuần tự hóa (serialization) các gói tin yêu cầu (requests) và phản hồi (responses) của các API chatbot. Được xây dựng dựa trên thư viện **Pydantic** tiêu chuẩn của FastAPI, module này đảm bảo tính toàn vẹn dữ liệu ở lớp biên API trước khi chuyển sâu vào xử lý logic nghiệp vụ.

Việc tách biệt các mô hình dữ liệu này giúp bảo vệ ứng dụng khỏi các lỗi định dạng không hợp lệ từ client, tự động sinh mã tài liệu hóa API chuẩn OpenAPI (Swagger UI), đồng thời cung cấp gợi ý kiểu (Type Hinting) rõ ràng cho lập trình viên xuyên suốt dự án.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
Định nghĩa cấu trúc và luật xác thực dữ liệu cho các đối tượng Pydantic sau:
-   **`ChatMessage`**: Cấu trúc biểu diễn một tin nhắn đơn lẻ trong lịch sử hội thoại.
    -   `role` (chuỗi ký tự): Vai trò của đối tượng phát ngôn (ví dụ: `"user"`, `"assistant"`, `"system"`).
    -   `content` (chuỗi ký tự): Nội dung chi tiết của tin nhắn văn bản.
-   **`ChatRequest`**: Cấu trúc payload của yêu cầu gửi tin nhắn chat đến AI.
    -   `message` (chuỗi ký tự): Câu hỏi hoặc nội dung tin nhắn mới từ người dùng.
    -   `session_id` (tùy chọn - chuỗi ký tự): Mã phiên hội thoại độc nhất phục vụ lưu trữ ngữ cảnh phía máy chủ.
    -   `history` (tùy chọn - danh sách `ChatMessage`): Lịch sử hội thoại thủ công do client truyền lên (sử dụng làm phương án dự phòng nếu không truyền `session_id`).
-   **`ChatResponse`**: Cấu trúc phản hồi chat đồng bộ dạng văn bản đầy đủ (không stream).
    -   `response` (chuỗi ký tự): Phản hồi trọn vẹn từ AI.
-   **`AvailableModel`**: Biểu diễn một mô hình AI đang sẵn sàng trong cấu hình hệ thống.
    -   `id` (chuỗi ký tự): Mã ID định danh của mô hình (ví dụ: `"qwen-3-8b"`).
    -   `name` (chuỗi ký tự): Tên hiển thị thân thiện trên giao diện (ví dụ: `"Qwen 3 8B (Thinking Model)"`).
-   **`ModelSelectRequest`**: Yêu cầu lựa chọn hoặc chuyển đổi mô hình ngôn ngữ.
    -   `model_id` (chuỗi ký tự): ID của mô hình đích người dùng muốn tải vào GPU.

## 3. Đầu vào (Inputs)
-   Các payload JSON thô (raw HTTP body) nhận được qua các cổng API FastAPI.

## 4. Đầu ra (Outputs)
-   Các đối tượng Pydantic Python có kiểu dữ liệu mạnh (strongly typed), đã được xác thực an toàn.
-   Tự động sinh cấu trúc mô tả API JSON cho hệ thống Swagger `/docs`.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [chat_schema.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/schemas/chat_schema.py)
-   **Đường dẫn tương đối**: `./backend/src/schemas/chat_schema.py`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   Không phụ thuộc trực tiếp vào các module khác ngoài thư viện `pydantic`.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Router_Chatbot.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_Chatbot.md) (`./Backend_Router_Chatbot.md`): Sử dụng các class `ChatRequest`, `ModelSelectRequest`, và `AvailableModel` để xác thực payload đầu vào cho các API tương ứng.
    -   [Backend_Conversation.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Conversation.md) (`./Backend_Conversation.md`): Sử dụng lược đồ cấu trúc tin nhắn để kiểm soát định dạng dữ liệu lưu trữ lịch sử cuộc trò chuyện.
