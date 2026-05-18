# Module: Backend Conversation (Session & Chat History Memory Manager)

## 1. Mô tả chi tiết (Detailed Description)
Module này chịu trách nhiệm lưu trữ và quản lý trạng thái hội thoại (**Session-based Chat Memory**) của người dùng. Để đảm bảo chatbot có khả năng ghi nhớ bối cảnh (context) của các câu hỏi trước đó và duy trì một cuộc trò chuyện tự nhiên, module cung cấp một trình quản lý bộ nhớ dạng in-memory qua lớp `SessionManager`.

Kiến trúc của `SessionManager` được thiết kế tối ưu với các điểm nổi bật:
1.  **Thread-Safe Operations**: Giao tiếp đồng thời từ nhiều yêu cầu HTTP (có thể từ cùng một người dùng hoặc nhiều người dùng khác nhau) được đồng bộ hóa bảo mật thông qua cơ chế khóa luồng `Lock` từ thư viện `threading`. Mọi tác vụ đọc ghi vào bộ nhớ đều yêu cầu sở hữu khóa để tránh hiện tượng tranh chấp dữ liệu (Race Conditions).
2.  **Memory Leak & Token Bloat Mitigation**: Lưu trữ lịch sử không giới hạn sẽ gây hao tổn RAM hệ thống theo thời gian và tăng chi phí token/VRAM khi đưa toàn bộ lịch sử vào LLM Prompt. `SessionManager` giải quyết triệt để vấn đề này bằng tham số giới hạn `max_history` (mặc định giữ lại tối đa **20 tin nhắn gần nhất**). Khi vượt qua ngưỡng, hệ thống tự động cắt tỉa (slice) các tin nhắn cũ hơn.
3.  **Hỗ trợ Cập nhật Động (Dynamic Modification)**: Cung cấp hàm đặc biệt `update_last_message` cho phép cập nhật lại nội dung của tin nhắn cuối cùng trong lịch sử. Hàm này cực kỳ hữu ích khi `ChatPipeline` đang stream phản hồi thì phát hiện nội dung đầu ra vi phạm bộ lọc an toàn của `GuardrailsManager`. Khi đó, hệ thống sẽ ngắt stream và cập nhật lại câu trả lời cuối cùng trong lịch sử kèm theo lời cảnh báo dừng an toàn.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Quản lý cấu trúc bộ nhớ**: Duy trì một từ điển Python `sessions` ánh xạ từ khóa `session_id` (chuỗi ký tự) sang danh sách các đối tượng tin nhắn `{"role": role, "content": content}`.
-   **Đảm bảo an toàn đa luồng**: Sử dụng khóa `self._lock` bao bọc tất cả các thao tác thay đổi trạng thái danh sách lịch sử.
-   **Truy xuất lịch sử**: Trả về bản sao an toàn (deep copy ở mức nông) của danh sách tin nhắn để tránh các tác động thay đổi ngoài ý muốn từ các module khác.
-   **Thêm tin nhắn**: Tự động tạo khóa phiên mới nếu `session_id` chưa tồn tại và chèn tin nhắn mới. Thực hiện cắt ngắn lịch sử nếu kích thước vượt ngưỡng `max_history`.
-   **Cập nhật tin nhắn cuối**: Kiểm tra tính hợp lệ của vai trò phát ngôn (role) của tin nhắn cuối cùng trước khi ghi đè nội dung mới.
-   **Giải phóng bộ nhớ**: Xóa hoàn toàn bản ghi của một phiên chat qua khóa `session_id` khi nhận yêu cầu làm sạch phiên.

## 3. Đầu vào (Inputs)
-   `session_id` (chuỗi ký tự): Mã phiên hội thoại độc nhất gửi từ client.
-   `role` (chuỗi ký tự): `"user"` hoặc `"assistant"`.
-   `content` / `new_content` (chuỗi ký tự): Nội dung tin nhắn văn bản.

## 4. Đầu ra (Outputs)
-   `List[Dict[str, str]]`: Danh sách các tin nhắn đã lưu theo đúng cấu trúc dữ liệu Role-Content phù hợp làm bối cảnh prompt cho LLM.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [session_manager.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/conversation/session_manager.py)
-   **Đường dẫn tương đối**: `./backend/src/conversation/session_manager.py`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Schemas.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Schemas.md) (`./Backend_Schemas.md`): Sử dụng lược đồ định nghĩa tin nhắn để định kiểu.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Router_Chatbot.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_Chatbot.md) (`./Backend_Router_Chatbot.md`): Gọi `get_history()` khi tiếp nhận yêu cầu chat API để chuẩn bị bối cảnh.
    -   [Backend_Pipeline.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Pipeline.md) (`./Backend_Pipeline.md`): Lưu tin nhắn mới của người dùng ngay sau khi viết lại câu hỏi, lưu phản hồi của AI sau khi stream hoàn tất và cập nhật thông báo an toàn nếu câu trả lời bị ngắt bởi Guardrails.
