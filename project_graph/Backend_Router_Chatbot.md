# Module: Backend Router Chatbot (Chat & Knowledge Management API)

## 1. Mô tả chi tiết (Detailed Description)
Module này định nghĩa bộ định tuyến API chính cho các tính năng liên quan đến hội thoại AI và quản lý tài nguyên tri thức (RAG). Được xây dựng dưới dạng sub-router của FastAPI với tiền tố `/api/chat`, module này phối hợp chặt chẽ với nhiều thành phần cốt lõi của backend để cung cấp các dịch vụ:
1.  **Hội thoại trực tuyến thời gian thực**: Nhận tin nhắn từ người dùng, duy trì ngữ cảnh bằng Session Manager, gửi qua Chat Pipeline và trả về kết quả dạng dòng chảy text (Token Streaming) thông qua `StreamingResponse`. Đặc biệt, API này sẽ chặn yêu cầu nếu hệ thống đang ở chế độ nạp tài liệu (`INDEXING`).
2.  **Quản lý mô hình LLM**: Cung cấp các endpoint xem danh sách mô hình AI, xem mô hình hiện tại đang chạy trên GPU và gửi yêu cầu thay đổi mô hình hoạt động.
3.  **Hệ thống nạp tài liệu ngầm (Asynchronous Indexing)**: Khi nhận được file PDF tải lên, router lưu tạm file, ghi nhận thông tin vào SQLite metadata, sau đó dùng `BackgroundTasks` của FastAPI để kích hoạt quá trình trích xuất văn bản và nhúng vector vào Qdrant chạy ngầm, giải phóng ngay lập tức luồng API cho người dùng.
4.  **Xem vết xử lý (Debugging Pipeline Trace)**: Xuất kết quả xử lý của giao dịch gần nhất (trạng thái an toàn đầu vào/đầu ra, độ tương đồng của cache hit, bộ định tuyến chủ đề) giúp phát triển và sửa lỗi nhanh chóng.
5.  **Chuyển đổi trạng thái hệ thống**: Thiết lập chế độ hoạt động của chatbot (`CHAT` hoặc `INDEXING`) giúp chuyển đổi linh hoạt thiết bị chạy mô hình (GPU sang CPU và ngược lại) để tối ưu dung lượng VRAM.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Định tuyến API Chatbot**:
    -   `GET /api/chat/models`: Trả về danh sách cấu hình các mô hình ngôn ngữ lớn khả dụng.
    -   `GET /api/chat/current-model`: Trả về ID của mô hình đang hoạt động trên bộ nhớ GPU.
    -   `POST /api/chat/model`: Tiếp nhận yêu cầu chuyển đổi mô hình và gọi `LLMManager` thực hiện hoán đổi.
    -   `POST /api/chat`: Tiếp nhận tin nhắn kèm `session_id`, trích xuất lịch sử chat, kiểm tra chế độ chạy và trả về stream văn bản từ `ChatPipeline`.
    -   `GET /api/chat/trace`: Trả về vết chi tiết gần nhất từ `ChatPipeline`.
-   **Đánh chỉ mục tài liệu RAG**:
    -   `POST /api/chat/upload-knowledge`: Tiếp nhận file PDF dưới dạng `UploadFile`, từ chối nếu không ở chế độ `INDEXING` hoặc file sai định dạng. Tạo luồng chạy ngầm để nạp dữ liệu bằng `ingest_pdf()` và dọn dẹp file tạm sau khi nạp xong.
    -   `GET /api/chat/knowledge/files`: Trích xuất danh sách và trạng thái của các file tri thức trong SQLite.
    -   `DELETE /api/chat/knowledge/files/{file_id}`: Xóa tệp tin khỏi cơ sở dữ liệu metadata và đồng bộ xóa các vector thuộc tài liệu đó trong Qdrant.
-   **Điều khiển chế độ hoạt động (App Mode Control)**:
    -   `GET /api/chat/app-mode`: Trả về chế độ hoạt động hiện tại (`CHAT` hoặc `INDEXING`).
    -   `POST /api/chat/app-mode`: Thay đổi chế độ chạy hệ thống và kích hoạt quy trình giải phóng VRAM.

## 3. Đầu vào (Inputs)
-   Các cấu trúc dữ liệu JSON từ request client (như `ChatRequest`, `ModelSelectRequest`, `AppModeRequest`).
-   Tệp tải lên: Dữ liệu nhị phân file PDF (`UploadFile`).
-   Tham số đường dẫn (Path parameters): `file_id` (kiểu số nguyên).

## 4. Đầu ra (Outputs)
-   `StreamingResponse`: Dòng chảy văn bản phản hồi từng chữ từ AI.
-   Phản hồi JSON: Cấu trúc trạng thái, danh sách mô hình, danh sách tệp tin, dữ liệu debug trace hoặc thông báo thành công.
-   Mã lỗi HTTP (400, 403, 404, 500) tương ứng với các tình huống nghiệp vụ sai lệch.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [chatbot.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/routers/chatbot.py)
-   **Đường dẫn tương đối**: `./backend/src/routers/chatbot.py`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Schemas.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Schemas.md) (`./Backend_Schemas.md`): Sử dụng các Pydantic models để kiểm tra tính hợp lệ của request payload.
    -   [Backend_Conversation.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Conversation.md) (`./Backend_Conversation.md`): Gọi Session Manager để trích xuất lịch sử cuộc trò chuyện dựa trên `session_id`.
    -   [Backend_LLM.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_LLM.md) (`./Backend_LLM.md`): Lấy cấu hình models, kiểm tra mô hình hiện tại, và gọi switch_model để hoán đổi mô hình.
    -   [Backend_Pipeline.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Pipeline.md) (`./Backend_Pipeline.md`): Gửi dữ liệu câu hỏi và lịch sử vào `ChatPipeline` để sinh phản hồi AI và trích vết `get_last_trace()`.
    -   [Backend_Knowledge_Base.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Knowledge_Base.md) (`./Backend_Knowledge_Base.md`): Gọi hàm `kb.ingest_pdf()` nạp file vào Qdrant DB và `kb.delete_file_vectors()` để xóa vector tài liệu.
    -   [Backend_App_Mode.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_App_Mode.md) (`./Backend_App_Mode.md`): Kiểm tra chế độ hiện tại, gọi `set_mode()` để hoán chuyển cấu hình phần cứng.
    -   [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Ghi log thông tin lỗi chạy ngầm.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Main.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Main.md) (`./Backend_Main.md`): Đăng ký sub-router này vào FastAPI application.
    -   [Frontend_App.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Frontend_App.md) (`./Frontend_App.md`): Giao tiếp API để thực hiện cuộc trò chuyện và quản trị dữ liệu tri thức.
