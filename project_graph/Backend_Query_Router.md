# Module: Backend Query Router (Intent Routing Classifier)

## 1. Mô tả chi tiết (Detailed Description)
Module này đóng vai trò là "ngã tư điều hướng" (Decision-Making Router) của hệ thống, chịu trách nhiệm phân tích ý định (intent classification) của câu hỏi người dùng đã được làm sạch và chuyển hướng xử lý sang luồng tối ưu nhất. Được định nghĩa trong lớp `RouterManager`, module này thực hiện phân loại câu hỏi thô thành một trong **3 nhánh chủ đề chính xác**:

1.  **`KNOWLEDGE` (Nhánh Tri thức doanh nghiệp/RAG)**:
    - Các câu hỏi liên quan đến tài liệu, quy định nội bộ, chính sách giao dịch, hướng dẫn sử dụng sản phẩm hoặc thông tin chi tiết có trong các file PDF được tải lên hệ thống.
    - Kích hoạt luồng: **Truy xuất vector trong Qdrant -> Rerank -> Bổ sung ngữ cảnh RAG -> Sinh câu trả lời**.
2.  **`FINANCIAL` (Nhánh Thị trường / Tài chính chung)**:
    - Các câu hỏi liên quan đến bảng giá cổ phiếu, rổ cổ phiếu VN30, biểu đồ giá vàng, giá nến kỹ thuật, phân tích xu hướng thị trường hoặc thông tin tài chính công khai.
    - Kích hoạt luồng: **Truy vấn cơ sở dữ liệu/API thị trường trực tiếp -> Sinh câu trả lời phân tích chuyên môn**.
3.  **`GENERAL` (Nhánh Chào hỏi / Xã giao)**:
    - Các câu hỏi thông thường, chào hỏi xã giao, đùa vui hoặc trò chuyện phiếm không liên quan đến chuyên ngành chứng khoán hay dữ liệu nội bộ công ty.
    - Kích hoạt luồng: **LLM trả lời trực tiếp bỏ qua RAG và bỏ qua lưu trữ Semantic Cache để tối ưu hóa tài nguyên**.

Cơ chế phân loại được thiết lập tối ưu: sử dụng một mô hình NLP phân loại chuyên biệt (hoặc prompt chỉ dẫn LLM gọn nhẹ) nạp cục bộ trên GPU, giúp đưa ra quyết định rẽ nhánh tức thì mà không làm tăng đáng kể độ trễ (latency) của toàn bộ chu kỳ phản hồi.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Quản lý cấu hình định tuyến**: Đọc và áp dụng các mô tả chủ đề định tuyến cùng cài đặt mô hình từ tệp cấu hình `router.yaml`.
-   **Nạp & Giải phóng Mô hình**: Quản lý quy trình nạp lười mô hình định tuyến vào bộ nhớ GPU (`ensure_loaded`) và thu hồi VRAM (`unload`) khi chuyển sang chế độ nạp tài liệu.
-   **Phân loại ý định (`classify`)**:
    - Nhận câu hỏi đã được viết lại bối cảnh đầy đủ.
    - Chạy mô hình phân loại (hoặc gọi suy luận prompt phân loại nhanh).
    - Ánh xạ kết quả về đúng 1 trong 3 hằng số chuỗi: `"KNOWLEDGE"`, `"FINANCIAL"`, hoặc `"GENERAL"`.
-   **Ghi nhận nhật ký quyết định**: Xuất log ghi rõ câu hỏi nào đã được chuyển sang nhánh nào kèm theo điểm số tự tin (confidence score) phục vụ cho việc giám sát chất lượng hệ thống.

## 3. Đầu vào (Inputs)
-   `text` (chuỗi ký tự): Câu hỏi của người dùng (thường là câu đã được viết lại rõ đại từ).
-   Cấu hình định nghĩa route từ `router.yaml`.

## 4. Đầu ra (Outputs)
-   Nhãn chủ đề định tuyến: `"KNOWLEDGE"`, `"FINANCIAL"`, hoặc `"GENERAL"`.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [router.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/router.py)
-   **Đường dẫn tương đối**: `./backend/src/utils/router.py`
-   **File cấu hình liên quan**: [router.yaml](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/config/router.yaml)

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Config.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Config.md) (`./Backend_Config.md`): Đọc cấu hình danh sách route định nghĩa và mô hình phân loại thông qua `AppConfig.get_router_config()`.
    -   [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Ghi log kết quả định tuyến chủ đề cho từng lượt chat, cũng như lỗi tải mô hình.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Main.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Main.md) (`./Backend_Main.md`): Làm ấm (warmup) nạp trước mô hình định tuyến khi hệ thống khởi chạy.
    -   [Backend_Pipeline.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Pipeline.md) (`./Backend_Pipeline.md`): Gọi `router.classify()` để phân nhánh luồng xử lý câu hỏi sau khi tra cứu bộ đệm bị trượt (Cache Miss). Giao dịch trích vết Trace cũng lấy nhãn định tuyến này để hiển thị trên debug UI.
    -   [Backend_App_Mode.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_App_Mode.md) (`./Backend_App_Mode.md`): Giải phóng bộ nhớ của mô hình định tuyến khỏi VRAM khi hệ thống chuyển sang chế độ indexing tài liệu, hoặc nạp lại mô hình khi chuyển về chế độ chat.
