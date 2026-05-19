# Module: Backend Semantic Cache (High-Performance Semantic Cache)

## 1. Mô tả chi tiết (Detailed Description)
Module này chịu trách nhiệm lưu trữ và tra cứu câu trả lời dựa trên ngữ nghĩa (**Semantic Caching**), giúp hệ thống tăng tốc phản hồi đáng kể và giảm thiểu chi phí chạy mô hình LLM. Được định nghĩa trong lớp `SemanticCache`, thành phần này sử dụng một cơ sở dữ liệu **SQLite** cục bộ kết hợp với việc tính toán **Độ tương đồng Cosine (Cosine Similarity)** trực tiếp bằng thư viện **NumPy**.

Kiến trúc hoạt động của `SemanticCache` được thiết kế vô cùng tinh tế qua các đặc trưng kỹ thuật sau:
1.  **Lưu trữ Vector nhị phân (Binary BLOB Storage)**: SQLite không có kiểu dữ liệu vector chuyên dụng. Để lưu trữ các mảng số thực float32 của vector embedding, module chuyển đổi mảng NumPy sang dạng chuỗi byte thô (`numpy.ndarray.tobytes()`) và ghi vào cột kiểu `BLOB` (Binary Large Object) của SQLite. Khi đọc ra, hệ thống chuyển ngược từ chuỗi byte sang mảng số thực bằng `numpy.frombuffer()`, đảm bảo tốc độ ghi đọc và tính chính xác tuyệt đối.
2.  **Context-Aware Caching (Bộ đệm theo ngữ cảnh hội thoại)**: Một thử thách của cache trong chatbot là cùng một câu hỏi *"cổ phiếu này có mua được không?"* nhưng ở các thời điểm hoặc ngữ cảnh trước đó khác nhau sẽ có câu trả lời khác nhau. Module giải quyết triệt để bằng cột `context_hash` (mã băm SHA-256 từ chuỗi lịch sử tin nhắn). Khi tra cứu:
    - Đầu tiên, hệ thống lọc các bản ghi có cùng `context_hash` (khớp chính xác bối cảnh hội thoại).
    - Tiến hành tính độ tương đồng vector của câu hỏi đã viết lại (`rewritten_query`).
    - Nếu không tìm thấy bản ghi khớp ngữ cảnh, hệ thống mới quét rộng ra các bản ghi tổng quát (global cache) để tìm kiếm các câu trả lời chung.
3.  **Heuristic Similarity Thresholding (Ngưỡng tương đồng linh hoạt)**: Hệ thống định cấu hình ngưỡng `similarity_threshold` (mặc định **0.85**). Bất kỳ câu hỏi nào có độ tương đồng lớn hơn hoặc bằng ngưỡng này sẽ lập tức tạo ra một **Cache Hit**, phản hồi ngay câu trả lời đã lưu cho người dùng mà không cần gọi LLM sinh lại.
4.  **Preventing Cache Bloat (Chống rác bộ đệm)**: Để tránh lưu trữ các câu trả lời trùng lặp hoặc các cuộc hội thoại phiếm tổng quát gây phình to cơ sở dữ liệu, module triển khai bộ lọc kép:
    - Chỉ cho phép lưu các giao dịch có chủ đề chuyên sâu (`KNOWLEDGE` hoặc `FINANCIAL`), bỏ qua `GENERAL` (như chào hỏi xã giao).
    - Trước khi lưu một bản ghi mới, gọi hàm `has_similar_response()` so sánh vector câu trả lời mới sinh với toàn bộ kho câu trả lời cũ bằng ngưỡng cực cao (**0.98**). Nếu câu trả lời mới gần như y hệt một phản hồi đã có, hệ thống sẽ bỏ qua không lưu bản ghi mới này.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Khởi tạo Database**: Thiết lập và tạo bảng dữ liệu `cache` trong file `data/semantic_cache.sqlite` nếu chưa tồn tại.
-   **Tra cứu Ngữ nghĩa (`lookup`)**:
    - Truy vấn các bản ghi ứng viên tiềm năng từ SQLite.
    - Chuyển đổi BLOB sang mảng vector NumPy.
    - Tính toán độ tương đồng Cosine giữa vector truy vấn và vector bản ghi.
    - Trả về phản hồi của bản ghi có điểm tương đồng cao nhất vượt ngưỡng cấu hình.
-   **Lưu trữ Giao dịch (`store`)**:
    - Tuần tự hóa vector nhúng câu hỏi và phản hồi sang dạng nhị phân.
    - Thực thi chèn bản ghi mới kèm các siêu dữ liệu (`session_id`, `context_hash`, `original_query`, `rewritten_query`, `route`, `response`).
-   **Kiểm tra trùng lặp (`has_similar_response`)**: Duyệt tìm xem câu trả lời chuẩn bị lưu có bị trùng lặp ngữ nghĩa với bất kỳ câu trả lời nào đã tồn tại trong DB không để tối ưu hóa bộ nhớ.

## 3. Đầu vào (Inputs)
-   `rewritten_query` (chuỗi ký tự): Câu hỏi đã làm sạch.
-   `query_vec` / `response_vec` (mảng số thực NumPy): Các vector embedding biểu diễn ngữ nghĩa.
-   `session_id` (chuỗi ký tự) và `context_hash` (chuỗi ký tự mã băm lịch sử).
-   `route` (chủ đề phân loại) và `response` (văn bản câu trả lời).

## 4. Đầu ra (Outputs)
-   **Khi tra cứu**: Trả về một tuple `(response, original_query, score, route)` nếu có Cache Hit, hoặc `None` nếu Cache Miss.
-   **Khi kiểm tra trùng**: Trả về giá trị Boolean (`True`/`False`).

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [semantic_cache.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/semantic_cache.py)
-   **Đường dẫn tương đối**: `./backend/src/utils/semantic_cache.py`
-   **Tệp tin cơ sở dữ liệu**: `backend/data/semantic_cache.sqlite`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Config.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Config.md) (`./Backend_Config.md`): Tải cấu hình đường dẫn SQLite database và ngưỡng tương đồng `similarity_threshold` từ `AppConfig`.
    -   [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Sử dụng logger để ghi nhận các lỗi kết nối cơ sở dữ liệu SQLite hoặc lỗi đọc ghi.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Pipeline.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Pipeline.md) (`./Backend_Pipeline.md`): Chat Pipeline gọi `cache.lookup()` ở giai đoạn đầu để tìm câu trả lời sẵn có, gọi `cache.has_similar_response()` để tránh lưu lặp, và gọi `cache.store()` ở giai đoạn cuối để lưu trữ giao dịch thành công.
