# Module: Backend Embeddings (Sentence Transformer Embeddings Wrapper)

## 1. Mô tả chi tiết (Detailed Description)
Module này chịu trách nhiệm sinh vector nhúng (**Vector Embeddings**) - cầu nối ngữ nghĩa chuyển đổi các văn bản ngôn ngữ tự nhiên sang dạng mảng số thực float32 có số chiều cố định (ví dụ: 384 hoặc 768 chiều). Được định nghĩa trong lớp `EmbeddingManager`, module này bao bọc thư viện **SentenceTransformers** cục bộ để phục vụ hai tính năng nền tảng của hệ thống:
1.  **Semantic Caching**: Nhúng vector câu hỏi đầu vào của người dùng để tính khoảng cách Cosine so sánh với cơ sở dữ liệu cache SQLite.
2.  **RAG Knowledge Base**: Nhúng vector các mảnh văn bản tài liệu PDF tải lên để lưu vào Qdrant, và nhúng vector câu hỏi của người dùng để tìm kiếm các văn bản có ngữ nghĩa tương quan gần nhất.

Một ưu điểm lớn trong thiết kế của `EmbeddingManager` là khả năng **chuyển đổi thiết bị chạy mô hình động (Dynamic Device Switching)** thông qua hàm `set_device()`. Tính năng này được phối hợp chặt chẽ với Trình quản lý chế độ hệ thống (`AppModeManager`):
-   Khi ở chế độ **Hội thoại (`CHAT`)**: Để tiết kiệm tối đa VRAM GPU dành riêng cho mô hình ngôn ngữ lớn LLM hoạt động, mô hình Embeddings được di chuyển chạy trên **CPU**.
-   Khi ở chế độ **Đánh chỉ mục (`INDEXING`)**: Khi người dùng tải lên tài liệu PDF lớn cần trích xuất hàng ngàn mảnh văn bản và mã hóa nhanh chóng, mô hình Embeddings được chuyển dịch sang chạy trên bộ tăng tốc **GPU (CUDA)**, giúp tăng tốc độ xử lý hàng trăm lần qua cơ chế tính toán song song song hành với tham số `batch_size = 8` hoặc `32`.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Quản lý mô hình SentenceTransformers**: Nạp lười (`ensure_loaded`) mô hình từ đường dẫn cục bộ hoặc tải xuống từ Hugging Face Hub dựa theo cấu hình trong `pipeline.yaml`.
-   **Điều phối thiết bị động**: Thực hiện dịch chuyển mô hình giữa bộ nhớ RAM vật lý (CPU) và bộ nhớ đồ họa VRAM (GPU CUDA) qua hàm `set_device()`.
-   **Mã hóa văn bản (`embed`)**:
    - Nhận vào một chuỗi ký tự đơn lẻ hoặc danh sách nhiều chuỗi văn bản.
    - Gọi hàm `encode()` của SentenceTransformers với các batch phân phối tối ưu.
    - Trả về mảng số thực float32 dạng NumPy đại diện cho ý nghĩa ngữ nghĩa của văn bản.
-   **Giải phóng bộ nhớ**: Xóa liên kết đối tượng và thực thi dọn dẹp CUDA Cache khi có yêu cầu dọn dẹp hệ thống.

## 3. Đầu vào (Inputs)
-   `texts` (chuỗi ký tự đơn lẻ hoặc danh sách các chuỗi ký tự): Văn bản cần mã hóa vector.
-   `batch_size` (tùy chọn - số nguyên): Số lượng văn bản mã hóa đồng thời trong một bước.
-   Cấu hình mô hình nhúng từ `pipeline.yaml` tại nhánh `embedding`.

## 4. Đầu ra (Outputs)
-   `numpy.ndarray`: Mảng vector số thực float32 hai chiều (hoặc một chiều nếu chỉ nhúng một câu), ví dụ kích thước `(N, 384)` hoặc `(384,)`.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [embeddings.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/embeddings.py)
-   **Đường dẫn tương đối**: `./backend/src/utils/embeddings.py`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Config.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Config.md) (`./Backend_Config.md`): Đọc tên mô hình, số chiều vector và thiết bị mặc định thông qua `AppConfig.get_pipeline_config()`.
    -   [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Ghi log tiến trình khởi tạo, đo đếm thời gian sinh vector và ghi nhận lỗi.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Main.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Main.md) (`./Backend_Main.md`): Làm ấm nạp trước mô hình Embeddings cục bộ khi khởi chạy ứng dụng FastAPI.
    -   [Backend_Pipeline.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Pipeline.md) (`./Backend_Pipeline.md`): Mã hóa câu hỏi của người dùng để tra cứu Semantic Cache và truy xuất RAG trong Qdrant.
    -   [Backend_Knowledge_Base.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Knowledge_Base.md) (`./Backend_Knowledge_Base.md`): Mã hóa các mảnh tài liệu PDF được cắt nhỏ để đẩy lên Qdrant DB.
    -   [Backend_App_Mode.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_App_Mode.md) (`./Backend_App_Mode.md`): Điều khiển di chuyển thiết bị chạy mô hình Embeddings giữa CPU và GPU khi hoán đổi chế độ hệ thống.
