# Module: Backend Reranker (Cross-Encoder Retrieval Reranker)

## 1. Mô tả chi tiết (Detailed Description)
Module này chịu trách nhiệm tối ưu hóa kết quả tìm kiếm tri thức RAG thông qua mô hình tái xếp hạng chéo (**Cross-Encoder Reranker**). Được định nghĩa trong lớp `RerankerManager`, module này bao bọc thư viện Hugging Face SentenceTransformers chuyên biệt để đánh giá lại độ tương quan thực tế giữa câu hỏi truy vấn của người dùng và các mảnh văn bản (chunks) được tìm kiếm sơ bộ từ Qdrant DB.

Kiến trúc RAG thông thường sử dụng mô hình Bi-Encoder (Embeddings) để mã hóa câu hỏi và tài liệu độc lập và so khớp nhanh bằng khoảng cách Cosine. Cơ chế này nhanh nhưng thiếu đi khả năng so sánh tương quan chéo sâu sắc giữa các từ. Reranker giải quyết triệt để vấn đề này bằng giải pháp **Cross-Attention Sequence Scoring (Chấm điểm chuỗi tương quan chéo)**:
1.  **Ghép đôi câu hỏi - tài liệu**: Module ghép câu hỏi và nội dung từng mảnh văn bản thành một cặp đầu vào dạng `[câu_hỏi, mảnh_văn_bản_i]`.
2.  **Đánh giá chéo đồng thời**: Mô hình Cross-Encoder đọc cả câu hỏi và tài liệu cùng một lúc, cho phép các cơ chế Attention phân tích sự tương tác ngữ nghĩa của từng từ trong câu hỏi đối với từng từ trong tài liệu.
3.  **Tái sắp xếp chính xác**: Mô hình xuất ra điểm số (score) thực tế thể hiện độ tương đồng ngữ nghĩa. Module sắp xếp lại danh sách các mảnh tài liệu theo thứ tự điểm số giảm dần và lọc ra **Top-3** mảnh chất lượng nhất để gửi tới LLM. Điều này giúp loại bỏ hoàn toàn các mảnh tài liệu gây nhiễu và cải thiện đáng kể độ chính xác của câu trả lời AI sinh ra.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Quản lý mô hình Cross-Encoder**: Nạp lười (`ensure_loaded`) mô hình tái xếp hạng chéo cục bộ hoặc từ Hugging Face Hub (Ví dụ: `cross-encoder/ms-marco-MiniLM-L-6-v2`) theo cấu hình trong `pipeline.yaml`.
-   **Chuẩn bị định dạng đầu vào**: Lặp qua các tài liệu thô trích xuất từ Qdrant, chuyển đổi và ghép đôi thành cấu trúc chuỗi đầu vào tương thích với mô hình.
-   **Chấm điểm & Sắp xếp (`rerank`)**:
    - Gọi mô hình Cross-Encoder chạy suy luận đánh giá điểm số tương quan cho tất cả các cặp.
    - Ghép nối tài liệu gốc với điểm số nhận được, thực hiện sắp xếp giảm dần theo điểm số.
    - Cắt danh sách và chỉ trả về tối đa `top_k` (mặc định Top-3) tài liệu tốt nhất.
-   **Giải phóng VRAM**: Thu hồi 100% dung lượng bộ nhớ GPU chiếm dụng bởi mô hình Cross-Encoder khi hệ thống chuyển đổi trạng thái hoạt động ngầm.

## 3. Đầu vào (Inputs)
-   `query` (chuỗi ký tự): Câu hỏi của người dùng đã viết lại.
-   `candidates` (Danh sách các đối tượng văn bản từ Qdrant): Các mảnh tri thức ứng viên cần đánh giá lại.
-   `top_k` (số nguyên): Số lượng tài liệu tốt nhất muốn lọc giữ lại.
-   Cấu hình mô hình reranker từ `pipeline.yaml` tại nhánh `reranker`.

## 4. Đầu ra (Outputs)
-   Danh sách chứa tối đa `top_k` tài liệu đã được sắp xếp lại thứ tự tối ưu ngữ nghĩa, sẵn sàng cung cấp ngữ cảnh sạch cho LLM.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [reranker.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/reranker.py)
-   **Đường dẫn tương đối**: `./backend/src/utils/reranker.py`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Config.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Config.md) (`./Backend_Config.md`): Tải cấu hình mô hình reranker qua `AppConfig.get_pipeline_config()`.
    -   [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Ghi log thông số điểm chấm chéo Reranker Scores và log lỗi nạp mô hình.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Main.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Main.md) (`./Backend_Main.md`): Làm ấm nạp trước mô hình Reranker cục bộ khi khởi chạy ứng dụng FastAPI.
    -   [Backend_Knowledge_Base.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Knowledge_Base.md) (`./Backend_Knowledge_Base.md`): Tích hợp Reranker vào cuối phương thức `kb.retrieve()` nhằm tinh chọn tri thức trước khi trả về.
    -   [Backend_App_Mode.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_App_Mode.md) (`./Backend_App_Mode.md`): Thu hồi bộ nhớ CrossEncoder khỏi VRAM khi chuyển đổi sang chế độ Indexing nạp dữ liệu.
