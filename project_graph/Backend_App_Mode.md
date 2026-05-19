# Module: Backend App Mode (VRAM Optimizer & Device Switcher)

## 1. Mô tả chi tiết (Detailed Description)
Module này chịu trách nhiệm quản lý chế độ hoạt động (**Application State Management**) và tối ưu hóa tài nguyên phần cứng (VRAM Optimizer). Được định nghĩa trong lớp `AppModeManager`, module này thực thi một cơ chế điều phối tài nguyên động độc đáo, cho phép chatbot hoạt động mượt mà trên các máy chủ có dung lượng bộ nhớ GPU (VRAM) hạn chế.

Hệ thống có hai chế độ hoạt động chính: **`CHAT` (Hội thoại)** và **`INDEXING` (Nạp tri thức)**. Do cả hai chế độ đều sử dụng các mô hình học sâu (Deep Learning) rất nặng, việc chạy đồng thời tất cả các mô hình trên GPU sẽ gây ra lỗi tràn bộ nhớ đồ họa (Out of Memory). `AppModeManager` giải quyết bài toán này bằng cách hoán đổi tài nguyên động:

```mermaid
graph TD
    subgraph CHAT MODE (Hội thoại)
        CPU1[Embeddings Model]
        GPU1[Query Router Model]
        GPU2[Core LLM Model]
    end
    subgraph INDEXING MODE (Nạp tài liệu)
        CPU2[Core LLM: UNLOADED]
        CPU3[Query Router: UNLOADED]
        GPU3[Embeddings Model: CUDA]
    end
    
    Trigger[Yêu cầu đổi Mode từ UI] --> Switch{set_mode}
    Switch -- CHAT --> CHAT
    Switch -- INDEXING --> INDEXING
```

Chi tiết quy trình chuyển đổi chế độ của `AppModeManager` được thực hiện đồng bộ như sau:
1.  **Chuyển sang `INDEXING` (Tải tài liệu PDF)**:
    -   Hệ thống gọi lệnh giải phóng mô hình **Query Router** và mô hình **Core LLM** trọn vẹn ra khỏi bộ nhớ GPU (VRAM) bằng cách gọi hàm `unload()`.
    -   Kích hoạt di chuyển mô hình **Embeddings** lên GPU (`get_embedder().set_device('cuda')`). Việc này giúp toàn bộ tài nguyên GPU được giải phóng để phục vụ duy nhất tác vụ nhúng vector các mảnh PDF với tốc độ tối đa.
2.  **Chuyển sang `CHAT` (Trò chuyện bình thường)**:
    -   Di chuyển mô hình **Embeddings** từ GPU về chạy trên bộ nhớ RAM hệ thống (**CPU**) bằng lệnh `get_embedder().set_device('cpu')`.
    -   Kích hoạt nạp lại mô hình **Query Router** lên GPU (`get_router().ensure_loaded()`). Mô hình ngôn ngữ lớn LLM chính sẽ được nạp lười ngay khi có câu hỏi đầu tiên gửi đến. Việc chuyển Embeddings sang CPU giúp GPU dư dả bộ nhớ để chạy suy luận LLM tốc độ cao mà không lo sợ tràn RAM.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Định nghĩa trạng thái**: Thiết lập Enum `Mode` gồm hai trạng thái `"CHAT"` và `"INDEXING"`.
-   **Chuyển đổi Chế độ Đồng bộ**: Chuyển đổi trạng thái hệ thống an toàn đa luồng thông qua sử dụng `Lock`.
-   **Quản lý Phụ thuộc Vòng (Circular Dependencies)**: Sử dụng kỹ thuật nạp lười các module (`from src.utils.router import get_router`, v.v.) bên trong hàm thực thi thay vì khai báo ở đầu file, tránh lỗi import vòng chéo trong Python.
-   **Điều phối tài nguyên VRAM**: Phối hợp trực tiếp các lệnh `unload()` và `set_device('cuda'/'cpu')` của các mô hình tương ứng để điều phối việc sử dụng phần cứng.

## 3. Đầu vào (Inputs)
-   `new_mode` (Enum `Mode`): Chế độ hoạt động đích (`Mode.CHAT` hoặc `Mode.INDEXING`).

## 4. Đầu ra (Outputs)
-   Giá trị Boolean (`True` nếu hoán đổi thành công, `False` nếu gặp ngoại lệ hệ thống).
-   Trạng thái thiết bị chạy mô hình được thay đổi trên thực tế.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [app_mode.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/app_mode.py)
-   **Đường dẫn tương đối**: `./backend/src/utils/app_mode.py`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Query_Router.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Query_Router.md) (`./Backend_Query_Router.md`): Gọi `get_router().unload()` để giải phóng hoặc `ensure_loaded()` để nạp lại mô hình phân loại intent.
    -   [Backend_Embeddings.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Embeddings.md) (`./Backend_Embeddings.md`): Gọi `get_embedder().set_device()` để chuyển dịch thiết bị chạy mô hình embedding sang CPU hoặc CUDA.
    -   [Backend_LLM.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_LLM.md) (`./Backend_LLM.md`): Gọi `get_llm().unload()` để thu hồi bộ nhớ đồ họa của mô hình ngôn ngữ lớn LLM khi ở chế độ indexing.
    -   [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Log thông báo tiến trình chuyển đổi chế độ và các lỗi liên quan.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Router_Chatbot.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_Chatbot.md) (`./Backend_Router_Chatbot.md`): Cung cấp các API kiểm tra chế độ hiện hành, chặn các yêu cầu tải file PDF nếu không ở chế độ Indexing, chặn yêu cầu chat nếu đang ở chế độ Indexing, và tiếp nhận yêu cầu hoán đổi chế độ từ người dùng.
