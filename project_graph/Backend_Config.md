# Module: Backend Config (System Configurations & Env Loader)

## 1. Mô tả chi tiết (Detailed Description)
Module này đóng vai trò là "lớp cấu hình" (Configuration Management Layer) tập trung của toàn bộ dự án Backend. Được định nghĩa trong lớp `AppConfig`, module chịu trách nhiệm tải, phân tích cú pháp (parsing) và phân phối các thông số thiết lập từ các tệp cấu hình **YAML** và biến môi trường (**Environment Variables**) trong tệp `.env`.

Kiến trúc nạp cấu hình của `AppConfig` được thiết kế đơn giản nhưng cực kỳ tin cậy:
1.  **Xác định thư mục gốc tuyệt đối (Base Directory Resolution)**:
    Khi khởi chạy, module tự động phân tích vị trí vật lý của chính nó trên đĩa cứng và sử dụng `Path(__file__).resolve().parent.parent.parent` để tìm ra thư mục gốc tuyệt đối của dự án backend (thư mục chứa tệp `.env` và thư mục con `config/`). Điều này đảm bảo hệ thống luôn tìm đúng tệp cấu hình bất kể server được khởi chạy từ thư mục làm việc (cwd) nào.
2.  **Đọc biến môi trường an toàn**: Sử dụng thư viện `python-dotenv` nạp tệp `.env` vào biến môi trường hệ thống (`os.environ`). Các khóa bảo mật như `LITELLM_BASE_URL`, `LITELLM_API_KEY`, và `FINLENS_API_KEY` được trích xuất an toàn và lưu trữ dưới dạng các thuộc tính lớp tĩnh (static class properties).
3.  **Phân tích cú pháp YAML bảo mật**:
    Phương thức `load_yaml` kết hợp đường dẫn thư mục gốc với thư mục con `config/` để tìm tệp cấu hình đích. Tệp tin được mở với bảng mã ký tự chuẩn quốc tế **UTF-8** (đảm bảo không bị lỗi font khi cấu hình các đoạn prompt chứa tiếng Việt có dấu) và được phân tích cú pháp thông qua trình giải mã an toàn `yaml.safe_load()`. Nếu tệp cấu hình bị thiếu, hệ thống tự động trả về một từ điển rỗng `{}` thay vì gây crash ứng dụng.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Định vị thư mục gốc**: Xác định đường dẫn tuyệt đối của thư mục backend.
-   **Tải tệp `.env`**: Kích hoạt `load_dotenv()` khi nạp module.
-   **Tải cấu hình YAML chuyên biệt**:
    -   `get_llm_config()`: Tải tệp `llms.yaml` chứa danh sách ID, tên và cài đặt của các mô hình LLM.
    -   `get_agents_config()`: Tải tệp `agents.yaml` chứa prompt hệ thống (system instructions) cho Agent.
    -   `get_pipeline_config()`: Tải tệp `pipeline.yaml` chứa cấu hình các mô hình cục bộ (Embeddings, Reranker) và ngưỡng tìm kiếm RAG.
    -   `get_router_config()`: Tải tệp `router.yaml` chứa cài đặt định tuyến câu hỏi.
    -   `get_guardrails_config()`: Tải tệp `guardrails.yaml` chứa các mẫu regex và quy tắc lọc an toàn.
-   **Phân phối biến tĩnh**: Trích xuất các biến môi trường cấu hình kết nối API của LiteLLM.

## 3. Đầu vào (Inputs)
-   Tệp cấu hình hệ thống: Các file `.yaml` trong thư mục [config](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/config/).
-   Tệp biến môi trường cục bộ: [backend/.env](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/.env).

## 4. Đầu ra (Outputs)
-   Từ điển Python (Dictionaries) chứa toàn bộ cặp Key-Value thông số cấu hình của từng module tương ứng.
-   Các chuỗi ký tự chứa thông tin kết nối API Key và Endpoint URL.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [app_config.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/app_config.py)
-   **Đường dẫn tương đối**: `./backend/src/utils/app_config.py`
-   **Thư mục cấu hình**: `./backend/config/`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   Không phụ thuộc trực tiếp vào các module khác của hệ thống ngoại trừ các thư viện Python bên ngoài (`pyyaml`, `python-dotenv`).
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Main.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Main.md) (`./Backend_Main.md`): Đọc cấu hình pipeline để quyết định có nạp trước LLM lúc khởi chạy hay không.
    -   [Backend_LLM.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_LLM.md) (`./Backend_LLM.md`): Tải thông tin models cục bộ và models đám mây.
    -   [Backend_Pipeline.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Pipeline.md) (`./Backend_Pipeline.md`): Tải thông tin thiết lập cho toàn bộ Chat Pipeline.
    -   [Backend_Knowledge_Base.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Knowledge_Base.md) (`./Backend_Knowledge_Base.md`): Lấy kích thước chunks và ngưỡng truy xuất Qdrant.
    -   [Backend_Query_Router.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Query_Router.md) (`./Backend_Query_Router.md`): Lấy cài đặt định tuyến intent.
    -   [Backend_Guardrails.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Guardrails.md) (`./Backend_Guardrails.md`): Lấy danh sách quy tắc lọc an toàn regex và mô hình an toàn.
