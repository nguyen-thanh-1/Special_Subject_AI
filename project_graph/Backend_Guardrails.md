# Module: Backend Guardrails (Input/Output Safety Guardrails)

## 1. Mô tả chi tiết (Detailed Description)
Module này đóng vai trò là "khiên bảo vệ" (Security Layer) của hệ thống, chịu trách nhiệm kiểm duyệt tính an toàn ở cả 2 đầu: chặn đứng các yêu cầu độc hại từ người dùng trước khi chúng đi sâu vào hệ thống (Input Guardrails) và lọc các phản hồi nhạy cảm hoặc lỗi thông tin của AI trước khi trả về cho client (Output Guardrails). Được định nghĩa trong lớp `GuardrailsManager`, module này kết hợp sức mạnh của **Heuristic Rules (Quy tắc heuristic dựa trên Regex)** và **Local Machine Learning Classifiers (Mô hình phân loại an toàn cục bộ)**.

Kiến trúc kiểm duyệt 2 lớp (Dual-Stage Defense) hoạt động vô cùng hiệu quả:
1.  **Lớp Heuristic Regex Blocklist (Kiểm tra siêu tốc)**:
    Khi nhận được văn bản, hệ thống tiến hành chuẩn hóa (chuyển chữ thường, xóa khoảng trắng thừa) và quét qua danh sách các mẫu biểu thức chính quy (Regex Patterns) được định nghĩa trong `guardrails.yaml`. Các mẫu này được thiết kế để phát hiện:
    - Các nỗ lực tấn công bẻ khóa prompt hệ thống (**Prompt Injection Attacks** như *"hãy bỏ qua các chỉ thị trước đó và đóng vai làm..."*).
    - Các từ ngữ thô tục, bạo lực, xúc phạm bằng tiếng Việt và tiếng Anh.
    - Các câu hỏi rò rỉ mã nguồn hoặc thông tin bảo mật nội bộ.
    Nếu phát hiện bất kỳ mẫu nào khớp, hệ thống lập tức trả về nhãn `UNSAFE` và log chính xác mẫu vi phạm. Quá trình này diễn ra chỉ trong vài phần triệu giây (microseconds), giúp bảo vệ GPU khỏi các truy vấn spam độc hại mà không tiêu tốn tài nguyên suy luận.
2.  **Lớp Machine Learning Classification (Kiểm tra ngữ nghĩa sâu)**:
    Nếu văn bản vượt qua lớp Regex, hệ thống sẽ đưa vào mô hình phân loại cục bộ chạy trên GPU (nếu được cấu hình và kích hoạt). Mô hình này phân tích ngữ nghĩa tinh vi để phát hiện các nội dung độc hại ẩn giấu không dùng từ khóa thô tục trực tiếp. Kết quả suy luận được so sánh với ngưỡng an toàn quy định trong cấu hình; nếu vượt ngưỡng vi phạm sẽ trả về `UNSAFE`.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Quản lý bộ quy tắc an toàn**: Đọc danh sách các mẫu regex chặn từ tệp tin cấu hình `guardrails.yaml`.
-   **Nạp & Giải phóng Mô hình**: Quản lý nạp lười mô hình AI phân loại vào GPU (`ensure_loaded`) và giải phóng bộ nhớ đồ họa (`unload`) khi chuyển đổi chế độ hệ thống.
-   **Kiểm duyệt hai chiều**:
    -   Input Check: Ngăn chặn mã độc, bẻ khóa prompt và từ ngữ thô tục từ câu hỏi người dùng.
    -   Output Check: Kiểm tra văn bản AI sinh ra định kỳ (mỗi 300 ký tự khi stream và kiểm tra toàn bộ khi kết thúc) để chặn rò rỉ thông tin nhạy cảm hoặc câu trả lời không phù hợp.
-   **Ghi nhận vết vi phạm**: Nhật ký chi tiết lỗi vi phạm để đội ngũ quản trị phân tích và tối ưu hóa bộ lọc.

## 3. Đầu vào (Inputs)
-   `text` (chuỗi ký tự): Văn bản cần kiểm tra tính an toàn (câu hỏi của user hoặc câu trả lời của AI).
-   Cấu hình quy tắc chặn từ `guardrails.yaml`.

## 4. Đầu ra (Outputs)
-   Nhãn phân loại: `"SAFE"` (An toàn) hoặc `"UNSAFE"` (Không an toàn).

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [guardrails.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/guardrails.py)
-   **Đường dẫn tương đối**: `./backend/src/utils/guardrails.py`
-   **File cấu hình liên quan**: [guardrails.yaml](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/config/guardrails.yaml)

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Config.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Config.md) (`./Backend_Config.md`): Đọc danh sách các mẫu regex và ngưỡng an toàn thông qua `AppConfig.get_guardrails_config()`.
    -   [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Log thông tin chi tiết về các trường hợp bị chặn (mẫu regex nào bị vi phạm, câu hỏi nào bị đánh dấu độc hại) và các lỗi tải mô hình.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Main.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Main.md) (`./Backend_Main.md`): Làm ấm (warmup) nạp trước mô hình kiểm duyệt khi server khởi chạy.
    -   [Backend_Pipeline.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Pipeline.md) (`./Backend_Pipeline.md`): Thực hiện kiểm tra câu hỏi người dùng ở đầu chu kỳ xử lý, kiểm tra định kỳ 300 ký tự khi stream, và kiểm duyệt toàn bộ phản hồi trước khi ghi vào Semantic Cache.
    -   [Backend_App_Mode.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_App_Mode.md) (`./Backend_App_Mode.md`): Giải phóng bộ nhớ của mô hình phân loại khỏi VRAM khi chuyển sang chế độ indexing tài liệu.
