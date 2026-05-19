# Module: Backend System Monitor (Rich Logger & GPU VRAM Profiler)

## 1. Mô tả chi tiết (Detailed Description)
Module này chịu trách nhiệm kiểm tra, giám sát hiệu năng (**System Observability & Monitoring**) và chẩn đoán hệ thống Backend. Được tích hợp từ hai tệp mã nguồn chuyên biệt `logger.py` và `vram.py`, module cung cấp các công cụ trực quan hóa nhật ký chạy trên terminal và theo dõi sát sao mức tiêu thụ dung lượng bộ nhớ đồ họa GPU (VRAM).

Các cấu trúc cốt lõi trong thiết kế giám sát hệ thống bao gồm:
1.  **Rich Console Logging (Nhật ký màu sắc)**:
    Sử dụng framework **Rich** tích hợp sâu vào thư viện `logging` tiêu chuẩn của Python. Thay vì xuất ra các dòng text đơn điệu, logger tự động hiển thị thời gian, mức độ log (INFO, WARNING, ERROR) kèm theo màu sắc và hiển thị traceback lỗi đẹp mắt (`rich_tracebacks=True`). Để giảm thiểu tối đa các dòng log gây nhiễu từ các thư viện HTTP hoặc AI bên thứ ba (như `httpx`, `huggingface_hub`, `transformers`), module tự động ghi đè và giới hạn mức log của các thư viện này ở ngưỡng `WARNING`.
2.  **Rich Panels & Tables (Giao diện dòng lệnh trực quan)**:
    Định nghĩa một loạt các hàm UI terminal giúp trực quan hóa hành trình xử lý tin nhắn của hệ thống:
    -   `log_user_input`: Đóng khung hộp thoại màu vàng nổi bật hiển thị câu hỏi của người dùng.
    -   `log_agent_response`: Đóng khung hộp thoại xanh dương hiển thị câu trả lời cuối cùng từ trợ lý ảo.
    -   `log_tool_call`: Vẽ một bảng (Table) viền cyan chi tiết hiển thị tên Tool được gọi, tham số truyền vào và kết quả trả về.
    -   `log_delegation`: Báo cáo tiến trình ủy quyền giữa các tác vụ.
    -   `log_llm_metrics`: Vẽ một bảng chỉ số màu hồng (Magenta Grid) báo cáo cực kỳ chi tiết các chỉ số suy luận: model sử dụng, độ trễ TTFT, tổng thời gian chạy, số lượng token đầu ra và tốc độ sinh chữ (Tokens/s).
3.  **Real-Time VRAM Profiling (Đo lường bộ nhớ GPU)**:
    Định nghĩa lớp dữ liệu `VramSnapshot` và các hàm tiện ích gọi trực tiếp vào ngữ cảnh PyTorch GPU (`torch.cuda`). Tiến hành thu thập chính xác các chỉ số:
    -   `allocated_mb`: Bộ nhớ VRAM thực tế đang được chiếm giữ bởi các tensor mô hình.
    -   `reserved_mb`: Bộ nhớ VRAM đang được PyTorch giữ chỗ sẵn từ hệ điều hành.
    -   `max_allocated_mb`: Đỉnh bộ nhớ VRAM lớn nhất đã bị chiếm dụng kể từ khi start.
    -   `delta_vram`: So sánh hai snapshot VRAM trước và sau khi thực hiện hoán swap mô hình, cho phép chẩn đoán ngay lập tức xem hệ thống có bị rò rỉ bộ nhớ đồ họa hay không.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Thiết lập Cấu hình Ghi log**: Tích hợp RichHandler, lọc log rác và định cấu hình ghi nhật ký tập trung cho dự án.
-   **Trực quan hóa Giao dịch**: Thiết lập các hàm vẽ hộp panel, bảng dữ liệu biểu diễn đầu vào của người dùng, công cụ gọi ngoài, và phản hồi AI.
-   **Thu thập Chỉ số VRAM**: Gọi API PyTorch trích xuất chi tiết phân phối dung lượng VRAM đang hoạt động trên GPU CUDA.
-   **Định dạng Hiển thị**: Chuyển đổi các cấu trúc snapshot VRAM sang định dạng chuỗi ngắn gọn phục vụ việc xuất log nhanh chóng.

## 3. Đầu vào (Inputs)
-   Các chuỗi thông tin nhật ký hệ thống.
-   Chỉ số trạng thái phần cứng thu thập thời gian thực từ driver đồ họa CUDA qua PyTorch.

## 4. Đầu ra (Outputs)
-   Các hộp panel, bảng biểu chỉ số hiệu năng hiển thị sống động trên màn hình Terminal của máy chủ.
-   Đối tượng snapshot dữ liệu `VramSnapshot`.
-   Chuỗi định dạng so sánh chênh lệch dung lượng bộ nhớ.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**:
    -   Trình ghi nhật ký: [logger.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/logger.py)
    -   Trình giám sát bộ nhớ đồ họa: [vram.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/vram.py)
-   **Đường dẫn tương đối**: `./backend/src/utils/logger.py` và `./backend/src/utils/vram.py`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   Không phụ thuộc vào các module logic khác của ứng dụng, chỉ phụ thuộc thư viện đồ họa dòng lệnh `rich` và thư viện học sâu `torch`.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Main.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Main.md) (`./Backend_Main.md`): Ghi nhận toàn bộ tiến trình khởi chạy ngầm và tải tài nguyên.
    -   [Backend_LLM.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_LLM.md) (`./Backend_LLM.md`): Gọi `get_vram_snapshot()` đo đạc VRAM khi hoán swap mô hình và xuất bảng LLM Metrics chi tiết sau mỗi câu trả lời.
    -   [Backend_Pipeline.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Pipeline.md) (`./Backend_Pipeline.md`): Log thông tin định tuyến, log độ tương đồng cosine của cache hit, log kết quả kiểm duyệt guardrails.
    -   [Backend_Knowledge_Base.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Knowledge_Base.md) (`./Backend_Knowledge_Base.md`): Ghi nhận tiến độ nạp PDF, thời gian sinh vector và lỗi trích xuất tài liệu.
    -   [Backend_Router_Chatbot.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_Chatbot.md) (`./Backend_Router_Chatbot.md`): Log thông báo lỗi và cảnh báo an toàn của luồng API.
    -   [Backend_App_Mode.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_App_Mode.md) (`./Backend_App_Mode.md`): Log vết chuyển đổi chế độ hệ thống CHAT/INDEXING.
