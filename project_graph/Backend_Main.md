# Module: Backend Main (FastAPI Application Entrypoint)

## 1. Mô tả chi tiết (Detailed Description)
Module này đóng vai trò là điểm vào chính (entrypoint) của toàn bộ ứng dụng Backend. Được xây dựng trên nền tảng framework **FastAPI**, nó chịu trách nhiệm thiết lập cấu hình cơ bản, kích hoạt CORS (Cross-Origin Resource Sharing) để cho phép frontend kết nối bảo mật, tích hợp các bộ định tuyến API (sub-routers) chuyên biệt, và quản lý vòng đời ứng dụng qua sự kiện khởi chạy (`startup`).

Một điểm nhấn đặc biệt trong kiến trúc của module này là việc khởi chạy một luồng chạy ẩn độc lập (**daemon background thread**) ngay khi server khởi động. Luồng ẩn này thực hiện song song các tác vụ nặng nhằm tối ưu trải nghiệm người dùng:
1. Nạp và làm ấm (`warmup`) các model AI trong Chat Pipeline (Guardrails, Intent Router, Embeddings, Reranker).
2. Tải trước model LLM chính (nếu được cấu hình trong `pipeline.yaml` tại mục `preload_main_llm`). Tác vụ này tốn tài nguyên VRAM nên được phân luồng để tránh nghẽn luồng xử lý HTTP chính.
3. Kích hoạt tính năng tải trước dữ liệu thị trường (pre-fetch quotes và xuất các file lịch sử CSV 5 năm của rổ VN30) lưu vào cache cục bộ, giúp API phản hồi tức thì khi frontend gửi yêu cầu.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
- **Khởi tạo FastAPI**: Định nghĩa ứng dụng FastAPI với tiêu đề `"Kafi Chatbot API"` và phiên bản `0.1.0`.
- **Cấu hình Middleware CORS**: Cấu hình `CORSMiddleware` cho phép kết nối từ mọi nguồn gốc (`allow_origins=["*"]`), hỗ trợ gửi thông tin xác thực, mọi phương thức HTTP và headers (điều chỉnh chặt chẽ hơn khi deploy production).
- **Tích hợp Sub-Routers**: Khai báo và ánh xạ các nhánh API:
    - `/api/chat` (chatbot, quản lý tệp tin, app mode)
    - `/api/market-data` (candlestick chart chung)
    - `/api/vn30` (dữ liệu cổ phiếu rổ VN30)
- **Định nghĩa Health Check Endpoint**: Cung cấp route root (`/`) dạng GET trả về trạng thái hoạt động của server (`{"status": "online", "message": "..."}`).
- **Quản lý Background Startup (Daemon Thread)**: Tạo luồng `Thread(target=_load, daemon=True)` khi server bắt đầu chạy (`@app.on_event("startup")`) để chạy ngầm tiến trình nạp mô hình và nạp cache thị trường, giúp API sẵn sàng phục vụ nhanh nhất có thể mà không chặn việc start của server.

## 3. Đầu vào (Inputs)
- **Cấu hình Hệ thống**: Đọc các thiết lập từ tệp YAML thông qua `AppConfig.get_pipeline_config()`.
- **Tín hiệu Khởi động**: Sự kiện `startup` từ ASGI server (Uvicorn).
- **HTTP Requests**: Các yêu cầu API từ Client/Frontend gửi đến server.

## 4. Đầu ra (Outputs)
- **FastAPI Instance**: Đối tượng ứng dụng sẵn sàng lắng nghe các kết nối trên cổng cấu hình (mặc định port `8000`).
- **Daemon Thread**: Tiến trình chạy ngầm thực thi chuẩn bị tài nguyên hệ thống.
- **HTTP Responses**: Kết quả phản hồi JSON cho các API endpoints.
- **Log Khởi động**: Các dòng log báo hiệu trạng thái tải mô hình ngầm thành công hoặc thất bại.

## 5. File/Thư mục vật lý (Physical Files)
- **Đường dẫn tuyệt đối**: [main.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/main.py)
- **Đường dẫn tương đối**: `./backend/src/main.py`

## 6. Liên kết Đồ thị (Graph Connections)
- **Gọi đến (Calls to / Depends on):**
    - [Backend_Router_Chatbot.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_Chatbot.md) (`./Backend_Router_Chatbot.md`): Tích hợp luồng định tuyến API hội thoại, upload file tri thức.
    - [Backend_Router_Market.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_Market.md) (`./Backend_Router_Market.md`): Tích hợp luồng định tuyến biểu đồ nến tổng quát.
    - [Backend_Router_VN30.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_VN30.md) (`./Backend_Router_VN30.md`): Tích hợp định tuyến VN30 và gọi hàm khởi tạo tải trước cache `pre_fetch_market_data()`.
    - [Backend_Pipeline.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Pipeline.md) (`./Backend_Pipeline.md`): Gọi hàm `warmup()` của Chat Pipeline để nạp trước các model NLP cục bộ.
    - [Backend_LLM.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_LLM.md) (`./Backend_LLM.md`): Gọi `get_llm().ensure_loaded()` để tải trước mô hình ngôn ngữ lớn (nếu cấu hình yêu cầu).
    - [Backend_Config.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Config.md) (`./Backend_Config.md`): Đọc cấu hình khởi chạy hệ thống từ `AppConfig`.
    - [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Sử dụng đối tượng `logger` để ghi nhận nhật ký hệ thống.
- **Được gọi bởi (Called by / Dependency of):**
    - Không có module nào gọi trực tiếp (Đây là entrypoint chạy qua lệnh `uvicorn src.main:app`).
