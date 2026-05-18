# Module: Backend Router VN30 (VN30 Market Data API)

## 1. Mô tả chi tiết (Detailed Description)
Module này chịu trách nhiệm phục vụ dữ liệu thị trường chuyên sâu cho rổ chỉ số **VN30** (30 cổ phiếu có vốn hóa và thanh khoản lớn nhất thị trường chứng khoán Việt Nam). Được xây dựng dưới dạng sub-router của FastAPI với tiền tố `/api/vn30`, module này giao tiếp trực tiếp với nhà cung cấp dữ liệu tài chính doanh nghiệp thông qua bộ SDK của **FinLens**.

Để đảm bảo hiệu năng tối ưu và khả năng hoạt động ổn định bất kể trạng thái kết nối mạng hay giới hạn tần suất (rate limits) của API nhà cung cấp, module này triển khai kiến trúc **Robust Hybrid Cache (Bộ đệm kép)**:
1.  **Quotes Caching**: Dữ liệu giá giao dịch hiện tại của 30 cổ phiếu được lưu trữ chung trong tệp `vn30_quotes.json`.
2.  **OHLCV Caching**: Mỗi mã cổ phiếu và mỗi khung thời gian (`timeframe`) được lưu vào một tệp JSON riêng biệt (Ví dụ: `ohlcv_FPT_1y.json`), lưu trữ toàn bộ chuỗi nến lịch sử.
3.  **Graceful Fallback**: Khi client gửi yêu cầu, hệ thống luôn ưu tiên gọi SDK FinLens thời gian thực. Trong trường hợp API gặp sự cố (mất mạng, hết hạn key API, vượt hạn mức request), hệ thống tự động bắt lỗi và tải dữ liệu từ tệp cache cục bộ lên phản hồi cho client mà không gây ra bất cứ gián đoạn nào (Zero downtime).
4.  **Tải trước khi khởi động (Warmup Pre-fetching)**: Module tích hợp một hàm nạp trước dữ liệu `pre_fetch_market_data()`. Hàm này được gọi trong một luồng phụ ngầm ngay khi khởi động backend, thực hiện gọi API điền đầy bộ đệm Quotes và kích hoạt trình xuất tệp lịch sử CSV 5 năm của toàn bộ rổ VN30.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Quản lý Danh mục rổ VN30**: Khai báo danh sách tĩnh gồm 30 cấu phần ticker chính thức (ACB, FPT, HPG, VCB, VIC, v.v.).
-   **Định tuyến API VN30**:
    -   `GET /api/vn30/tickers`: Trả về danh sách 30 cấu phần VN30 và tổng số lượng để frontend hiển thị.
    -   `GET /api/vn30/quotes`: Báo giá phiên giao dịch gần nhất cho 30 cổ phiếu (bao gồm: giá mở, cao, thấp, đóng, khối lượng, chênh lệch giá, phần trăm thay đổi). Lưu cache hoặc đọc fallback từ cache `data/market_cache/vn30_quotes.json`.
    -   `GET /api/vn30/ohlcv/{symbol}`: Trả về chuỗi nến biểu đồ cho một mã cụ thể dựa theo tham số `timeframe` (`1d`, `5d`, `1m`, `3m`, `1y`, `5y`).
-   **Chuyển đổi Độ phân giải Biểu đồ**:
    -   Với khung thời gian ngắn (`1d`, `5d`): Gọi API lấy nến intraday chi tiết có độ phân giải **15 phút** (`15m`).
    -   Với khung thời gian dài (`1m`, `3m`, `1y`, `5y`): Gọi API lấy nến đóng cửa cuối ngày (**EOD - 1d**).
-   **Điều phối tải trước (Background Pre-fetcher)**:
    -   `pre_fetch_market_data()`: Kích hoạt tải bảng giá hiện tại vào bộ đệm và gọi tiện ích xuất lịch sử CSV của toàn bộ 30 mã để chuẩn bị sẵn sàng dữ liệu phân tích cục bộ.

## 3. Đầu vào (Inputs)
-   HTTP Requests kèm tham số truy vấn (`timeframe`) và tham số đường dẫn (`symbol`).
-   Biến môi trường: `FINLENS_API_KEY` dùng để cấu hình SDK.
-   Tệp tin cache cục bộ trong thư mục `data/market_cache/`.

## 4. Đầu ra (Outputs)
-   Bảng JSON chứa danh sách báo giá hiện thời VN30.
-   Mảng JSON danh sách các nến OHLCV vẽ biểu đồ TradingView.
-   Các tệp tin cache lưu trên đĩa cứng (`ohlcv_{symbol}_{tf}.json`).
-   Mã lỗi HTTP 404 nếu mã CP không thuộc VN30, hoặc 503 nếu lỗi API và không có tệp đệm hỗ trợ.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [vn30.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/routers/vn30.py)
-   **Đường dẫn tương đối**: `./backend/src/routers/vn30.py`
-   **Thư mục lưu trữ bộ đệm**: `data/market_cache/`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Market_Exporter.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Market_Exporter.md) (`./Backend_Market_Exporter.md`): Gọi hàm `export_vn30_historical_csv()` để xuất file CSV dữ liệu lịch sử 5 năm khi khởi chạy ngầm.
    -   [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Sử dụng logger để ghi nhận các ngoại lệ và tiến trình tải trước dữ liệu.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Main.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Main.md) (`./Backend_Main.md`): Đăng ký sub-router vào hệ thống FastAPI và chạy ngầm hàm `pre_fetch_market_data()` khi server bắt đầu khởi chạy.
    -   [Frontend_App.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Frontend_App.md) (`./Frontend_App.md`): Polling giá thị trường liên tục và yêu cầu nến dữ liệu để vẽ biểu đồ kỹ thuật.
