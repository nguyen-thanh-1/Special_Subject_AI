# Module: Backend Market Exporter (Historical CSV Data Exporter)

## 1. Mô tả chi tiết (Detailed Description)
Module này là một công cụ tiện ích dữ liệu hệ thống (**System Data Utility**), chịu trách nhiệm xuất bản và lưu trữ dữ liệu nến chứng khoán lịch sử EOD (End of Day) của toàn bộ rổ VN30. Được định nghĩa trong tệp `market_exporter.py` với hàm nghiệp vụ chính là `export_vn30_historical_csv()`.

Khi hệ thống khởi chạy hoặc có yêu cầu kích hoạt, module này sẽ thực hiện một quy trình tự động hóa khép kín:
1.  **Khởi tạo Thư mục lưu trữ**: Tạo thư mục đích lưu trữ dữ liệu `data/vn30_historical_csv/` tại thư mục làm việc cục bộ.
2.  **Giao tiếp SDK FinLens**: Khởi tạo kết nối FinLens client thời gian thực.
3.  **Lập khoảng thời gian lịch sử**: Tính toán mốc thời gian bắt đầu và kết thúc của khoảng thời gian **5 năm** trở về trước dựa trên ngày hiện tại (`_date_range_for_timeframe("5y")`).
4.  **Tải và Xuất dữ liệu hàng loạt (Batch Export)**:
    Lặp qua danh sách 30 mã cổ phiếu VN30. Với mỗi mã cổ phiếu:
    - Gọi API FinLens EOD tải chuỗi dữ liệu nến 5 năm.
    - Chuẩn hóa tiêu đề các cột dữ liệu sang dạng viết hoa chữ cái đầu (`Capitalize` các cột thành: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`) để đảm bảo tính đồng nhất định dạng cho các module phân tích dữ liệu chứng khoán khác.
    - Xử lý sắp xếp các dòng tăng dần theo mốc ngày tháng (`Date`).
    - Lưu trữ trực tiếp dữ liệu thành tệp CSV độc lập tương ứng với tên mã (Ví dụ: `data/vn30_historical_csv/FPT_5y_daily.csv`).
    - Bắt lỗi riêng lẻ cho từng mã chứng khoán để đảm bảo nếu một mã gặp sự cố kết nối, tiến trình xuất dữ liệu của các mã còn lại vẫn tiếp tục diễn ra bình thường.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Đọc dữ liệu lịch sử EOD**: Giao tiếp SDK FinLens tải nến đóng cửa hàng ngày 5 năm ròng của rổ VN30.
-   **Chuẩn hóa dữ liệu thô**: Chuyển đổi chỉ mục DataFrame Pandas (`index`) thành cột `Date`, chuẩn hóa kiểu dữ liệu ngày tháng và tiêu đề các cột chỉ số kỹ thuật.
-   **Lưu trữ đĩa cứng**: Xuất DataFrame ra định dạng CSV mã hóa UTF-8 lưu trữ tại thư mục chỉ định phục vụ phân tích tài chính ngoại tuyến hoặc kiểm toán dữ liệu.

## 3. Đầu vào (Inputs)
-   Tham số cấu hình thời gian 5 năm.
-   Dữ liệu phản hồi từ FinLens EOD API.
-   Danh sách 30 mã VN30 constituents.

## 4. Đầu ra (Outputs)
-   30 tệp tin CSV chứa dữ liệu nến lịch sử 5 năm phân tách riêng theo mã tại thư mục `data/vn30_historical_csv/`.
-   Nhật ký ghi nhận tiến trình xuất file thành công hoặc cảnh báo thất bại.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [market_exporter.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/market_exporter.py)
-   **Đường dẫn tương đối**: `./backend/src/utils/market_exporter.py`
-   **Thư mục chứa tệp xuất**: `backend/data/vn30_historical_csv/`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Router_VN30.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_VN30.md) (`./Backend_Router_VN30.md`): Sử dụng danh sách tĩnh các mã `VN30_TICKERS`, sử dụng hàm lấy client FinLens `_get_client()`, và sử dụng hàm tính toán thời gian `_date_range_for_timeframe()`.
    -   [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Ghi log nhật ký tiến trình lưu trữ và xuất lỗi.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Router_VN30.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_VN30.md) (`./Backend_Router_VN30.md`): Khởi chạy ngầm hàm `export_vn30_historical_csv()` khi khởi động server lúc pre_fetch dữ liệu rổ VN30.
    -   [Backend_Stock_Analysis.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Stock_Analysis.md) (`./Backend_Stock_Analysis.md`): Sử dụng các file CSV OHLCV đã xuất làm nguồn dữ liệu đầu vào cho 35 hàm phân tích cổ phiếu.
