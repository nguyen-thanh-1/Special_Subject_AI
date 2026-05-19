# Module: Backend Router Market (General Chart Data API)

## 1. Mô tả chi tiết (Detailed Description)
Module này định nghĩa các route API phục vụ dữ liệu biểu đồ chứng khoán / tài sản chung của hệ thống. Được đăng ký là sub-router của FastAPI với tiền tố `/api`, hiện tại module này cung cấp một API duy nhất `/api/market-data` để phục vụ dữ liệu nến (OHLCV) lịch sử phục vụ việc hiển thị biểu đồ trên giao diện Client.

Module này đọc trực tiếp từ tệp tin CSV lưu trữ dữ liệu tài sản của hệ thống (`gold_data.csv` nằm ở thư mục gốc của dự án). Khi có yêu cầu API, hệ thống sẽ:
1.  Tìm kiếm đường dẫn tuyệt đối của file `gold_data.csv` từ vị trí tương đối của file router.
2.  Nếu file không tồn tại, trả về danh sách rỗng `[]` để tránh gây crash cho ứng dụng frontend.
3.  Sử dụng thư viện **Pandas** để đọc dữ liệu CSV. Để tránh các lỗi liên quan đến múi giờ khác nhau giữa các dòng (mixed timezone offsets), dữ liệu cột ngày tháng (`Date`) được chuẩn hóa về định dạng UTC (`utc=True`) và sắp xếp theo trình tự thời gian tăng dần.
4.  Lặp qua các dòng trong dữ liệu và ánh xạ thành cấu trúc JSON chuẩn dạng biểu đồ nến (Open, High, Low, Close) với mốc thời gian dạng Unix Timestamp (giây) phục vụ trực tiếp cho thư viện `lightweight-charts`.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Định tuyến API Biểu đồ**:
    -   `GET /api/market-data`: Đọc tệp dữ liệu CSV gốc, xử lý cấu trúc dữ liệu nến và phản hồi dữ liệu nến dạng danh sách JSON.
-   **Chuẩn hóa dữ liệu**: Sử dụng thư viện Pandas xử lý định dạng ngày tháng, chuyển đổi kiểu dữ liệu dấu phẩy động cho các chỉ số Open, High, Low, Close, và định dạng Unix timestamp.

## 3. Đầu vào (Inputs)
-   HTTP GET request từ Client.
-   Tệp tin dữ liệu cục bộ: `gold_data.csv` tại thư mục gốc của repository.

## 4. Đầu ra (Outputs)
-   Danh sách các đối tượng nến JSON theo định dạng:
    ```json
    [
      {
        "time": 1715961600,
        "open": 2345.5,
        "high": 2360.2,
        "low": 2340.0,
        "close": 2355.8
      }
    ]
    ```
-   Danh sách rỗng `[]` trong trường hợp tệp CSV bị lỗi hoặc không tồn tại.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**: [market_data.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/routers/market_data.py)
-   **Đường dẫn tương đối**: `./backend/src/routers/market_data.py`
-   **File dữ liệu liên quan**: [gold_data.csv](file:///c:/Users/Admin/Desktop/Kafi_chatbot/gold_data.csv)

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   Không phụ thuộc trực tiếp vào các module Python khác ngoại trừ thư viện bên ngoài như `pandas`.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Main.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Main.md) (`./Backend_Main.md`): Đăng ký sub-router này vào ứng dụng FastAPI chính.
    -   [Frontend_App.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Frontend_App.md) (`./Frontend_App.md`): Truy vấn dữ liệu để hiển thị các biểu đồ nến tổng quan.
