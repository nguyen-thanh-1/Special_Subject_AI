# Kế hoạch Triển khai: Tích hợp Tính năng Phân tích Cổ phiếu VN30 (Stock Analysis Integration Plan)

Tài liệu này đề xuất giải pháp kỹ thuật, kiến trúc và kế hoạch triển khai tính năng **Phân tích Cổ phiếu VN30** theo yêu cầu từ khách hàng Kafi.

---

## 1. Phân tích các Phương án Kiến trúc (Architectural Options Evaluation)

Để xử lý dữ liệu chuỗi thời gian 5 năm (khoảng 1.250 dòng nến mỗi mã cổ phiếu) và đưa ra tư vấn đầu tư chuẩn xác, chúng ta phân tích 3 hướng tiếp cận:

### ❌ Phương án A: Nạp trực tiếp toàn bộ dữ liệu CSV thô vào LLM Prompt
*   **Cách hoạt động**: Đọc tệp CSV 5 năm của mã cổ phiếu, định dạng thành chuỗi văn bản và chèn thẳng vào prompt để LLM tự phân tích.
*   **Nhược điểm**:
    *   **Tràn cửa sổ ngữ cảnh & Chi phí cao**: 1.250 dòng dữ liệu tiêu tốn ~35.000 tokens mỗi câu hỏi. Thời gian xử lý (TTFT) cực kỳ lâu và tốn tài nguyên.
    *   **Lỗi tính toán (Hallucination)**: LLM rất yếu trong việc tính toán số học phức tạp. Nó không thể tự tính toán chính xác RSI, đường SMA chéo hay độ lệch biên độ để đưa ra giá mục tiêu từ hàng nghìn con số thô.

### ❌ Phương án B: Huấn luyện mô hình Machine Learning riêng biệt (LSTM / XGBoost)
*   **Cách hoạt động**: Thiết lập một luồng huấn luyện ngoại tuyến (offline training) để dự đoán giá ngày mai, sau đó truyền chỉ số dự báo vào LLM.
*   **Nhược điểm**:
    *   **Quá phức tạp**: Cần cài đặt nhiều thư viện nặng, tinh chỉnh siêu tham số và tốn thời gian tính toán.
    *   **Độ tin cậy thấp**: Dự đoán giá chứng khoán bằng ML cơ bản rất dễ sai lệch và rủi ro tuân thủ (compliance) trong tài chính là rất cao.

###  Phương án C (Đề xuất): Chiến lược Hybrid Quantitative-LLM (Thuật toán nén dữ liệu + AI cố vấn)
*   **Cách hoạt động**:
    1.  **Quantitative Engine (Động cơ Định lượng)**: Viết một mô-đun Python chuyên dụng `stock_analyzer.py` sử dụng thư viện `pandas` và `numpy` (đã có sẵn trong hệ thống) để đọc CSV. Tính toán các chỉ báo kỹ thuật cốt lõi một cách **tuyệt đối chính xác bằng toán học**:
        *   **Xu hướng (Trend)**: Đường trung bình động đơn giản `SMA20`, `SMA50`, `SMA200`.
        *   **Động lượng (Momentum)**: Chỉ số sức mạnh tương đối `RSI (14)`.
        *   **Biến động & Quản trị rủi ro**: Chỉ báo biến động thực tế trung bình `ATR (14)` (để đặt Stop-loss tối ưu).
        *   **Vùng giá**: Xác định Vùng mua an toàn (hỗ trợ kỹ thuật), Giá mục tiêu (Target Price) và Cắt lỗ (Stop-loss).
    2.  **Nén thông tin**: Chuyển đổi 5 năm dữ liệu thô thành một chuỗi JSON tóm tắt chỉ số định lượng siêu nhẹ (~15 dòng).
    3.  **LLM Synthesizer (AI cố vấn)**: Nạp JSON chỉ số định lượng này vào `FinancialAgent`. AI sẽ đóng vai trò chuyên gia quản lý tài sản chuyên nghiệp để tổng hợp dữ liệu thành báo cáo đầu ra ngắn gọn, dễ hiểu đúng cấu trúc khách hàng yêu cầu.
*   **Ưu điểm**:
    *   **Chính xác 100% về mặt toán học**: Triệt tiêu hoàn toàn ảo giác (hallucination) của AI do việc tính toán chỉ báo được thực hiện bằng công thức Pandas chuẩn mực.
    *   **Siêu nhẹ & Tiết kiệm Token**: Tiết kiệm 99% token, giữ cho chatbot phản hồi nhanh như chớp (dưới 2 giây).
    *   **Đầu ra chuẩn xác**: Đảm bảo AI luôn trả về đầy đủ 5 tiêu chí: Khuyến nghị, Vùng giá, Tỷ trọng, Thời gian nắm giữ và Quản trị rủi ro.

---

## 2. Các thay đổi đề xuất (Proposed Changes)

Để triển khai Phương án C một cách tinh gọn nhất, chúng ta sẽ thực hiện các chỉnh sửa sau:

### Component: Backend Analysis Engine

#### [NEW] [stock_analyzer.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/stock_analyzer.py)
*   Tạo lớp `StockAnalyzer` chịu trách nhiệm:
    *   Phát hiện đường dẫn CSV lịch sử dựa trên mã cổ phiếu (ví dụ: `data/vn30_historical_csv/FPT_5y_daily.csv`).
    *   Tính toán `SMA20`, `SMA50`, `SMA200`, `RSI(14)`, `ATR(14)` bằng Pandas.
    *   Trả về một dictionary chứa đầy đủ các thông số định lượng và khuyến nghị kỹ thuật cơ bản.

### Component: Backend Orchestration & Agent

#### [MODIFY] [chat_pipeline.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/pipeline/chat_pipeline.py)
*   Import `VN30_TICKERS` và lớp `StockAnalyzer`.
*   Tại bước định tuyến (khi `route == Routes.FINANCIAL`):
    *   Kiểm tra xem câu hỏi hoặc truy vấn đã viết lại (`rewritten_query`) có chứa bất kỳ mã chứng khoán nào trong rổ VN30 hay không.
    *   Nếu phát hiện mã chứng khoán (ví dụ: `FPT`):
        *   Gọi `StockAnalyzer.analyze(ticker)` để tính toán chỉ số định lượng.
        *   Tích hợp khối dữ liệu JSON định lượng này vào câu hỏi chuyển đến `FinancialAgent`.

#### [MODIFY] [financial_agent.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/agents/financial_agent.py)
*   Cập nhật system prompt của `FinancialAgent` để nhận dạng dữ liệu định lượng cổ phiếu.
*   Yêu cầu AI tổng hợp câu trả lời theo đúng **Đầu ra tiêu chuẩn ngắn gọn** mà Kafi yêu cầu:
    1.  **Mục tiêu**: Tối ưu lợi nhuận & Quản trị rủi ro.
    2.  **Khuyến nghị hành động (Signal)**: Mua (Buy), Bán (Sell), Nắm giữ (Hold), hoặc Quan sát thêm (Watchlist).
    3.  **Vùng giá**: Vùng mua an toàn, Giá mục tiêu, Giá cắt lỗ bắt buộc.
    4.  **Thời gian nắm giữ**: T+ (Ngắn hạn), Trung hạn (3-6 tháng), hoặc Dài hạn.
    5.  **Tỷ trọng giải ngân**: % vốn khuyên dùng (ví dụ: giải ngân thăm dò 10-20%).

### Component: Project Graph & Documentation

#### [NEW] [Backend_Stock_Analyzer.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Stock_Analyzer.md)
*   Viết tài liệu mô tả chi tiết node `StockAnalyzer` mới (Input, Output, Nhiệm vụ, file vật lý, kết nối).

#### [MODIFY] [README.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/README.md)
*   Thêm node `Backend_Stock_Analyzer` vào sơ đồ kiến trúc Mermaid và danh mục liên kết để đảm bảo Graph của chúng ta luôn cập nhật chính xác 100%.

---

## 3. Câu hỏi và Thảo luận với Khách hàng (User Review Required)

> [!IMPORTANT]
> Hãy cho tôi biết ý kiến của bạn về các điểm dưới đây trước khi tôi tiến hành lập checklist chi tiết và thực thi:
> 
> 1.  **Về phương pháp Hybrid**: Bạn có đồng ý với phương án sử dụng Pandas để tính toán chỉ báo định lượng kỹ thuật (`SMA`, `RSI`, `ATR`) trước rồi nạp kết quả nén vào LLM để đảm bảo độ chính xác toán học và tiết kiệm token không?
> 2.  **Về các chỉ báo kỹ thuật đề xuất**: Bộ chỉ báo gồm `SMA` (xu hướng), `RSI` (động lượng quá mua/bán), và `ATR` (dùng để tính vùng Stop-loss khoa học bằng $CurrentPrice - 2 \times ATR$) đã đủ phù hợp với nhu cầu cơ bản chưa, hay bạn muốn bổ sung thêm chỉ báo nào khác?
> 3.  **Về cách trình bày**: Kết quả phân tích sẽ được AI trả về trực tiếp dưới dạng một khối văn bản Markdown định dạng tuyệt đẹp, ngắn gọn và dễ hiểu ngay trong khung chat của giao diện React hiện tại. Bạn có muốn thiết kế thêm một Card giao diện React riêng biệt cho phần này không? (Phương án hiển thị trực tiếp bằng Markdown là tinh gọn và nhanh nhất).
