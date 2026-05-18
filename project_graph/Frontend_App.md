# Module: Frontend Application (Vite React Client)

## 1. Mô tả chi tiết (Detailed Description)
Module này chịu trách nhiệm cho giao diện người dùng (User Interface) trực quan của ứng dụng **Kafi Chatbot**. Được xây dựng dưới dạng ứng dụng đơn trang (**Single Page Application - SPA**) sử dụng thư viện **React**, ngôn ngữ **TypeScript** và công cụ build siêu tốc **Vite**. 

Giao diện được thiết kế theo phong cách hiện đại với chủ đề tối màu (dark theme), chia thành 4 cột chức năng chính linh hoạt:
1.  **Thanh điều hướng bên trái (Navigation Sidebar)**: Chứa các biểu tượng điều hướng chính (Thị trường, Biểu đồ, Danh mục, Lịch sử, AI Support, Quản lý kiến thức).
2.  **Cột Danh sách Cổ phiếu / Chat overlay**: Hiển thị bảng theo dõi (Watchlist) của rổ VN30 hoặc tích hợp phần quản lý RAG tùy thuộc vào tab đang hoạt động.
3.  **Bảng điều khiển Biểu đồ chính**: Hiển thị biểu đồ nến chứng khoán tương tác sử dụng thư viện hiệu năng cao **lightweight-charts** của TradingView.
4.  **Bảng đặt lệnh (Order Board)**: Cho phép thao tác đặt lệnh mua/bán giả lập theo thời gian thực.

Module này tích hợp cơ chế đồng bộ hóa dữ liệu thời gian thực mạnh mẽ:
-   **Real-time Polling**: Định kỳ mỗi 5 giây gửi request lấy giá cổ phiếu mới nhất từ `/api/vn30/quotes` và cập nhật biểu đồ nến.
-   **SSE (Server-Sent Events) Streaming**: Khi người dùng nhắn tin với trợ lý ảo AI, ứng dụng sử dụng API `fetch` đọc luồng stream `ReadableStream` trả về từ backend, giúp hiển thị phản hồi của AI theo từng chữ (token streaming) vô cùng mượt mà.
-   **Định dạng văn bản & công thức nâng cao**: Tích hợp các bộ lọc và parse markdown (`react-markdown`, `remark-gfm`) kết hợp hiển thị công thức toán học tài chính qua LaTeX (`remark-math`, `rehype-katex`) và chuẩn hóa bảng dữ liệu Markdown để hiển thị bảng biểu cực kỳ gọn gàng.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Vẽ Biểu đồ Kỹ thuật**: Khởi tạo và quản lý vòng đời của biểu đồ nến TradingView thông qua tham chiếu container `chartContainerRef`.
-   **Đồng bộ Giá cổ phiếu**: Tải danh sách cổ phiếu VN30 và cập nhật giá hiện tại mỗi 5 giây để cập nhật tức thời bảng watchlist và nến cuối cùng.
-   **Điều phối Hội thoại AI**:
    -   Quản lý lịch sử chat và tự động cuộn xuống dưới cùng (`scrollChatToBottom`) khi có tin nhắn mới hoặc đang stream.
    -   Tạo ngẫu nhiên `sessionId` độc nhất khi mở trang để định danh cuộc trò chuyện trên backend.
    -   Giao tiếp API `/api/chat` bằng phương thức POST để nhận và giải mã luồng stream văn bản.
    -   Hỗ trợ chuyển đổi mô hình ngôn ngữ thông qua dropdown chọn model kết nối tới `/api/chat/model`.
-   **Nạp & Quản lý Tài liệu RAG**:
    -   Cung cấp khu vực kéo thả (drag & drop) hoặc click chọn file PDF để tải tài liệu lên backend.
    -   Hiển thị danh sách file đang xử lý kèm trạng thái (`processing`, `completed`, `error`) bằng cách gọi `/api/chat/knowledge/files` mỗi 10 giây.
    -   Cung cấp nút chuyển đổi Chế độ ứng dụng (`CHAT` vs `INDEXING`) thông qua API `/api/chat/app-mode` để tối ưu hóa VRAM của hệ thống.
    -   Cho phép xóa tệp tin khỏi bộ não vector database bằng cách gửi yêu cầu DELETE.

## 3. Đầu vào (Inputs)
-   **Tương tác của người dùng**: Nhập tin nhắn, click nút đặt lệnh, chọn mã cổ phiếu, upload file PDF, kéo thả file, đổi chế độ app.
-   **API Responses từ Backend**:
    -   `GET /api/vn30/quotes`: Danh sách báo giá VN30 mới nhất.
    -   `GET /api/vn30/ohlcv/{symbol}`: Dữ liệu nến lịch sử chứng khoán.
    -   `POST /api/chat`: Luồng văn bản stream phản hồi từ chatbot.
    -   `GET /api/chat/models`: Danh sách các model AI khả dụng.
    -   `GET /api/chat/knowledge/files`: Danh sách các tài liệu trong DB.
    -   `GET /api/chat/app-mode`: Trạng thái hệ thống hiện tại.

## 4. Đầu ra (Outputs)
-   **Giao diện DOM hoàn chỉnh**: Cấu trúc HTML/CSS sống động, phản hồi mượt mà trên trình duyệt.
-   **API Requests**: Các gói tin GET/POST/DELETE gửi đến backend theo đúng chuẩn schema.
-   **File tải lên**: Các byte dữ liệu của file PDF được gửi đi dưới dạng `FormData` chứa file.

## 5. File/Thư mục vật lý (Physical Files)
-   **Thư mục gốc**: [frontend](file:///c:/Users/Admin/Desktop/Kafi_chatbot/frontend/)
-   **Files chính**:
    -   [App.tsx](file:///c:/Users/Admin/Desktop/Kafi_chatbot/frontend/src/App.tsx): File logic giao diện cốt lõi (51KB).
    -   [main.tsx](file:///c:/Users/Admin/Desktop/Kafi_chatbot/frontend/src/main.tsx): Điểm gắn React app vào DOM (`root`).
    -   [index.css](file:///c:/Users/Admin/Desktop/Kafi_chatbot/frontend/src/index.css): Định nghĩa các style cơ bản và tùy biến scrollbar, màu nền.

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Router_Chatbot.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_Chatbot.md) (`./Backend_Router_Chatbot.md`): Gửi câu hỏi chat, đổi model LLM, đổi mode CHAT/INDEXING, xem/xóa file tri thức.
    -   [Backend_Router_VN30.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_VN30.md) (`./Backend_Router_VN30.md`): Polling danh sách giá VN30 và truy vấn dữ liệu vẽ biểu đồ nến OHLCV.
-   **Được gọi bởi (Called by / Dependency of):**
    -   Đây là module phía Client, hiển thị trực tiếp cho Người dùng cuối (User).
