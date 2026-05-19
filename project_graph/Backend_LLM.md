# Module: Backend LLM (Core LLM & Text Generation Engine)

## 1. Mô tả chi tiết (Detailed Description)
Module này chịu trách nhiệm quản lý vòng đời và thực thi suy luận của các mô hình ngôn ngữ lớn (**Large Language Models - LLMs**) trên GPU. Được định nghĩa trong lớp `LLMManager` phối hợp cùng lớp chuyên biệt `HFTextGen` (Hugging Face Text Generation), module này cung cấp một động cơ sinh văn bản dạng stream (dòng chảy từng chữ) với hiệu năng tối ưu, hỗ trợ cả cấu hình LLM đám mây và LLM cục bộ chạy offline.

Để giải quyết bài toán giới hạn phần cứng (VRAM) của máy chủ chạy các mô hình AI lớn (Ví dụ: Qwen-8B, DeepSeek-Distill 7B), module triển khai các giải pháp kiến trúc cực kỳ chuyên nghiệp:
1.  **Lazy Model Loading (Tải mô hình lười)**: Hệ thống không tự động nạp các model LLM nặng hàng chục GB vào VRAM ngay khi khởi động (trừ khi cấu hình yêu cầu tải trước). Thay vào đó, mô hình chỉ thực sự được tải khi nhận câu hỏi chat đầu tiên, giúp tiết kiệm tối đa tài nguyên lúc chờ.
2.  **VRAM Clean Swapping (Giải phóng VRAM triệt để)**: Khi người dùng thực hiện đổi mô hình qua dropdown trên giao diện, lớp `LLMManager` thực hiện quy trình giải phóng bộ nhớ nghiêm ngặt:
    - Ngắt liên kết đối tượng mô hình cũ (`self._model = None`).
    - Gọi dọn rác hệ thống Python (`gc.collect()`).
    - Gọi xóa bộ nhớ đệm CUDA (`torch.cuda.empty_cache()`).
    Quy trình này đảm bảo thu hồi tối đa 100% dung lượng VRAM trước khi nạp mô hình mới, tránh hoàn toàn lỗi tràn bộ nhớ đồ họa (**Out of Memory - OOM**).
3.  **4-bit / 8-bit Quantization (Lượng tử hóa)**: Lớp `HFTextGen` tích hợp thư viện `bitsandbytes` hỗ trợ tải mô hình cục bộ dưới dạng nén lượng tử 4-bit hoặc 8-bit. Việc này giảm kích thước bộ nhớ VRAM cần thiết xuống **4 lần** (ví dụ model 8B chỉ tốn khoảng 5.5GB VRAM thay vì 16GB), giúp chạy mô hình cực mượt trên các GPU dòng Geforce thương mại.
4.  **Hỗ trợ Causal & Multimodal Models**: Tự động nhận diện lớp mô hình để sử dụng `AutoModelForCausalLM` hoặc `AutoModelForVision2Seq` (phục vụ các mô hình phân tích đa phương tiện hình ảnh/văn bản).
5.  **LLM Performance Auditing (Đo lường hiệu năng)**: Lọc và đo đạc chính xác các chỉ số quan trọng của từng phiên suy luận bao gồm:
    - **TTFT (Time-To-First-Token)**: Thời gian từ lúc nhận request đến chữ đầu tiên xuất hiện (giây) - thước đo độ nhạy cảm của hệ thống.
    - **Speed (Tokens/sec)**: Tốc độ sinh chữ của mô hình.
    - **Aborted Rate**: Ghi nhận tỷ lệ người dùng chủ động ngắt phản hồi giữa chừng để tối ưu hóa tài nguyên server.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Quản lý Vòng đời LLM**: Nạp (`ensure_loaded`), chuyển đổi (`switch_model`), và giải phóng (`unload`) mô hình khỏi bộ nhớ đồ họa một cách an toàn và đồng bộ hóa qua khóa `Lock`.
-   **Tải mô hình cục bộ qua Hugging Face**: Khởi tạo cấu hình tokenizer, xác định thiết bị chạy (`cuda`), cấu hình tham số lượng tử hóa `BitsAndBytesConfig`, tải weights mô hình từ thư mục cục bộ hoặc tải tự động từ Hugging Face Hub.
-   **Điều khiển Suy luận Stream**:
    -   Sử dụng lớp `TextIteratorStreamer` chạy trên luồng phụ để tạo vòng lặp yielding trả về từng token văn bản bất đồng bộ.
    -   Thiết lập tham số sinh: `max_new_tokens` (độ dài tối đa), `temperature` (độ sáng tạo), `top_p`, `repetition_penalty`.
    -   Áp dụng luật dừng (`StoppingCriteria`) để chặn mô hình lặp lại vô hạn hoặc sinh ra các từ khóa cấm (`bad_words_ids`).
-   **Đo lường & Log Metrics**: Ghi nhận thời gian, đếm số token đầu ra và xuất bảng số liệu đẹp mắt bằng `logger.log_llm_metrics` khi kết thúc phiên.

## 3. Đầu vào (Inputs)
-   `model_id` (chuỗi ký tự): ID mô hình muốn chuyển đổi.
-   `user_input` (chuỗi ký tự): Câu hỏi mở rộng cần LLM suy luận.
-   `history` (danh sách tin nhắn dạng Role-Content): Ngữ cảnh trò chuyện để mô hình bám sát.
-   Cấu hình mô hình từ `llms.yaml`.

## 4. Đầu ra (Outputs)
-   `Generator[str, None, None]`: Luồng stream chứa các token chữ phản hồi liên tục.
-   Số liệu thống kê hiệu năng in ra bảng Rich Console tại Server.

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối**:
    -   Trình quản lý mô hình chung: [llm.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/llm.py)
    -   Động cơ sinh text HuggingFace: [hf_textgen.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/hf_textgen.py)
-   **Đường dẫn tương đối**: `./backend/src/utils/llm.py` và `./backend/src/utils/hf_textgen.py`

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   [Backend_Config.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Config.md) (`./Backend_Config.md`): Đọc danh sách cấu hình mô hình từ `AppConfig`.
    -   [Backend_System_Monitor.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_System_Monitor.md) (`./Backend_System_Monitor.md`): Ghi log thông số đo lường LLM Metrics và báo lỗi nạp mô hình. Gọi hàm đo lường `get_vram_snapshot()` từ `vram.py` để theo dõi sự thay đổi bộ nhớ khi hoán swap mô hình.
-   **Được gọi bởi (Called by / Dependency of):**
    -   [Backend_Main.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Main.md) (`./Backend_Main.md`): Thực hiện gọi nạp trước LLM mặc định khi khởi động ứng dụng FastAPI.
    -   [Backend_Router_Chatbot.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Router_Chatbot.md) (`./Backend_Router_Chatbot.md`): Endpoint API lấy thông tin mô hình hiện tại, danh sách mô hình và hoán đổi mô hình.
    -   [Backend_Agents.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Agents.md) (`./Backend_Agents.md`): Lớp Agent gọi để thực hiện sinh câu phân tích hoặc viết lại câu hỏi truy vấn.
    -   [Backend_App_Mode.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_App_Mode.md) (`./Backend_App_Mode.md`): Giải phóng hoàn toàn LLM ra khỏi VRAM khi chuyển sang chế độ `INDEXING` nạp tài liệu.
