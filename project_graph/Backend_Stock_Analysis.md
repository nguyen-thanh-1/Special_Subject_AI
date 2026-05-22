# Module: Backend Stock Analysis (Stock Analysis Library)

## 1. Mô tả chi tiết (Detailed Description)
Module này là một **thư viện phân tích cổ phiếu toàn diện** (Stock Analysis Library), được tổ chức dưới dạng Python Package tại `backend/src/utils/stock_analysis/`. Thư viện cung cấp **35 hàm tính toán** bao phủ 6 nhóm chỉ số phân tích tài chính: Phân tích kỹ thuật, Quản trị rủi ro, Định giá, Hiệu suất kinh doanh, Cổ tức, và Vĩ mô nâng cao.

Kiến trúc chia tách module theo nhóm chức năng:
1.  **`technical.py`** — 9 hàm phân tích kỹ thuật (SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, OBV, ADX, Fibonacci Retracement). Tất cả chỉ cần dữ liệu OHLCV từ CSV.
2.  **`risk.py`** — 6 hàm quản trị rủi ro (Beta, Sharpe Ratio, VaR, Max Drawdown, D/E Ratio, Current Ratio). 4 hàm tính được từ OHLCV, 2 cần BCTC.
3.  **`valuation.py`** — 6 hàm định giá (P/E, P/B, P/S, EV/EBITDA, PEG, Market Cap). Tất cả cần dữ liệu BCTC bổ sung.
4.  **`performance.py`** — 7 hàm hiệu suất kinh doanh (EPS, ROE, ROA, ROIC, GPM, NPM, Asset Turnover). Tất cả cần BCTC.
5.  **`dividend.py`** — 4 hàm cổ tức (DPS, Dividend Yield, Payout Ratio, DDM Gordon). Tất cả cần dữ liệu cổ tức.
6.  **`macro.py`** — 5 hàm vĩ mô nâng cao (NAV, WACC, DCF, Market Liquidity, Foreign Ownership Room). 1 hàm tính từ OHLCV, 4 cần BCTC.

Mỗi hàm đều có **docstring chi tiết bằng tiếng Việt** bao gồm: công thức tính, ý nghĩa chỉ số, dữ liệu đầu vào cần thiết, tham số tùy chỉnh, giá trị trả về, và ví dụ sử dụng. Các hàm cần dữ liệu chưa có sẵn được đánh dấu `⚠️ DỮ LIỆU CẦN BỔ SUNG` trong docstring.

## 2. Nhiệm vụ và Trách nhiệm (Responsibilities)
-   **Phân tích kỹ thuật (Technical Analysis)**: Tính toán các chỉ báo xu hướng (MA, MACD, ADX), dao động (RSI, Stochastic, Bollinger), khối lượng (OBV), và hỗ trợ/kháng cự (Fibonacci) từ dữ liệu giá lịch sử.
-   **Quản trị rủi ro (Risk Management)**: Đo lường biến động (Beta), hiệu quả điều chỉnh rủi ro (Sharpe), mức tổn thất tiềm ẩn (VaR, Max Drawdown), và tình trạng nợ (D/E, Current Ratio).
-   **Định giá (Valuation)**: Cung cấp các hệ số định giá tương đối (P/E, P/B, P/S, EV/EBITDA, PEG) và tuyệt đối (Market Cap).
-   **Hiệu suất kinh doanh (Performance)**: Đánh giá khả năng sinh lời (EPS, ROE, ROA, ROIC) và hiệu quả hoạt động (GPM, NPM, Asset Turnover).
-   **Cổ tức (Dividend)**: Phân tích chính sách cổ tức (DPS, Yield, Payout Ratio) và định giá dòng cổ tức (DDM Gordon).
-   **Vĩ mô & nâng cao (Macro & Advanced)**: Định giá nội tại (DCF, NAV, WACC), đánh giá thanh khoản thị trường, và room sở hữu nước ngoài.

## 3. Đầu vào (Inputs)
-   **Dữ liệu OHLCV** từ 30 tệp CSV tại `data/vn30_historical_csv/` (các cột: `Symbol`, `Date`, `Open`, `High`, `Low`, `Close`, `Volume`).
-   **Dữ liệu BCTC** (khi có): Lợi nhuận ròng, Tổng tài sản, Tổng nợ, Vốn CSH, Doanh thu, EBITDA, Số CP lưu hành, v.v.
-   **Dữ liệu cổ tức** (khi có): Tổng cổ tức chi trả, DPS.
-   **Dữ liệu thị trường bổ sung** (khi có): VN-Index (cho Beta), Room nước ngoài.

## 4. Đầu ra (Outputs)
-   **pd.Series**: Các chuỗi chỉ số theo thời gian (SMA, EMA, RSI, OBV, v.v.).
-   **pd.DataFrame**: Bảng nhiều cột cho các chỉ báo phức hợp (MACD, Bollinger, Stochastic, ADX, Market Liquidity).
-   **dict**: Kết quả điểm (Fibonacci levels, Max Drawdown details, DCF valuation).
-   **float**: Giá trị đơn (P/E, ROE, Sharpe, Beta, EPS, v.v.).

## 5. File/Thư mục vật lý (Physical Files)
-   **Đường dẫn tuyệt đối package**: [stock_analysis/](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/stock_analysis/)
-   **Đường dẫn tương đối**: `./backend/src/utils/stock_analysis/`
-   **Các file trong package**:
    -   [__init__.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/stock_analysis/__init__.py) — Entry-point, re-export toàn bộ 35 hàm.
    -   [technical.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/stock_analysis/technical.py) — 9 hàm phân tích kỹ thuật.
    -   [risk.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/stock_analysis/risk.py) — 6 hàm quản trị rủi ro.
    -   [valuation.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/stock_analysis/valuation.py) — 6 hàm định giá.
    -   [performance.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/stock_analysis/performance.py) — 7 hàm hiệu suất kinh doanh.
    -   [dividend.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/stock_analysis/dividend.py) — 4 hàm cổ tức.
    -   [macro.py](file:///c:/Users/Admin/Desktop/Kafi_chatbot/backend/src/utils/stock_analysis/macro.py) — 5 hàm vĩ mô & nâng cao.
-   **Nguồn dữ liệu CSV**: `backend/data/vn30_historical_csv/` (30 tệp, ví dụ `FPT_5y_daily.csv`).

## 6. Liên kết Đồ thị (Graph Connections)
-   **Gọi đến (Calls to / Depends on):**
    -   Thư viện bên ngoài: `numpy`, `pandas`, `scipy.stats` (chỉ cho VaR parametric).
    -   Dữ liệu CSV được xuất bởi [Backend_Market_Exporter.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Market_Exporter.md) (`./Backend_Market_Exporter.md`).
-   **Được gọi bởi (Called by / Dependency of):**
    -   Hiện tại là thư viện độc lập (standalone library). Có thể được tích hợp bởi [Backend_Agents.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Agents.md) hoặc [Backend_Pipeline.md](file:///c:/Users/Admin/Desktop/Kafi_chatbot/project_graph/Backend_Pipeline.md) khi xây dựng tool phân tích tự động cho chatbot.

## 7. Bảng tóm tắt 35 hàm

| # | Hàm | Module | Dữ liệu OHLCV | Ghi chú |
|---|-----|--------|:--------------:|---------|
| 1 | `sma()` | technical | ✅ | SMA(20,50,200) |
| 2 | `ema()` | technical | ✅ | EMA(12,26,9) |
| 3 | `rsi()` | technical | ✅ | Wilder's smoothing |
| 4 | `macd()` | technical | ✅ | Line + Signal + Histogram |
| 5 | `bollinger_bands()` | technical | ✅ | Upper + Middle + Lower |
| 6 | `stochastic_oscillator()` | technical | ✅ | %K + %D |
| 7 | `obv()` | technical | ✅ | Cần Close + Volume |
| 8 | `adx()` | technical | ✅ | +DI, −DI, ADX |
| 9 | `fibonacci_retracement()` | technical | ✅ | 7 mức Fibonacci |
| 10 | `beta()` | risk | ⚠️ | Cần thêm VN-Index CSV |
| 11 | `sharpe_ratio()` | risk | ✅ | Giả định Rf = 3.5% |
| 12 | `value_at_risk()` | risk | ✅ | Historical + Parametric |
| 13 | `max_drawdown()` | risk | ✅ | Peak-to-trough |
| 14 | `debt_to_equity()` | risk | ❌ | Cần Bảng CĐKT |
| 15 | `current_ratio()` | risk | ❌ | Cần Bảng CĐKT |
| 16 | `price_to_earnings()` | valuation | ❌ | Cần EPS từ BCTC |
| 17 | `price_to_book()` | valuation | ❌ | Cần BVPS từ BCTC |
| 18 | `price_to_sales()` | valuation | ❌ | Cần Doanh thu |
| 19 | `ev_ebitda()` | valuation | ❌ | Cần EV + EBITDA |
| 20 | `peg_ratio()` | valuation | ❌ | Cần P/E + EPS growth |
| 21 | `market_cap()` | valuation | ❌ | Cần shares outstanding |
| 22 | `eps()` | performance | ❌ | Cần lợi nhuận ròng |
| 23 | `roe()` | performance | ❌ | Cần LNST + Vốn CSH |
| 24 | `roa()` | performance | ❌ | Cần LNST + Tổng TS |
| 25 | `roic()` | performance | ❌ | Cần EBIT + Vốn ĐT |
| 26 | `gross_profit_margin()` | performance | ❌ | Cần DT + GVHB |
| 27 | `net_profit_margin()` | performance | ❌ | Cần LNST + DT |
| 28 | `asset_turnover()` | performance | ❌ | Cần DT + Tổng TS |
| 29 | `dps()` | dividend | ❌ | Cần cổ tức chi trả |
| 30 | `dividend_yield()` | dividend | ❌ | Cần DPS |
| 31 | `payout_ratio()` | dividend | ❌ | Cần DPS + EPS |
| 32 | `ddm_gordon()` | dividend | ❌ | Cần D1, r, g |
| 33 | `nav()` | macro | ❌ | Cần TS + Nợ |
| 34 | `wacc()` | macro | ❌ | Cần E, D, Re, Rd, T |
| 35 | `dcf()` | macro | ❌ | Cần FCF + WACC |
| 36 | `market_liquidity()` | macro | ✅ | Close × Volume |
| 37 | `foreign_ownership_room()` | macro | ❌ | Cần dữ liệu HOSE/HNX |
