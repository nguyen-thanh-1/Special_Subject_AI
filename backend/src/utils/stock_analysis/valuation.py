"""
Valuation — Định giá cổ phiếu
================================
⚠️  TẤT CẢ các hàm trong module này YÊU CẦU dữ liệu từ Báo cáo tài chính
    (Income Statement, Balance Sheet) mà CSV OHLCV hiện tại CHƯA CÓ.
    Khi có nguồn dữ liệu bổ sung, chỉ cần truyền tham số vào là tính được.
"""

from __future__ import annotations
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  1. P/E — Price to Earnings Ratio
# ═══════════════════════════════════════════════════════════════════════

def price_to_earnings(
    market_price: float,
    earnings_per_share: float,
) -> float:
    """
    P/E — Hệ số giá trên lợi nhuận.

    Công thức:
        P/E = Giá_thị_trường / EPS

    Ý nghĩa:
        Cho biết nhà đầu tư sẵn sàng trả bao nhiêu đồng cho 1 đồng lợi nhuận.
        P/E cao → kỳ vọng tăng trưởng cao hoặc cổ phiếu đắt.
        Benchmark VN-Index: P/E trung bình ~12–18x.

    ⚠️  DỮ LIỆU CẦN BỔ SUNG:
        - market_price (float): Giá thị trường hiện tại (có sẵn từ CSV cột "Close").
        - earnings_per_share (float): EPS = Lợi_nhuận_ròng / Số_CP_lưu_hành
          → Cần Báo cáo KQKD (lợi nhuận ròng) + số cổ phiếu lưu hành.

    Trả về:
        float — Hệ số P/E.

    Ví dụ:
        >>> pe = price_to_earnings(market_price=73.9, earnings_per_share=5.2)
    """
    if earnings_per_share == 0:
        return np.nan
    return float(market_price / earnings_per_share)


# ═══════════════════════════════════════════════════════════════════════
#  2. P/B — Price to Book Ratio
# ═══════════════════════════════════════════════════════════════════════

def price_to_book(
    market_price: float,
    book_value_per_share: float,
) -> float:
    """
    P/B — Hệ số giá trên giá trị sổ sách.

    Công thức:
        P/B  = Giá_thị_trường / BVPS
        BVPS = (Tổng_tài_sản − Tổng_nợ) / Số_CP_lưu_hành

    Ý nghĩa:
        So sánh giá thị trường với giá trị tài sản ròng.
        P/B < 1 → giá thấp hơn tài sản ròng (cơ hội hoặc cảnh báo).

    ⚠️  DỮ LIỆU CẦN BỔ SUNG:
        - market_price (float): Giá thị trường (cột "Close").
        - book_value_per_share (float): BVPS
          → Cần Bảng CĐKT: tổng tài sản, tổng nợ, số CP lưu hành.

    Trả về:
        float — Hệ số P/B.
    """
    if book_value_per_share == 0:
        return np.nan
    return float(market_price / book_value_per_share)


# ═══════════════════════════════════════════════════════════════════════
#  3. P/S — Price to Sales Ratio
# ═══════════════════════════════════════════════════════════════════════

def price_to_sales(
    market_cap: float,
    net_revenue: float,
) -> float:
    """
    P/S — Hệ số giá trên doanh thu.

    Công thức:
        P/S = Vốn_hóa_thị_trường / Doanh_thu_thuần
        Hoặc: P/S = Giá_CP / (Doanh_thu / Số_CP)

    Ý nghĩa:
        Dùng khi công ty chưa có lợi nhuận.
        P/S thấp hơn đồng ngành → tiềm năng tăng giá.

    ⚠️  DỮ LIỆU CẦN BỔ SUNG:
        - market_cap  (float): Vốn hóa = Giá × Số CP lưu hành.
        - net_revenue (float): Doanh thu thuần (Báo cáo KQKD).

    Trả về:
        float — Hệ số P/S.
    """
    if net_revenue == 0:
        return np.nan
    return float(market_cap / net_revenue)


# ═══════════════════════════════════════════════════════════════════════
#  4. EV/EBITDA
# ═══════════════════════════════════════════════════════════════════════

def ev_ebitda(
    market_cap: float,
    total_debt: float,
    cash: float,
    ebitda: float,
) -> float:
    """
    EV/EBITDA — Enterprise Value / EBITDA.

    Công thức:
        EV = Vốn_hóa + Nợ_vay − Tiền_mặt
        EV/EBITDA = EV / EBITDA
        EBITDA = EBIT + Khấu_hao & Phân_bổ

    Ý nghĩa:
        Định giá toàn diện hơn P/E vì loại trừ cấu trúc vốn và thuế.

    ⚠️  DỮ LIỆU CẦN BỔ SUNG:
        - market_cap (float): Vốn hóa thị trường.
        - total_debt (float): Tổng nợ vay (Bảng CĐKT).
        - cash       (float): Tiền và tương đương tiền (Bảng CĐKT).
        - ebitda     (float): EBITDA (Báo cáo KQKD + khấu hao).

    Trả về:
        float — Hệ số EV/EBITDA.
    """
    if ebitda == 0:
        return np.nan
    ev = market_cap + total_debt - cash
    return float(ev / ebitda)


# ═══════════════════════════════════════════════════════════════════════
#  5. PEG — Price/Earnings to Growth
# ═══════════════════════════════════════════════════════════════════════

def peg_ratio(
    pe_ratio: float,
    eps_growth_rate: float,
) -> float:
    """
    PEG — Hệ số P/E điều chỉnh theo tăng trưởng.

    Công thức:
        PEG = P/E / Tốc_độ_tăng_trưởng_EPS (%)

    Ý nghĩa:
        PEG < 1 → CP có thể bị định giá thấp so với tốc độ tăng trưởng.
        Peter Lynch phổ biến chỉ số này.

    ⚠️  DỮ LIỆU CẦN BỔ SUNG:
        - pe_ratio        (float): Hệ số P/E (tính từ hàm price_to_earnings).
        - eps_growth_rate (float): Tốc độ tăng trưởng EPS (%), ví dụ 15.0 cho 15%.
          → Cần EPS từ nhiều kỳ BCTC.

    Trả về:
        float — Hệ số PEG.
    """
    if eps_growth_rate == 0:
        return np.nan
    return float(pe_ratio / eps_growth_rate)


# ═══════════════════════════════════════════════════════════════════════
#  6. Market Cap — Vốn hóa thị trường
# ═══════════════════════════════════════════════════════════════════════

def market_cap(
    price: float,
    shares_outstanding: float,
) -> float:
    """
    Market Cap — Vốn hóa thị trường.

    Công thức:
        Market Cap = Giá_CP × Số_CP_lưu_hành

    Ý nghĩa:
        Quy mô thị trường của công ty.
        Large-cap:  > 10.000 tỷ VND
        Mid-cap:    1.000 – 10.000 tỷ VND
        Small-cap:  < 1.000 tỷ VND

    ⚠️  DỮ LIỆU CẦN BỔ SUNG:
        - price (float): Giá cổ phiếu (có sẵn từ CSV cột "Close").
        - shares_outstanding (float): Số cổ phiếu lưu hành (cần nguồn bổ sung).

    Trả về:
        float — Vốn hóa thị trường (đơn vị tiền tệ gốc).
    """
    return float(price * shares_outstanding)
