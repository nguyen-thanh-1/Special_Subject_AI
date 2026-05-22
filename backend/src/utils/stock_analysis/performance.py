"""
Performance — Hiệu suất kinh doanh
=====================================
⚠️  TẤT CẢ các hàm trong module này YÊU CẦU dữ liệu từ Báo cáo tài chính.
    Khi có nguồn dữ liệu, chỉ cần nạp tham số vào là tính được.
"""

from __future__ import annotations
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  1. EPS — Earnings Per Share
# ═══════════════════════════════════════════════════════════════════════

def eps(
    net_income: float,
    preferred_dividends: float = 0.0,
    avg_shares_outstanding: float = 1.0,
) -> float:
    """
    EPS — Lợi nhuận trên mỗi cổ phiếu.

    Công thức:
        EPS = (Lợi_nhuận_ròng − Cổ_tức_ưu_đãi) / Số_CP_lưu_hành_bình_quân

    Ý nghĩa:
        Chỉ số cơ bản nhất đo khả năng sinh lời. EPS tăng đều → kinh doanh tốt.

    ⚠️  DỮ LIỆU CẦN: Báo cáo KQKD + thông tin CP lưu hành.
        - net_income (float): Lợi nhuận ròng sau thuế.
        - preferred_dividends (float): Cổ tức ưu đãi (nếu có). Mặc định 0.
        - avg_shares_outstanding (float): Số CP lưu hành bình quân.

    Trả về:  float — Giá trị EPS.
    """
    if avg_shares_outstanding == 0:
        return np.nan
    return float((net_income - preferred_dividends) / avg_shares_outstanding)


# ═══════════════════════════════════════════════════════════════════════
#  2. ROE — Return on Equity
# ═══════════════════════════════════════════════════════════════════════

def roe(
    net_income: float,
    avg_equity: float,
) -> float:
    """
    ROE — Lợi nhuận trên vốn chủ sở hữu.

    Công thức:
        ROE = Lợi_nhuận_ròng / Vốn_CSH_bình_quân × 100%
        DuPont: ROE = Biên_lợi_nhuận × Vòng_quay_TS × Đòn_bẩy_TC

    Ý nghĩa:
        Đo hiệu quả sử dụng vốn cổ đông. ROE > 15% được coi là tốt.

    ⚠️  DỮ LIỆU CẦN: BCTC (lợi nhuận ròng, vốn CSH bình quân).
        - net_income (float): Lợi nhuận ròng sau thuế.
        - avg_equity (float): Vốn chủ sở hữu bình quân.

    Trả về:  float — ROE (%), ví dụ 15.5 nghĩa là 15.5%.
    """
    if avg_equity == 0:
        return np.nan
    return float(net_income / avg_equity * 100)


# ═══════════════════════════════════════════════════════════════════════
#  3. ROA — Return on Assets
# ═══════════════════════════════════════════════════════════════════════

def roa(
    net_income: float,
    avg_total_assets: float,
) -> float:
    """
    ROA — Lợi nhuận trên tổng tài sản.

    Công thức:
        ROA = Lợi_nhuận_ròng / Tổng_tài_sản_bình_quân × 100%

    Ý nghĩa:
        Đo hiệu quả sử dụng toàn bộ tài sản. ROA > 5% → tốt (DN thông thường).

    ⚠️  DỮ LIỆU CẦN: BCTC (lợi nhuận ròng, tổng tài sản bình quân).
        - net_income (float): Lợi nhuận ròng sau thuế.
        - avg_total_assets (float): Tổng tài sản bình quân.

    Trả về:  float — ROA (%).
    """
    if avg_total_assets == 0:
        return np.nan
    return float(net_income / avg_total_assets * 100)


# ═══════════════════════════════════════════════════════════════════════
#  4. ROIC — Return on Invested Capital
# ═══════════════════════════════════════════════════════════════════════

def roic(
    ebit: float,
    tax_rate: float,
    equity: float,
    long_term_debt: float,
) -> float:
    """
    ROIC — Lợi nhuận trên vốn đầu tư.

    Công thức:
        ROIC   = NOPAT / Vốn_đầu_tư
        NOPAT  = EBIT × (1 − Thuế_suất)
        Vốn ĐT = Vốn_CSH + Nợ_vay_dài_hạn

    Ý nghĩa:
        ROIC > WACC → doanh nghiệp tạo ra giá trị thực sự.

    ⚠️  DỮ LIỆU CẦN: BCTC.
        - ebit (float): Lợi nhuận trước lãi vay và thuế.
        - tax_rate (float): Thuế suất thực tế, ví dụ 0.20 cho 20%.
        - equity (float): Vốn chủ sở hữu.
        - long_term_debt (float): Nợ vay dài hạn.

    Trả về:  float — ROIC (%).
    """
    invested_capital = equity + long_term_debt
    if invested_capital == 0:
        return np.nan
    nopat = ebit * (1 - tax_rate)
    return float(nopat / invested_capital * 100)


# ═══════════════════════════════════════════════════════════════════════
#  5. GPM — Gross Profit Margin
# ═══════════════════════════════════════════════════════════════════════

def gross_profit_margin(
    revenue: float,
    cost_of_goods_sold: float,
) -> float:
    """
    GPM — Biên lợi nhuận gộp.

    Công thức:
        GPM = (Doanh_thu − Giá_vốn) / Doanh_thu × 100%

    Ý nghĩa:
        Phản ánh hiệu quả sản xuất và định giá sản phẩm.
        Biên cao → lợi thế cạnh tranh mạnh.

    ⚠️  DỮ LIỆU CẦN: Báo cáo KQKD.
        - revenue (float): Doanh thu thuần.
        - cost_of_goods_sold (float): Giá vốn hàng bán.

    Trả về:  float — GPM (%).
    """
    if revenue == 0:
        return np.nan
    return float((revenue - cost_of_goods_sold) / revenue * 100)


# ═══════════════════════════════════════════════════════════════════════
#  6. NPM — Net Profit Margin
# ═══════════════════════════════════════════════════════════════════════

def net_profit_margin(
    net_income: float,
    revenue: float,
) -> float:
    """
    NPM — Biên lợi nhuận ròng.

    Công thức:
        NPM = Lợi_nhuận_ròng / Doanh_thu × 100%

    Ý nghĩa:
        Phản ánh khả năng kiểm soát chi phí toàn diện.

    ⚠️  DỮ LIỆU CẦN: Báo cáo KQKD.
        - net_income (float): Lợi nhuận ròng sau thuế.
        - revenue (float): Doanh thu thuần.

    Trả về:  float — NPM (%).
    """
    if revenue == 0:
        return np.nan
    return float(net_income / revenue * 100)


# ═══════════════════════════════════════════════════════════════════════
#  7. Asset Turnover — Vòng quay tài sản
# ═══════════════════════════════════════════════════════════════════════

def asset_turnover(
    revenue: float,
    avg_total_assets: float,
) -> float:
    """
    Asset Turnover — Vòng quay tài sản.

    Công thức:
        Asset Turnover = Doanh_thu / Tổng_tài_sản_bình_quân

    Ý nghĩa:
        Đo hiệu quả sử dụng tài sản để tạo doanh thu.

    ⚠️  DỮ LIỆU CẦN: Báo cáo KQKD + Bảng CĐKT.
        - revenue (float): Doanh thu thuần.
        - avg_total_assets (float): Tổng tài sản bình quân.

    Trả về:  float — Hệ số vòng quay.
    """
    if avg_total_assets == 0:
        return np.nan
    return float(revenue / avg_total_assets)
