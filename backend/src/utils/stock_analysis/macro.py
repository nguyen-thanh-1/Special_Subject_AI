"""
Macro & Advanced — Vĩ mô và định giá nâng cao
=================================================
Hầu hết cần dữ liệu BCTC. Riêng market_liquidity tính được từ OHLCV.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
#  1. NAV — Net Asset Value
# ═══════════════════════════════════════════════════════════════════════

def nav(
    total_assets: float,
    total_liabilities: float,
    num_units: float,
) -> float:
    """
    NAV — Giá trị tài sản ròng.

    Công thức:
        NAV = (Tổng_tài_sản − Tổng_nợ) / Số_đơn_vị_quỹ

    Ý nghĩa:
        Dùng cho quỹ đầu tư và ETF. Tính cuối ngày giao dịch.

    ⚠️  DỮ LIỆU CẦN: Bảng CĐKT của quỹ.
        - total_assets (float): Tổng tài sản.
        - total_liabilities (float): Tổng nợ.
        - num_units (float): Số đơn vị quỹ (hoặc số CP lưu hành).

    Trả về:  float — NAV per unit.
    """
    if num_units == 0:
        return np.nan
    return float((total_assets - total_liabilities) / num_units)


# ═══════════════════════════════════════════════════════════════════════
#  2. WACC — Weighted Average Cost of Capital
# ═══════════════════════════════════════════════════════════════════════

def wacc(
    equity: float,
    debt: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float,
) -> float:
    """
    WACC — Chi phí vốn bình quân gia quyền.

    Công thức:
        V    = E + D
        WACC = (E/V) × Re + (D/V) × Rd × (1 − T)

    Ý nghĩa:
        Tỷ suất chiết khấu khi tính DCF.
        ROIC > WACC → doanh nghiệp tạo giá trị thực sự.

    ⚠️  DỮ LIỆU CẦN: BCTC + thị trường.
        - equity (float): Vốn chủ sở hữu (E).
        - debt (float): Nợ vay (D).
        - cost_of_equity (float): Chi phí vốn CSH (Re), ví dụ 0.12 cho 12%.
        - cost_of_debt (float): Chi phí nợ (Rd), ví dụ 0.08 cho 8%.
        - tax_rate (float): Thuế suất (T), ví dụ 0.20 cho 20%.

    Trả về:  float — WACC (dạng thập phân, ví dụ 0.095 cho 9.5%).
    """
    v = equity + debt
    if v == 0:
        return np.nan
    return float((equity / v) * cost_of_equity + (debt / v) * cost_of_debt * (1 - tax_rate))


# ═══════════════════════════════════════════════════════════════════════
#  3. DCF — Discounted Cash Flow
# ═══════════════════════════════════════════════════════════════════════

def dcf(
    free_cash_flows: list[float],
    wacc_rate: float,
    terminal_growth: float,
    shares_outstanding: float | None = None,
) -> dict:
    """
    DCF — Chiết khấu dòng tiền tự do.

    Công thức:
        DCF = Σ[ FCFt / (1+WACC)^t ] + Terminal Value / (1+WACC)^n
        Terminal Value = FCFn × (1+g) / (WACC − g)

    Ý nghĩa:
        Phương pháp định giá toàn diện nhất.
        Giá trị nội tại = tổng PV của dòng tiền tự do tương lai.

    ⚠️  DỮ LIỆU CẦN: Báo cáo lưu chuyển tiền tệ + dự phóng.
        - free_cash_flows (list[float]): Dòng tiền tự do dự kiến các năm tới.
        - wacc_rate (float): WACC (từ hàm wacc()), ví dụ 0.10 cho 10%.
        - terminal_growth (float): Tốc độ tăng trưởng vĩnh viễn (g), ví dụ 0.03.
        - shares_outstanding (float|None): Số CP lưu hành (tính giá trị/CP).

    Trả về:
        dict — {"pv_fcf": float, "terminal_value": float,
                "enterprise_value": float, "per_share": float|None}

    Ví dụ:
        >>> result = dcf(
        ...     free_cash_flows=[5000, 5500, 6000, 6500, 7000],
        ...     wacc_rate=0.10, terminal_growth=0.03,
        ...     shares_outstanding=1_000_000
        ... )
    """
    if wacc_rate <= terminal_growth:
        return {"error": "WACC phải > terminal_growth"}

    n = len(free_cash_flows)
    # PV of projected FCFs
    pv_fcf = sum(
        fcf / (1 + wacc_rate) ** (t + 1)
        for t, fcf in enumerate(free_cash_flows)
    )

    # Terminal Value
    last_fcf = free_cash_flows[-1] if free_cash_flows else 0
    tv = last_fcf * (1 + terminal_growth) / (wacc_rate - terminal_growth)
    pv_tv = tv / (1 + wacc_rate) ** n

    ev = pv_fcf + pv_tv

    result = {
        "pv_fcf": round(pv_fcf, 2),
        "terminal_value": round(pv_tv, 2),
        "enterprise_value": round(ev, 2),
        "per_share": None,
    }
    if shares_outstanding and shares_outstanding > 0:
        result["per_share"] = round(ev / shares_outstanding, 2)

    return result


# ═══════════════════════════════════════════════════════════════════════
#  4. Thanh khoản thị trường — Market Liquidity
# ═══════════════════════════════════════════════════════════════════════

def market_liquidity(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """
    Thanh khoản thị trường — Market Liquidity.

    Công thức:
        Thanh khoản phiên  = Khối_lượng × Giá_đóng_cửa
        Thanh khoản BQ(n)  = SMA(n) của thanh khoản phiên

    Ý nghĩa:
        Thanh khoản cao → dễ mua/bán, chênh lệch bid-ask nhỏ.
        Theo dõi theo phiên và bình quân 20 phiên.

    ✅  TÍNH ĐƯỢC TỪ CSV OHLCV:
        - close  (pd.Series): Giá đóng cửa (cột "Close").
        - volume (pd.Series): Khối lượng giao dịch (cột "Volume").

    Tham số:
        - window (int): Số phiên tính trung bình. Mặc định 20.

    Trả về:
        pd.DataFrame — ["daily_value", "avg_value"]

    Ví dụ:
        >>> liq = market_liquidity(df["Close"], df["Volume"])
    """
    daily = close * volume
    avg = daily.rolling(window=window).mean()
    return pd.DataFrame({
        "daily_value": daily,
        "avg_value": avg,
    }, index=close.index)


# ═══════════════════════════════════════════════════════════════════════
#  5. Room ngoại — Foreign Ownership
# ═══════════════════════════════════════════════════════════════════════

def foreign_ownership_room(
    max_foreign_pct: float,
    current_foreign_pct: float,
) -> float:
    """
    Room ngoại — Tỷ lệ sở hữu nước ngoài còn lại.

    Công thức:
        Room_còn_lại = Tỷ_lệ_tối_đa − Tỷ_lệ_NĐT_nước_ngoài_hiện_tại

    Ý nghĩa:
        Đặc thù thị trường Việt Nam.
        Hết room ngoại → NĐT nước ngoài không mua thêm được.
        Giới hạn thông thường: 49% hoặc 100% tùy ngành.

    ⚠️  DỮ LIỆU CẦN: Từ sàn HOSE/HNX.
        - max_foreign_pct (float): Tỷ lệ sở hữu NN tối đa (%), ví dụ 49.0.
        - current_foreign_pct (float): Tỷ lệ NN hiện tại (%), ví dụ 42.5.

    Trả về:  float — Room còn lại (%).
    """
    return float(max_foreign_pct - current_foreign_pct)
