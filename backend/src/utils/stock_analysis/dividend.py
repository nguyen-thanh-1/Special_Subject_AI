"""
Dividend — Cổ tức
===================
⚠️  TẤT CẢ các hàm yêu cầu dữ liệu cổ tức và BCTC (chưa có trong CSV OHLCV).
"""

from __future__ import annotations
import numpy as np


def dps(
    total_dividends_paid: float,
    shares_outstanding: float,
) -> float:
    """
    DPS — Cổ tức mỗi cổ phiếu (Dividends Per Share).

    Công thức:
        DPS = Tổng_cổ_tức_chi_trả / Số_CP_lưu_hành

    ⚠️  DỮ LIỆU CẦN: Báo cáo lưu chuyển tiền tệ / thông báo cổ tức.
        - total_dividends_paid (float): Tổng cổ tức chi trả trong kỳ.
        - shares_outstanding (float): Số cổ phiếu lưu hành.

    Trả về:  float — DPS.
    """
    if shares_outstanding == 0:
        return np.nan
    return float(total_dividends_paid / shares_outstanding)


def dividend_yield(
    dividends_per_share: float,
    market_price: float,
) -> float:
    """
    Dividend Yield — Tỷ suất cổ tức.

    Công thức:
        Dividend Yield = DPS / Giá_thị_trường × 100%

    Ý nghĩa:
        > 5% thường hấp dẫn. So sánh với lãi suất ngân hàng để đánh giá.

    ⚠️  DỮ LIỆU CẦN:
        - dividends_per_share (float): DPS (từ hàm dps()).
        - market_price (float): Giá thị trường (có sẵn từ CSV cột "Close").

    Trả về:  float — Tỷ suất cổ tức (%).
    """
    if market_price == 0:
        return np.nan
    return float(dividends_per_share / market_price * 100)


def payout_ratio(
    dividends_per_share: float,
    earnings_per_share: float,
) -> float:
    """
    Payout Ratio — Tỷ lệ chi trả cổ tức.

    Công thức:
        Payout Ratio = DPS / EPS × 100%

    Ý nghĩa:
        < 40% → công ty giữ lợi nhuận tái đầu tư.
        > 80% → chi trả gần hết, ít dư địa tăng trưởng.

    ⚠️  DỮ LIỆU CẦN: DPS + EPS (từ BCTC).
        - dividends_per_share (float): DPS.
        - earnings_per_share (float): EPS.

    Trả về:  float — Payout Ratio (%).
    """
    if earnings_per_share == 0:
        return np.nan
    return float(dividends_per_share / earnings_per_share * 100)


def ddm_gordon(
    next_dividend: float,
    required_rate: float,
    growth_rate: float,
) -> float:
    """
    DDM — Dividend Discount Model (Gordon Growth Model).

    Công thức:
        P = D1 / (r − g)
        D1 = Cổ tức kỳ tới
        r  = Tỷ suất yêu cầu của nhà đầu tư
        g  = Tốc độ tăng trưởng cổ tức (g < r)

    Ý nghĩa:
        Định giá CP dựa trên dòng cổ tức tương lai chiết khấu về hiện tại.
        Chỉ áp dụng khi g < r.

    ⚠️  DỮ LIỆU CẦN:
        - next_dividend (float): Cổ tức dự kiến kỳ tới (D1).
        - required_rate (float): Tỷ suất yêu cầu, ví dụ 0.12 cho 12%.
        - growth_rate   (float): Tốc độ tăng trưởng cổ tức, ví dụ 0.05 cho 5%.

    Trả về:  float — Giá trị nội tại cổ phiếu theo DDM.

    Ví dụ:
        >>> fair_value = ddm_gordon(next_dividend=3000, required_rate=0.12, growth_rate=0.05)
    """
    if required_rate <= growth_rate:
        return np.nan  # g phải < r
    return float(next_dividend / (required_rate - growth_rate))
