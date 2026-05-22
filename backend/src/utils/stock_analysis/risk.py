"""
Risk Analysis — Quản trị rủi ro
=================================
Các chỉ số đánh giá rủi ro đầu tư.

Dữ liệu OHLCV tính được:  Beta, Sharpe Ratio, VaR, Max Drawdown
Cần thêm BCTC:            D/E Ratio, Current Ratio
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
#  1. Beta — Hệ số biến động
# ═══════════════════════════════════════════════════════════════════════

def beta(
    stock_close: pd.Series,
    market_close: pd.Series,
) -> float:
    """
    Beta — Hệ số biến động so với thị trường.

    Công thức:
        Beta = Cov(Ri, Rm) / Var(Rm)
        Ri = Lợi suất hàng ngày của cổ phiếu
        Rm = Lợi suất hàng ngày của thị trường (VN-Index)

    Ý nghĩa:
        β > 1  → biến động mạnh hơn thị trường.
        β < 1  → ổn định hơn thị trường.
        β = 0  → không tương quan.
        β < 0  → nghịch chiều thị trường.

    Dữ liệu cần:
        - stock_close  (pd.Series): Giá đóng cửa cổ phiếu (cột "Close").
        - market_close (pd.Series): Giá đóng cửa VN-Index.
          ⚠️ Cần thêm CSV VN-Index (chưa có sẵn, cần bổ sung).

    Trả về:
        float — Giá trị Beta.

    Ví dụ:
        >>> stock = pd.read_csv("data/vn30_historical_csv/FPT_5y_daily.csv")
        >>> vnindex = pd.read_csv("data/vnindex_daily.csv")  # cần bổ sung
        >>> b = beta(stock["Close"], vnindex["Close"])
    """
    stock_ret = stock_close.pct_change().dropna()
    market_ret = market_close.pct_change().dropna()

    # Align index
    aligned = pd.concat([stock_ret, market_ret], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return np.nan

    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else np.nan


# ═══════════════════════════════════════════════════════════════════════
#  2. Sharpe Ratio — Tỷ lệ Sharpe
# ═══════════════════════════════════════════════════════════════════════

def sharpe_ratio(
    close: pd.Series,
    risk_free_rate: float = 0.035,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe Ratio — Lợi suất được điều chỉnh theo rủi ro.

    Công thức:
        Sharpe = (Rp − Rf) / σp
        Rp = Lợi suất trung bình hàng năm của danh mục/cổ phiếu
        Rf = Lãi suất phi rủi ro (mặc định 3.5%/năm tại VN)
        σp = Độ lệch chuẩn hàng năm

    Ý nghĩa:
        Sharpe > 1 → tốt.  Sharpe > 2 → rất tốt.
        Số càng cao → hiệu quả đầu tư càng cao.

    Dữ liệu cần:
        - close (pd.Series): Giá đóng cửa (cột "Close").

    Tham số:
        - risk_free_rate (float): Lãi suất phi rủi ro hàng năm. Mặc định 0.035 (3.5%).
        - periods_per_year (int): Số phiên giao dịch/năm. Mặc định 252.

    Trả về:
        float — Giá trị Sharpe Ratio.

    Ví dụ:
        >>> sr = sharpe_ratio(df["Close"], risk_free_rate=0.04)
    """
    daily_ret = close.pct_change().dropna()
    if daily_ret.empty:
        return np.nan

    ann_return = daily_ret.mean() * periods_per_year
    ann_std = daily_ret.std() * np.sqrt(periods_per_year)

    return float((ann_return - risk_free_rate) / ann_std) if ann_std != 0 else np.nan


# ═══════════════════════════════════════════════════════════════════════
#  3. VaR — Value at Risk (Giá trị rủi ro)
# ═══════════════════════════════════════════════════════════════════════

def value_at_risk(
    close: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    VaR — Value at Risk (Giá trị rủi ro).

    Phương pháp tham số:
        VaR(95%) = μ − 1.645 × σ

    Phương pháp lịch sử:
        VaR = percentile(daily_returns, 1 − confidence)

    Ý nghĩa:
        Mức tổn thất tối đa trong ngày bình thường với xác suất 95%.
        VD: VaR = −2.5% nghĩa là có 95% khả năng bạn không mất quá 2.5% trong 1 ngày.

    Dữ liệu cần:
        - close (pd.Series): Giá đóng cửa (cột "Close").

    Tham số:
        - confidence (float): Mức tin cậy. Mặc định 0.95.
        - method (str): "historical" hoặc "parametric". Mặc định "historical".

    Trả về:
        float — Giá trị VaR (lợi suất âm, ví dụ −0.025 nghĩa là −2.5%).

    Ví dụ:
        >>> var_95 = value_at_risk(df["Close"], confidence=0.95)
        >>> var_99 = value_at_risk(df["Close"], confidence=0.99, method="parametric")
    """
    daily_ret = close.pct_change().dropna()
    if daily_ret.empty:
        return np.nan

    if method == "parametric":
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)
        return float(daily_ret.mean() + z * daily_ret.std())
    else:  # historical
        return float(np.percentile(daily_ret, (1 - confidence) * 100))


# ═══════════════════════════════════════════════════════════════════════
#  4. Maximum Drawdown — Drawdown tối đa
# ═══════════════════════════════════════════════════════════════════════

def max_drawdown(close: pd.Series) -> dict:
    """
    Maximum Drawdown — Mức sụt giảm lớn nhất từ đỉnh xuống đáy.

    Công thức:
        Max DD = (Đỉnh − Đáy) / Đỉnh × 100%

    Ý nghĩa:
        Càng thấp → quản trị rủi ro càng tốt.
        VD: −30% nghĩa là giá đã từng giảm 30% từ đỉnh gần nhất.

    Dữ liệu cần:
        - close (pd.Series): Giá đóng cửa (cột "Close").

    Trả về:
        dict — {"max_drawdown_pct": float, "peak_value": float,
                "trough_value": float, "peak_idx": index, "trough_idx": index}

    Ví dụ:
        >>> mdd = max_drawdown(df["Close"])
        >>> print(f"Max Drawdown: {mdd['max_drawdown_pct']:.2f}%")
    """
    cumulative_max = close.cummax()
    drawdown = (close - cumulative_max) / cumulative_max

    trough_idx = drawdown.idxmin()
    peak_idx = close.loc[:trough_idx].idxmax()

    peak_val = float(close.loc[peak_idx])
    trough_val = float(close.loc[trough_idx])
    mdd_pct = float((trough_val - peak_val) / peak_val * 100) if peak_val != 0 else np.nan

    return {
        "max_drawdown_pct": mdd_pct,
        "peak_value": peak_val,
        "trough_value": trough_val,
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
    }


# ═══════════════════════════════════════════════════════════════════════
#  5. D/E Ratio — Tỷ lệ nợ/vốn chủ sở hữu
# ═══════════════════════════════════════════════════════════════════════

def debt_to_equity(
    total_debt: float,
    total_equity: float,
) -> float:
    """
    D/E Ratio — Tỷ lệ nợ trên vốn chủ sở hữu.

    Công thức:
        D/E = Tổng_nợ_vay / Vốn_chủ_sở_hữu

    Ý nghĩa:
        D/E cao → đòn bẩy tài chính cao, rủi ro cao nhưng khuếch đại lợi nhuận.
        Ngân hàng thường có D/E rất cao (bình thường trong ngành).

    ⚠️  DỮ LIỆU CHƯA CÓ SẴN — Cần Bảng cân đối kế toán (Balance Sheet):
        - total_debt  (float): Tổng nợ vay (từ BCTC dòng "Nợ phải trả").
        - total_equity (float): Vốn chủ sở hữu (từ BCTC dòng "Vốn chủ sở hữu").

    Trả về:
        float — Giá trị D/E.

    Ví dụ:
        >>> de = debt_to_equity(total_debt=50_000_000, total_equity=30_000_000)
    """
    if total_equity == 0:
        return np.nan
    return float(total_debt / total_equity)


# ═══════════════════════════════════════════════════════════════════════
#  6. Current Ratio — Tỷ lệ thanh toán hiện hành
# ═══════════════════════════════════════════════════════════════════════

def current_ratio(
    current_assets: float,
    current_liabilities: float,
) -> float:
    """
    Current Ratio — Tỷ lệ thanh toán hiện hành.

    Công thức:
        Current Ratio = Tài_sản_ngắn_hạn / Nợ_ngắn_hạn

    Ý nghĩa:
        > 1  → có thể thanh toán nợ ngắn hạn.
        < 1  → rủi ro thanh khoản.
        Lý tưởng: 1.5 – 3.0.

    ⚠️  DỮ LIỆU CHƯA CÓ SẴN — Cần Bảng cân đối kế toán:
        - current_assets      (float): Tài sản ngắn hạn.
        - current_liabilities (float): Nợ ngắn hạn.

    Trả về:
        float — Giá trị Current Ratio.

    Ví dụ:
        >>> cr = current_ratio(current_assets=100_000, current_liabilities=60_000)
    """
    if current_liabilities == 0:
        return np.nan
    return float(current_assets / current_liabilities)
