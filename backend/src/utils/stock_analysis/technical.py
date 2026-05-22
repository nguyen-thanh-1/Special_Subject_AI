"""
Technical Analysis — Phân tích kỹ thuật
========================================
Tất cả các hàm trong module này CHỈ CẦN dữ liệu OHLCV (Open, High, Low, Close, Volume).
Dữ liệu nguồn: CSV files tại  data/vn30_historical_csv/<SYMBOL>_5y_daily.csv

Cột CSV: Symbol, Date, Open, High, Low, Close, Volume
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
#  1. MA — Đường trung bình động (Moving Averages)
# ═══════════════════════════════════════════════════════════════════════

def sma(close: pd.Series, window: int = 20) -> pd.Series:
    """
    SMA — Simple Moving Average (Đường trung bình động đơn giản).

    Công thức:
        SMA(n) = (P1 + P2 + ... + Pn) / n

    Dữ liệu cần:
        - close (pd.Series): Chuỗi giá đóng cửa (cột "Close" từ CSV).

    Tham số:
        - window (int): Số phiên tính trung bình. Mặc định 20.
          Thông dụng: 20 (ngắn hạn), 50 (trung hạn), 200 (dài hạn).

    Trả về:
        pd.Series — Chuỗi giá trị SMA, NaN cho các phiên đầu chưa đủ dữ liệu.

    Ví dụ:
        >>> df = pd.read_csv("data/vn30_historical_csv/FPT_5y_daily.csv")
        >>> sma_20 = sma(df["Close"], window=20)
        >>> sma_50 = sma(df["Close"], window=50)
    """
    return close.rolling(window=window).mean()


def ema(close: pd.Series, span: int = 12) -> pd.Series:
    """
    EMA — Exponential Moving Average (Đường trung bình động hàm mũ).

    Công thức:
        EMA(n) = Giá_hôm_nay × k  +  EMA_hôm_qua × (1 − k)
        k = 2 / (n + 1)

    Dữ liệu cần:
        - close (pd.Series): Chuỗi giá đóng cửa (cột "Close" từ CSV).

    Tham số:
        - span (int): Số phiên. Mặc định 12.
          Thông dụng: 12 (MACD fast), 26 (MACD slow), 9 (MACD signal).

    Trả về:
        pd.Series — Chuỗi giá trị EMA.

    Ví dụ:
        >>> ema_12 = ema(df["Close"], span=12)
        >>> ema_26 = ema(df["Close"], span=26)
    """
    return close.ewm(span=span, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════════════
#  2. RSI — Chỉ số sức mạnh tương đối (Relative Strength Index)
# ═══════════════════════════════════════════════════════════════════════

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI — Relative Strength Index (Chỉ số sức mạnh tương đối).

    Công thức:
        RSI = 100 − [100 / (1 + RS)]
        RS  = Trung_bình_tăng / Trung_bình_giảm  (trong 'period' phiên)

    Ý nghĩa:
        RSI > 70 → quá mua (overbought), có thể giảm giá.
        RSI < 30 → quá bán (oversold), có thể tăng giá.
        Dao động trong khoảng 0–100.

    Dữ liệu cần:
        - close (pd.Series): Chuỗi giá đóng cửa.

    Tham số:
        - period (int): Số phiên. Mặc định 14 (chuẩn J. Welles Wilder).

    Trả về:
        pd.Series — Chuỗi giá trị RSI (0–100).

    Ví dụ:
        >>> rsi_14 = rsi(df["Close"], period=14)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing (EMA-style)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ═══════════════════════════════════════════════════════════════════════
#  3. MACD — Moving Average Convergence Divergence
# ═══════════════════════════════════════════════════════════════════════

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence Divergence.

    Công thức:
        MACD Line  = EMA(fast) − EMA(slow)       (mặc định: EMA12 − EMA26)
        Signal Line = EMA(signal) của MACD Line   (mặc định: EMA9)
        Histogram   = MACD Line − Signal Line

    Ý nghĩa:
        MACD cắt lên Signal → tín hiệu MUA.
        MACD cắt xuống Signal → tín hiệu BÁN.
        Histogram dương → xu hướng tăng.

    Dữ liệu cần:
        - close (pd.Series): Chuỗi giá đóng cửa.

    Tham số:
        - fast (int): Chu kỳ EMA nhanh. Mặc định 12.
        - slow (int): Chu kỳ EMA chậm. Mặc định 26.
        - signal (int): Chu kỳ đường tín hiệu. Mặc định 9.

    Trả về:
        pd.DataFrame với 3 cột: ["macd_line", "signal_line", "histogram"]

    Ví dụ:
        >>> result = macd(df["Close"])
        >>> result["macd_line"], result["signal_line"], result["histogram"]
    """
    ema_fast = ema(close, span=fast)
    ema_slow = ema(close, span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
    }, index=close.index)


# ═══════════════════════════════════════════════════════════════════════
#  4. Bollinger Bands — Dải Bollinger
# ═══════════════════════════════════════════════════════════════════════

def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands — Dải Bollinger.

    Công thức:
        Middle = SMA(window)
        Upper  = SMA(window) + num_std × σ(window)
        Lower  = SMA(window) − num_std × σ(window)
        σ = Độ lệch chuẩn của giá đóng cửa trong 'window' phiên.

    Ý nghĩa:
        Giá chạm dải trên → có thể quá mua.
        Giá chạm dải dưới → có thể quá bán.
        Dải thu hẹp → sắp bùng nổ biến động (squeeze).

    Dữ liệu cần:
        - close (pd.Series): Chuỗi giá đóng cửa.

    Tham số:
        - window (int): Chu kỳ SMA. Mặc định 20.
        - num_std (float): Bội số độ lệch chuẩn. Mặc định 2.0.

    Trả về:
        pd.DataFrame với 3 cột: ["upper", "middle", "lower"]

    Ví dụ:
        >>> bb = bollinger_bands(df["Close"])
        >>> bb["upper"], bb["middle"], bb["lower"]
    """
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    return pd.DataFrame({
        "upper": middle + num_std * std,
        "middle": middle,
        "lower": middle - num_std * std,
    }, index=close.index)


# ═══════════════════════════════════════════════════════════════════════
#  5. Stochastic Oscillator — Dao động Stochastic
# ═══════════════════════════════════════════════════════════════════════

def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator — Dao động Stochastic.

    Công thức:
        %K = (Đóng_cửa − Thấp_nhất(n)) / (Cao_nhất(n) − Thấp_nhất(n)) × 100
        %D = SMA(d_period) của %K

    Ý nghĩa:
        %K > 80 → quá mua.   %K < 20 → quá bán.
        Thường dùng tham số (14,3) hoặc (5,3).

    Dữ liệu cần:
        - high  (pd.Series): Chuỗi giá cao nhất  (cột "High").
        - low   (pd.Series): Chuỗi giá thấp nhất (cột "Low").
        - close (pd.Series): Chuỗi giá đóng cửa  (cột "Close").

    Trả về:
        pd.DataFrame với 2 cột: ["%K", "%D"]

    Ví dụ:
        >>> stoch = stochastic_oscillator(df["High"], df["Low"], df["Close"])
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    denom = highest_high - lowest_low
    k = ((close - lowest_low) / denom.replace(0, np.nan)) * 100
    d = k.rolling(window=d_period).mean()

    return pd.DataFrame({"%K": k, "%D": d}, index=close.index)


# ═══════════════════════════════════════════════════════════════════════
#  6. OBV — On-Balance Volume (Khối lượng cân bằng)
# ═══════════════════════════════════════════════════════════════════════

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV — On-Balance Volume (Khối lượng cân bằng).

    Công thức:
        Nếu đóng cửa tăng: OBV = OBV_hôm_qua + Volume
        Nếu đóng cửa giảm: OBV = OBV_hôm_qua − Volume
        Nếu không đổi:      OBV = OBV_hôm_qua

    Ý nghĩa:
        OBV tăng khi giá tăng → xác nhận xu hướng.
        OBV divergence với giá → cảnh báo đảo chiều.

    Dữ liệu cần:
        - close  (pd.Series): Chuỗi giá đóng cửa (cột "Close").
        - volume (pd.Series): Chuỗi khối lượng    (cột "Volume").

    Trả về:
        pd.Series — Chuỗi giá trị OBV tích lũy.

    Ví dụ:
        >>> obv_series = obv(df["Close"], df["Volume"])
    """
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ═══════════════════════════════════════════════════════════════════════
#  7. ADX — Average Directional Index (Chỉ số xu hướng trung bình)
# ═══════════════════════════════════════════════════════════════════════

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    ADX — Average Directional Index (Chỉ số xu hướng trung bình).

    Công thức:
        +DM = High_today − High_yesterday  (nếu > 0 và > −DM)
        −DM = Low_yesterday − Low_today     (nếu > 0 và > +DM)
        TR  = max(H−L, |H−C_prev|, |L−C_prev|)
        +DI = Smoothed(+DM) / Smoothed(TR) × 100
        −DI = Smoothed(−DM) / Smoothed(TR) × 100
        DX  = |+DI − −DI| / (+DI + −DI) × 100
        ADX = SMA(period) of DX

    Ý nghĩa:
        ADX > 25 → xu hướng mạnh.  ADX < 20 → sideway.
        Không cho biết chiều hướng, chỉ đo sức mạnh xu hướng.

    Dữ liệu cần:
        - high  (pd.Series): Cột "High".
        - low   (pd.Series): Cột "Low".
        - close (pd.Series): Cột "Close".

    Tham số:
        - period (int): Chu kỳ. Mặc định 14.

    Trả về:
        pd.DataFrame với 3 cột: ["+DI", "-DI", "ADX"]

    Ví dụ:
        >>> result = adx(df["High"], df["Low"], df["Close"])
        >>> result["ADX"]
    """
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    # Wilder's smoothing
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = (smooth_plus / atr.replace(0, np.nan)) * 100
    minus_di = (smooth_minus / atr.replace(0, np.nan)) * 100

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx_val = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame({
        "+DI": plus_di,
        "-DI": minus_di,
        "ADX": adx_val,
    }, index=close.index)


# ═══════════════════════════════════════════════════════════════════════
#  8. Fibonacci Retracement — Thoái lui Fibonacci
# ═══════════════════════════════════════════════════════════════════════

def fibonacci_retracement(
    high: pd.Series,
    low: pd.Series,
) -> dict[str, float]:
    """
    Fibonacci Retracement — Thoái lui Fibonacci.

    Công thức:
        Mức_thoái_lui = Đỉnh − (Đỉnh − Đáy) × Tỉ_lệ
        Tỉ lệ: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%

    Ý nghĩa:
        Các mức hỗ trợ/kháng cự tiềm năng khi giá điều chỉnh.
        61.8% (Golden Ratio) là mức quan trọng nhất.

    Dữ liệu cần:
        - high (pd.Series): Cột "High" (dùng tìm đỉnh trong khoảng thời gian).
        - low  (pd.Series): Cột "Low"  (dùng tìm đáy trong khoảng thời gian).

    Trả về:
        dict — {"peak", "trough", "0.0%", "23.6%", "38.2%", "50.0%", "61.8%", "78.6%", "100.0%"}

    Ví dụ:
        >>> # Fibonacci cho 60 phiên gần nhất
        >>> levels = fibonacci_retracement(df["High"].tail(60), df["Low"].tail(60))
        >>> levels["61.8%"]
    """
    peak = float(high.max())
    trough = float(low.min())
    diff = peak - trough

    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {f"{r*100:.1f}%": round(peak - diff * r, 4) for r in ratios}
    levels["peak"] = peak
    levels["trough"] = trough
    return levels
