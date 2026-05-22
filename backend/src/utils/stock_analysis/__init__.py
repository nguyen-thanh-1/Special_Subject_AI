"""
Stock Analysis Library — Kafi Chatbot
======================================
Thư viện phân tích cổ phiếu toàn diện cho rổ VN30.

Cấu trúc:
    stock_analysis/
    ├── __init__.py          # Entry-point, re-export tất cả hàm
    ├── technical.py         # Phân tích kỹ thuật (MA, RSI, MACD, Bollinger, Stochastic, OBV, ADX, Fibonacci)
    ├── risk.py              # Quản trị rủi ro (Beta, Sharpe, VaR, MaxDrawdown, D/E, Current Ratio)
    ├── valuation.py         # Định giá (P/E, P/B, P/S, EV/EBITDA, PEG, Market Cap)
    ├── performance.py       # Hiệu suất kinh doanh (EPS, ROE, ROA, ROIC, GPM, NPM, Asset Turnover)
    ├── dividend.py          # Cổ tức (DPS, Dividend Yield, Payout Ratio, DDM)
    └── macro.py             # Vĩ mô & nâng cao (NAV, WACC, DCF, Thanh khoản, Room ngoại)

Sử dụng:
    from src.utils.stock_analysis import rsi, macd, sma, bollinger_bands
    from src.utils.stock_analysis import max_drawdown, beta, sharpe_ratio

    import pandas as pd
    df = pd.read_csv("data/vn30_historical_csv/FPT_5y_daily.csv")
    rsi_series = rsi(df["Close"])
"""

# ── Technical Analysis ──────────────────────────────────────────────
from .technical import (
    sma,
    ema,
    rsi,
    macd,
    bollinger_bands,
    stochastic_oscillator,
    obv,
    adx,
    fibonacci_retracement,
)

# ── Risk ────────────────────────────────────────────────────────────
from .risk import (
    beta,
    sharpe_ratio,
    value_at_risk,
    max_drawdown,
    debt_to_equity,
    current_ratio,
)

# ── Valuation ───────────────────────────────────────────────────────
from .valuation import (
    price_to_earnings,
    price_to_book,
    price_to_sales,
    ev_ebitda,
    peg_ratio,
    market_cap,
)

# ── Performance ─────────────────────────────────────────────────────
from .performance import (
    eps,
    roe,
    roa,
    roic,
    gross_profit_margin,
    net_profit_margin,
    asset_turnover,
)

# ── Dividend ────────────────────────────────────────────────────────
from .dividend import (
    dps,
    dividend_yield,
    payout_ratio,
    ddm_gordon,
)

# ── Macro & Advanced ────────────────────────────────────────────────
from .macro import (
    nav,
    wacc,
    dcf,
    market_liquidity,
    foreign_ownership_room,
)

__all__ = [
    # Technical
    "sma", "ema", "rsi", "macd", "bollinger_bands",
    "stochastic_oscillator", "obv", "adx", "fibonacci_retracement",
    # Risk
    "beta", "sharpe_ratio", "value_at_risk", "max_drawdown",
    "debt_to_equity", "current_ratio",
    # Valuation
    "price_to_earnings", "price_to_book", "price_to_sales",
    "ev_ebitda", "peg_ratio", "market_cap",
    # Performance
    "eps", "roe", "roa", "roic",
    "gross_profit_margin", "net_profit_margin", "asset_turnover",
    # Dividend
    "dps", "dividend_yield", "payout_ratio", "ddm_gordon",
    # Macro
    "nav", "wacc", "dcf", "market_liquidity", "foreign_ownership_room",
]
