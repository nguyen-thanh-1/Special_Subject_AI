from __future__ import annotations

import os
from datetime import date, timedelta, datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from src.utils.logger import logger

router = APIRouter(prefix="/api/vn30", tags=["vn30"])

# ─── VN30 constituents (hardcoded, updated Jul 2024) ───────────────────────
VN30_TICKERS: List[str] = [
    "ACB", "BCM", "BID", "BVH", "CTG",
    "FPT", "GAS", "GVR", "HDB", "HPG",
    "LPB", "MBB", "MSN", "MWG", "PLX",
    "POW", "SAB", "SHB", "SSB", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB",
    "VIC", "VJC", "VNM", "VPB", "VRE",
]

_FINLENS_API_KEY = os.getenv("FINLENS_API_KEY", "")

def _get_client():
    """Lazy-init finlens client."""
    if not _FINLENS_API_KEY:
        raise HTTPException(status_code=503, detail="FINLENS_API_KEY not configured")
    try:
        import finlens as fn
        return fn.client(apiKey=_FINLENS_API_KEY)
    except ImportError:
        raise HTTPException(status_code=503, detail="finlens package not installed. Run: pip install finlens")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to init finlens client: {e}")


def _date_range_for_timeframe(tf: str):
    """Return (start, end) date strings for a given timeframe code."""
    today = date.today()
    end_str = today.strftime("%Y-%m-%d")
    deltas = {
        "1d":  timedelta(days=2),      # today + yesterday buffer
        "5d":  timedelta(days=7),
        "1m":  timedelta(days=31),
        "3m":  timedelta(days=92),
        "1y":  timedelta(days=365),
        "5y":  timedelta(days=365 * 5),
    }
    delta = deltas.get(tf, timedelta(days=31))
    start_str = (today - delta).strftime("%Y-%m-%d")
    return start_str, end_str


def _interval_for_timeframe(tf: str) -> str:
    """Return appropriate finlens interval for a timeframe."""
    if tf in ("1d", "5d"):
        return "15m"   # intraday granularity
    return "1d"        # EOD for longer periods


# ─── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/tickers")
def get_vn30_tickers():
    """Return the list of VN30 constituent tickers."""
    return {"tickers": VN30_TICKERS, "count": len(VN30_TICKERS)}


@router.get("/quotes")
def get_vn30_quotes():
    """
    Return latest EOD quotes for all VN30 stocks.
    Returns: list of {symbol, open, high, low, close, volume, change, change_pct}
    """
    cli = _get_client()
    today = date.today()
    start = (today - timedelta(days=5)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    results = []
    for symbol in VN30_TICKERS:
        try:
            stock = cli.eod.stock
            df = stock.ohlcv(symbol=symbol, start=start, end=end)
            if df is None or len(df) == 0:
                continue

            df = df.sort_values("Date") if "Date" in df.columns else df.sort_index()
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else latest

            close = float(latest.get("Close", latest.get("close", 0)))
            open_ = float(latest.get("Open", latest.get("open", close)))
            high = float(latest.get("High", latest.get("high", close)))
            low = float(latest.get("Low", latest.get("low", close)))
            volume = float(latest.get("Volume", latest.get("volume", 0)))
            prev_close = float(prev.get("Close", prev.get("close", close)))

            change = close - prev_close
            change_pct = (change / prev_close * 100) if prev_close != 0 else 0.0

            results.append({
                "symbol": symbol,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
            })
        except Exception as e:
            logger.warning(f"[vn30] failed to fetch quote for {symbol}: {e}")
            results.append({"symbol": symbol, "close": None, "change_pct": None})

    return results


@router.get("/ohlcv/{symbol}")
def get_stock_ohlcv(
    symbol: str,
    timeframe: str = Query(default="1m", description="One of: 1d, 5d, 1m, 3m, 1y, 5y"),
):
    """
    Return OHLCV candlestick data for a given symbol and timeframe.
    For 1d/5d: uses intraday (15m intervals).
    For 1m/3m/1y/5y: uses EOD daily data.
    """
    symbol = symbol.upper()
    if symbol not in VN30_TICKERS:
        raise HTTPException(status_code=404, detail=f"'{symbol}' is not in VN30 index")

    tf = timeframe.lower()
    start_str, end_str = _date_range_for_timeframe(tf)
    cli = _get_client()

    try:
        if tf in ("1d", "5d"):
            # Intraday 15m candles
            df = cli.intraday.stock.ohlcv(symbol=symbol, start=start_str, end=end_str, interval="15m")
        else:
            # End-of-day daily candles
            df = cli.eod.stock.ohlcv(symbol=symbol, start=start_str, end=end_str)

        if df is None or len(df) == 0:
            return []

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        date_col = "date" if "date" in df.columns else df.columns[0]
        df = df.sort_values(date_col)

        candles = []
        for _, row in df.iterrows():
            dt = row[date_col]
            if hasattr(dt, "timestamp"):
                ts = int(dt.timestamp())
            else:
                ts = int(datetime.strptime(str(dt)[:10], "%Y-%m-%d").timestamp())

            candles.append({
                "time": ts,
                "open": float(row.get("open", row.get("close", 0))),
                "high": float(row.get("high", row.get("close", 0))),
                "low": float(row.get("low", row.get("close", 0))),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
            })

        return candles

    except Exception as e:
        logger.error(f"[vn30] ohlcv fetch failed for {symbol}/{tf}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
