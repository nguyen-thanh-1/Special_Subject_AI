from __future__ import annotations

import os
from datetime import date, timedelta, datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from src.utils.logger import logger
import json
from pathlib import Path
import pandas as pd

router = APIRouter(prefix="/api/vn30", tags=["vn30"])

# ─── Configuration ────────────────────────────────────────────────────────
VN30_TICKERS: List[str] = [
    "ACB", "BCM", "BID", "BVH", "CTG",
    "FPT", "GAS", "GVR", "HDB", "HPG",
    "LPB", "MBB", "MSN", "MWG", "PLX",
    "POW", "SAB", "SHB", "SSB", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB",
    "VIC", "VJC", "VNM", "VPB", "VRE",
]

CACHE_DIR = Path("data/market_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _save_cache(filename: str, data: any):
    try:
        with open(CACHE_DIR / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[vn30] failed to save cache {filename}: {e}")

def _load_cache(filename: str) -> Optional[any]:
    path = CACHE_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[vn30] failed to load cache {filename}: {e}")
        return None

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
    Tries Finlens first, falls back to local cache if API fails.
    """
    try:
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
                # Try to find symbol in previous cache if exists
                cached_quotes = _load_cache("vn30_quotes.json")
                if cached_quotes:
                    sym_cache = next((q for q in cached_quotes if q["symbol"] == symbol), None)
                    if sym_cache:
                        results.append(sym_cache)
                        continue
                results.append({"symbol": symbol, "close": None, "change_pct": None})

        if results:
            _save_cache("vn30_quotes.json", results)
        return results

    except Exception as e:
        logger.error(f"[vn30] API quotes fetch failed: {e}. Falling back to full cache.")
        cached = _load_cache("vn30_quotes.json")
        if cached:
            return cached
        raise HTTPException(status_code=503, detail="Market data unavailable (API error and no cache)")


@router.get("/ohlcv/{symbol}")
def get_stock_ohlcv(
    symbol: str,
    timeframe: str = Query(default="1m", description="One of: 1d, 5d, 1m, 3m, 1y, 5y"),
):
    """
    Return OHLCV candlestick data for a given symbol and timeframe.
    Tries API first, falls back to 'data/market_cache/ohlcv_{symbol}_{tf}.json' if fails.
    """
    symbol = symbol.upper()
    tf = timeframe.lower()
    cache_file = f"ohlcv_{symbol}_{tf}.json"

    if symbol not in VN30_TICKERS:
        raise HTTPException(status_code=404, detail=f"'{symbol}' is not in VN30 index")

    try:
        start_str, end_str = _date_range_for_timeframe(tf)
        cli = _get_client()

        if tf in ("1d", "5d"):
            # Intraday 15m candles
            df = cli.intraday.stock.ohlcv(symbol=symbol, start=start_str, end=end_str, interval="15m")
        else:
            # End-of-day daily candles
            df = cli.eod.stock.ohlcv(symbol=symbol, start=start_str, end=end_str)

        if df is None or len(df) == 0:
            cached = _load_cache(cache_file)
            return cached if cached else []

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

        if candles:
            _save_cache(cache_file, candles)
        return candles

    except Exception as e:
        logger.error(f"[vn30] ohlcv fetch failed for {symbol}/{tf}: {e}. Trying cache.")
        cached = _load_cache(cache_file)
        if cached:
            return cached
        raise HTTPException(status_code=503, detail=f"OHLCV data for {symbol} unavailable")

def pre_fetch_market_data():
    """Background task to pre-fetch and cache VN30 quotes and export CSVs on startup."""
    logger.info("[vn30] starting background pre-fetch and CSV export...")
    try:
        # 1. Fetch quotes for JSON cache (for sidebar)
        get_vn30_quotes()
        
        # 2. Export 5-year historical CSVs
        from src.utils.market_exporter import export_vn30_historical_csv
        export_vn30_historical_csv()
        
        logger.info("[vn30] background pre-fetch and CSV export complete.")
    except Exception as e:
        logger.warning(f"[vn30] background pre-fetch failed: {e}")

