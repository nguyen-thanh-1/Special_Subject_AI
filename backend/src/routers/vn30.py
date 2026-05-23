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

OFFLINE_MODE = os.getenv("FINLENS_OFFLINE", "true").lower() == "true"
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


def _generate_quotes_from_dataset() -> List[dict]:
    results = []
    historical_dir = Path("data/vn30_historical_csv")
    if not historical_dir.exists():
        historical_dir = Path("backend/data/vn30_historical_csv")
    
    for symbol in VN30_TICKERS:
        csv_path = historical_dir / f"{symbol}_5y_daily.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if not df.empty and len(df) >= 2:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date").reset_index(drop=True)
                    
                    latest = df.iloc[-1]
                    prev = df.iloc[-2]
                    
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
                    continue
            except Exception as e:
                logger.warning(f"[vn30] failed to read CSV quote for {symbol}: {e}")
        
        results.append({
            "symbol": symbol,
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "close": 0.0,
            "volume": 0.0,
            "change": 0.0,
            "change_pct": 0.0,
        })
    return results


def _get_ohlcv_from_dataset(symbol: str, tf: str) -> List[dict]:
    historical_dir = Path("data/vn30_historical_csv")
    if not historical_dir.exists():
        historical_dir = Path("backend/data/vn30_historical_csv")
    
    csv_path = historical_dir / f"{symbol}_5y_daily.csv"
    if not csv_path.exists():
        return []
        
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return []
            
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        mapping = {
            "1d": 2,
            "5d": 5,
            "1m": 22,
            "3m": 66,
            "1y": 252,
            "5y": len(df)
        }
        limit = mapping.get(tf, 22)
        df_sliced = df.tail(limit)
        
        candles = []
        for _, row in df_sliced.iterrows():
            dt = row["Date"]
            ts = int(dt.timestamp())
            
            candles.append({
                "time": ts,
                "open": float(row.get("Open", row.get("open", 0))),
                "high": float(row.get("High", row.get("high", 0))),
                "low": float(row.get("Low", row.get("low", 0))),
                "close": float(row.get("Close", row.get("close", 0))),
                "volume": float(row.get("Volume", row.get("volume", 0))),
            })
        return candles
    except Exception as e:
        logger.error(f"[vn30] failed to load ohlcv from CSV for {symbol}: {e}")
        return []


# ─── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/tickers")
def get_vn30_tickers():
    """Return the list of VN30 constituent tickers."""
    return {"tickers": VN30_TICKERS, "count": len(VN30_TICKERS)}


@router.get("/quotes")
def get_vn30_quotes():
    """
    Return latest EOD quotes for all VN30 stocks.
    Uses local offline dataset if OFFLINE_MODE is True or if FinLens API calls fail.
    """
    if OFFLINE_MODE:
        results = _generate_quotes_from_dataset()
        if results:
            _save_cache("vn30_quotes.json", results)
        return results

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
                # Try to load from CSV dataset as fallback
                csv_quotes = _generate_quotes_from_dataset()
                sym_cache = next((q for q in csv_quotes if q["symbol"] == symbol), None) if csv_quotes else None
                if sym_cache:
                    results.append(sym_cache)
                else:
                    results.append({"symbol": symbol, "close": 0.0, "change_pct": 0.0})

        if results:
            _save_cache("vn30_quotes.json", results)
        return results

    except Exception as e:
        logger.error(f"[vn30] API quotes fetch failed: {e}. Falling back to CSV dataset.")
        results = _generate_quotes_from_dataset()
        if results:
            return results
        raise HTTPException(status_code=503, detail="Market data unavailable")


@router.get("/ohlcv/{symbol}")
def get_stock_ohlcv(
    symbol: str,
    timeframe: str = Query(default="1m", description="One of: 1d, 5d, 1m, 3m, 1y, 5y"),
):
    """
    Return OHLCV candlestick data for a given symbol and timeframe.
    Tries API first, falls back to CSV dataset if offline or fails.
    """
    symbol = symbol.upper()
    tf = timeframe.lower()
    cache_file = f"ohlcv_{symbol}_{tf}.json"

    if symbol not in VN30_TICKERS:
        raise HTTPException(status_code=404, detail=f"'{symbol}' is not in VN30 index")

    if OFFLINE_MODE:
        candles = _get_ohlcv_from_dataset(symbol, tf)
        if candles:
            _save_cache(cache_file, candles)
        return candles

    try:
        start_str, end_str = _date_range_for_timeframe(tf)
        cli = _get_client()

        if tf in ("1d", "5d"):
            df = cli.intraday.stock.ohlcv(symbol=symbol, start=start_str, end=end_str, interval="15m")
        else:
            df = cli.eod.stock.ohlcv(symbol=symbol, start=start_str, end=end_str)

        if df is None or len(df) == 0:
            candles = _get_ohlcv_from_dataset(symbol, tf)
            return candles if candles else []

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
        logger.error(f"[vn30] ohlcv fetch failed for {symbol}/{tf}: {e}. Falling back to CSV dataset.")
        candles = _get_ohlcv_from_dataset(symbol, tf)
        if candles:
            return candles
        raise HTTPException(status_code=503, detail=f"OHLCV data for {symbol} unavailable")


@router.get("/analysis/{symbol}")
def get_stock_analysis(symbol: str):
    """
    Calculate and return all possible stock analysis indicators 
    for a given symbol using the 5-year daily CSV data.
    """
    symbol = symbol.upper()
    if symbol not in VN30_TICKERS:
        raise HTTPException(status_code=404, detail=f"'{symbol}' is not in VN30 index")

    # Resolve CSV file path
    csv_path = Path("data/vn30_historical_csv") / f"{symbol}_5y_daily.csv"
    if not csv_path.exists():
        csv_path = Path("backend/data/vn30_historical_csv") / f"{symbol}_5y_daily.csv"

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Historical CSV for {symbol} not found. Run pre-fetch or export first.")

    try:
        import numpy as np
        df = pd.read_csv(csv_path)
        if df.empty or len(df) < 50:
            raise HTTPException(status_code=400, detail=f"Not enough historical data for {symbol} to calculate indicators")

        # Ensure sorted by date
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        from src.utils import stock_analysis as sa

        # Calculate Technical Indicators
        close_series = df["Close"]
        high_series = df["High"]
        low_series = df["Low"]
        volume_series = df["Volume"]

        sma20 = sa.sma(close_series, 20)
        sma50 = sa.sma(close_series, 50)
        sma200 = sa.sma(close_series, 200)
        ema12 = sa.ema(close_series, 12)
        ema26 = sa.ema(close_series, 26)
        rsi14 = sa.rsi(close_series, 14)
        macd_df = sa.macd(close_series)
        bb_df = sa.bollinger_bands(close_series)
        stoch_df = sa.stochastic_oscillator(high_series, low_series, close_series)
        obv_series = sa.obv(close_series, volume_series)
        adx_df = sa.adx(high_series, low_series, close_series)
        fib_levels = sa.fibonacci_retracement(high_series.tail(60), low_series.tail(60))

        # Calculate Risk Indicators
        mdd_res = sa.max_drawdown(close_series)
        sharpe = sa.sharpe_ratio(close_series)
        var_val = sa.value_at_risk(close_series, confidence=0.95)
        liq_df = sa.market_liquidity(close_series, volume_series)

        def _to_float_or_none(val):
            if pd.isna(val) or (isinstance(val, (int, float)) and np.isnan(val)):
                return None
            try:
                return float(val)
            except Exception:
                return None

        # Format latest values for response
        res = {
            "symbol": symbol,
            "latest_date": str(df["Date"].iloc[-1])[:10],
            "price": float(close_series.iloc[-1]),
            "technical": {
                "sma20": _to_float_or_none(sma20.iloc[-1]),
                "sma50": _to_float_or_none(sma50.iloc[-1]),
                "sma200": _to_float_or_none(sma200.iloc[-1]),
                "ema12": _to_float_or_none(ema12.iloc[-1]),
                "ema26": _to_float_or_none(ema26.iloc[-1]),
                "rsi14": _to_float_or_none(rsi14.iloc[-1]),
                "macd": {
                    "line": _to_float_or_none(macd_df["macd_line"].iloc[-1]),
                    "signal": _to_float_or_none(macd_df["signal_line"].iloc[-1]),
                    "histogram": _to_float_or_none(macd_df["histogram"].iloc[-1]),
                },
                "bollinger": {
                    "upper": _to_float_or_none(bb_df["upper"].iloc[-1]),
                    "middle": _to_float_or_none(bb_df["middle"].iloc[-1]),
                    "lower": _to_float_or_none(bb_df["lower"].iloc[-1]),
                },
                "stochastic": {
                    "k": _to_float_or_none(stoch_df["%K"].iloc[-1]),
                    "d": _to_float_or_none(stoch_df["%D"].iloc[-1]),
                },
                "obv": _to_float_or_none(obv_series.iloc[-1]),
                "adx": {
                    "plus_di": _to_float_or_none(adx_df["+DI"].iloc[-1]),
                    "minus_di": _to_float_or_none(adx_df["-DI"].iloc[-1]),
                    "adx": _to_float_or_none(adx_df["ADX"].iloc[-1]),
                },
                "fibonacci": fib_levels,
            },
            "risk": {
                "max_drawdown": {
                    "pct": _to_float_or_none(mdd_res["max_drawdown_pct"]),
                    "peak": _to_float_or_none(mdd_res["peak_value"]),
                    "trough": _to_float_or_none(mdd_res["trough_value"]),
                },
                "sharpe_ratio": _to_float_or_none(sharpe),
                "var_95": _to_float_or_none(var_val),
                "liquidity": {
                    "daily": _to_float_or_none(liq_df["daily_value"].iloc[-1]),
                    "avg20": _to_float_or_none(liq_df["avg_value"].iloc[-1]),
                }
            }
        }
        return res
    except Exception as e:
        logger.error(f"[vn30] failed to calculate analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate indicators: {e}")

def pre_fetch_market_data():
    """Background task to pre-fetch and cache VN30 quotes and export CSVs on startup."""
    logger.info("[vn30] starting background pre-fetch and CSV export...")
    try:
        # 1. Fetch quotes (from dataset if offline, from API if online)
        get_vn30_quotes()
        
        # 2. Export 5-year historical CSVs only if NOT offline
        if not OFFLINE_MODE:
            from src.utils.market_exporter import export_vn30_historical_csv
            export_vn30_historical_csv()
        else:
            logger.info("[vn30] Running in OFFLINE mode. Using local pre-downloaded CSV dataset.")
        
        logger.info("[vn30] background pre-fetch and CSV export complete.")
    except Exception as e:
        logger.warning(f"[vn30] background pre-fetch failed: {e}")

