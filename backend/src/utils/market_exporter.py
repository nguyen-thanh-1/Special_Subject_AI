import os
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from src.routers.vn30 import VN30_TICKERS, _get_client, _date_range_for_timeframe
from src.utils.logger import logger

def export_vn30_historical_csv():
    """
    Fetches 5-year historical daily data for all VN30 stocks 
    and saves each to a separate CSV file.
    """
    output_dir = Path("data/vn30_historical_csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting export of 5-year historical data for {len(VN30_TICKERS)} stocks...")
    
    try:
        cli = _get_client()
    except Exception as e:
        logger.error(f"Cannot initialize Finlens client: {e}")
        return

    start_str, end_str = _date_range_for_timeframe("5y")
    
    for symbol in VN30_TICKERS:
        try:
            logger.info(f"Exporting {symbol}...")
            # Fetch daily (EOD) data
            df = cli.eod.stock.ohlcv(symbol=symbol, start=start_str, end=end_str)
            
            if df is None or df.empty:
                logger.warning(f"No data found for {symbol}")
                continue
                
            # Normalize column names for consistency
            df.columns = [c.capitalize() for c in df.columns]
            if "Date" not in df.columns:
                df.reset_index(inplace=True)
                if "index" in df.columns:
                    df.rename(columns={"index": "Date"}, inplace=True)
            
            # Sort by date
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            
            # Save to CSV
            file_path = output_dir / f"{symbol}_5y_daily.csv"
            df.to_csv(file_path, index=False, encoding="utf-8")
            logger.info(f"Saved {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export {symbol}: {e}")

if __name__ == "__main__":
    # If run directly as a script
    import sys
    # Add src to path if needed
    sys.path.append(os.path.join(os.getcwd(), "src"))
    export_vn30_historical_csv()
