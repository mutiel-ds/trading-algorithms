import pandas as pd
import yfinance as yf

from typing import Optional
from datetime import date, datetime

from utils.logging import get_logger, Logger
from utils.exceptions import DataNotFoundError

logger: Logger = get_logger()

def download_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame:
    """
    Downloads stock data from Yahoo Finance API.

    Args:
        symbol: The stock symbol. Example: 'AAPL' (i.e., Apple)
        start_date: The first date (in YYYY-MM-DD format) from which to download data. Example: '2020-01-01'
        end_date: The last date (in YYYY-MM-DD format) from which to download data. Example: '2025-01-01'
        interval: The interval between consecutive stock data points. Example: '1D' (i.e., 1 day)

    Returns:
        The stock data, if found, as a Pandas DataFrame.
    """
    start_dt: date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt: date = datetime.strptime(end_date, "%Y-%m-%d").date()

    logger.info(msg=f"Starting to download stock data for '{symbol}'!")
    stock_data: Optional[pd.DataFrame] = yf.download(
        tickers=symbol,
        start=start_dt,
        end=end_dt,
        interval=interval,
        auto_adjust=True
    )

    if (stock_data is None) or (stock_data.empty):
        _msg: str = f"No stock data found for '{symbol}' with the provided arguments. Please try again."
        logger.error(msg=_msg)
        raise DataNotFoundError(_msg)

    logger.info(msg=f"Successfully downloaded {len(stock_data)} rows of stock data for '{symbol}'.")
    return stock_data