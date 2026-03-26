from __future__ import annotations
import pandas as pd
import yfinance as yf


def download_price_data(
    tickers: list,
    start_date: str,
    end_date = None,
):
    """Download ticker price series"""

    unique_tickers = sorted(set(tickers))

    raw = yf.download(
        tickers=unique_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for tickers={unique_tickers}, "
            f"start={start_date}, end={end_date}."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = unique_tickers

    available = [t for t in unique_tickers if t in prices.columns]
    prices = prices[available]

    # Clean index
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]

    prices = prices.dropna(how="all")

    return prices


def prepare_price_panel(
    portfolio_tickers: list,
    hedge_ticker: str,
    start_date: str,
    end_date = None,
):
    """Format price series"""

    all_tickers = list(portfolio_tickers) + [hedge_ticker]
    prices = download_price_data(all_tickers, start_date, end_date)

    # Verify ticker exists
    if hedge_ticker not in prices.columns:
        raise ValueError(
            f"Hedge ticker '{hedge_ticker}' could not be downloaded. "
            "Check network or ticker validity."
        )

    clean = prices.dropna(how="any")
    if len(clean) < 0.60 * len(prices):
        filled = prices.ffill(limit=2)
        clean = filled.dropna(how="any")

    if clean.empty:
        raise ValueError(
            "Price panel is empty after alignment. "
            "Check tickers and date range."
        )

    ordered_portfolio = [t for t in portfolio_tickers if t in clean.columns]
    col_order = ordered_portfolio + [hedge_ticker]
    clean = clean[col_order]

    return clean
