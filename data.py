from __future__ import annotations

import yfinance as yf
import pandas as pd
import streamlit as st

from settings import cfg


@st.cache_data(ttl=cfg["data"]["cache_ttl"])
def load_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = (
        raw["Close"]
        if isinstance(raw["Close"], pd.DataFrame)
        else raw["Close"].to_frame(tickers[0])
    )
    return prices.dropna().pct_change().dropna()


@st.cache_data(ttl=cfg["data"]["cache_ttl"])
def fetch_euribor_3m() -> float | None:
    try:
        raw = yf.download(
            cfg["data"]["euribor_ticker"],
            period="5d",
            progress=False,
            auto_adjust=True,
        )
        return float(raw["Close"].dropna().iloc[-1]) / 100
    except Exception:
        return None
