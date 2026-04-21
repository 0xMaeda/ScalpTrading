import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


APP_TITLE = "Quick Flip Scalper — Scanner & Charting"
DEFAULT_WATCHLIST = "SPY, QQQ, NVDA, AAPL, TSLA, AMD, META, MSFT"
DEFAULT_MAX_SCAN = 250
DEFAULT_HOT_CANDIDATES = 300
DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_MIN_PRICE = 2.0
DEFAULT_MIN_REL_VOLUME = 0.5


@dataclass
class StrategyConfig:
    session_open_hour: int = 9
    session_open_minute: int = 30
    opening_range_minutes: int = 15
    reversal_window_minutes: int = 90
    atr_period: int = 14
    atr_threshold_pct: float = 0.25
    wick_ratio_min: float = 2.0
    body_ratio_max: float = 0.45


@dataclass
class SignalResult:
    symbol: str
    trade_date: pd.Timestamp
    open15_direction: str
    box_high: float
    box_low: float
    box_range: float
    atr14: float
    liquidity_threshold: float
    is_liquidity: bool
    signal_type: Optional[str]
    signal_side: Optional[str]
    signal_time: Optional[pd.Timestamp]
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    rr: Optional[float]
    notes: str


def parse_session_open(session_open: str) -> Tuple[int, int]:
    try:
        hour, minute = [int(x) for x in session_open.split(":")]
        return hour, minute
    except Exception:
        return 9, 30


@st.cache_data(ttl=60 * 5, show_spinner=False)
def fetch_yahoo_screen(url: str) -> pd.DataFrame:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        response = requests.get(url, headers=headers, timeout=12)
        if response.status_code != 200 or not response.text:
            return pd.DataFrame()
        tables = pd.read_html(response.text)
        if not tables:
            return pd.DataFrame()
        df = tables[0].copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_hot_universe() -> pd.DataFrame:
    sources = {
        "most_active": "https://finance.yahoo.com/markets/stocks/most-active/",
        "gainers": "https://finance.yahoo.com/markets/stocks/gainers/",
        "losers": "https://finance.yahoo.com/markets/stocks/losers/",
    }

    frames = []
    for source_name, url in sources.items():
        df = fetch_yahoo_screen(url)
        if df.empty:
            continue

        rename_map = {}
        for col in df.columns:
            lc = str(col).lower()
            if lc == "symbol":
                rename_map[col] = "Symbol"
            elif lc in {"name", "company", "company name"}:
                rename_map[col] = "Name"
            elif "% change" in lc or "change %" in lc:
                rename_map[col] = "PctChange"
            elif lc == "volume":
                rename_map[col] = "Volume"
            elif "avg vol" in lc:
                rename_map[col] = "AvgVolume"
            elif lc in {"price", "last price"}:
                rename_map[col] = "Price"
            elif lc == "market cap":
                rename_map[col] = "MarketCap"
        df = df.rename(columns=rename_map)

        if "Symbol" not in df.columns:
            continue

        keep_cols = [c for c in ["Symbol", "Name", "Price", "PctChange", "Volume", "AvgVolume", "MarketCap"] if c in df.columns]
        df = df[keep_cols].copy()
        df["Source"] = source_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged["Symbol"] = merged["Symbol"].astype(str).str.upper().str.strip()
    merged = merged[merged["Symbol"].ne("")].copy()

    for col in ["Price", "PctChange", "Volume", "AvgVolume"]:
        if col in merged.columns:
            merged[col] = (
                merged[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("+", "", regex=False)
                .replace({"-": np.nan, "": np.nan})
            )
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    agg_map = {c: "first" for c in merged.columns if c not in {"Symbol", "PctChange", "Volume", "AvgVolume", "Source"}}
    if "PctChange" in merged.columns:
        agg_map["PctChange"] = "max"
    if "Volume" in merged.columns:
        agg_map["Volume"] = "max"
    if "AvgVolume" in merged.columns:
        agg_map["AvgVolume"] = "max"
    agg_map["Source"] = lambda s: ",".join(sorted(set([str(x) for x in s if pd.notna(x)])))

    merged = merged.groupby("Symbol", as_index=False).agg(agg_map)

    for col in ["PctChange", "Volume", "AvgVolume", "Price"]:
        if col not in merged.columns:
            merged[col] = np.nan

    merged["AbsPctChange"] = merged["PctChange"].abs()
    merged["RelVolume"] = merged["Volume"] / merged["AvgVolume"]
    merged.loc[~np.isfinite(merged["RelVolume"]), "RelVolume"] = np.nan

    for col in ["AbsPctChange", "Volume", "RelVolume"]:
        merged[f"{col}Rank"] = merged[col].fillna(-1).rank(pct=True, method="average")

    merged["HotScore"] = (
        merged["AbsPctChangeRank"] * 0.45
        + merged["VolumeRank"] * 0.35
        + merged["RelVolumeRank"] * 0.20
    )

    merged = merged.sort_values(["HotScore", "Volume", "AbsPctChange"], ascending=[False, False, False]).reset_index(drop=True)
    return merged


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_exchange_universe(universe_name: str) -> List[str]:
    if universe_name == "MANUAL":
        return []

    symbols: List[str] = []
    try:
        if universe_name in {"NASDAQ", "NASDAQ+NYSE"}:
            nasdaq_df = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|")
            if "Symbol" in nasdaq_df.columns:
                s = nasdaq_df["Symbol"].astype(str)
                s = s[~s.str.contains("File Creation Time", na=False)]
                s = s[~s.str.contains("\$", na=False)]
                s = s[~s.str.contains("\^", na=False)]
                symbols.extend(s.tolist())

        if universe_name in {"NYSE", "NASDAQ+NYSE"}:
            other_df = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|")
            if {"ACT Symbol", "Exchange"}.issubset(other_df.columns):
                nyse_df = other_df[other_df["Exchange"].astype(str).str.upper() == "N"].copy()
                s = nyse_df["ACT Symbol"].astype(str)
                s = s[~s.str.contains("File Creation Time", na=False)]
                s = s[~s.str.contains("\$", na=False)]
                s = s[~s.str.contains("\^", na=False)]
                symbols.extend(s.tolist())
    except Exception:
        return []

    cleaned = []
    seen = set()
    for sym in symbols:
        sym = sym.strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        cleaned.append(sym)
    return cleaned


@st.cache_data(ttl=60 * 15, show_spinner=False)
def download_intraday(symbol: str, period: str = "10d", interval: str = "5m") -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        prepost=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    return df.dropna()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_daily(symbol: str, period: str = "3mo") -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
        prepost=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index)
    return df.dropna()


def compute_atr(df_daily: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df_daily["High"]
    low = df_daily["Low"]
    close = df_daily["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def is_green(row: pd.Series) -> bool:
    return float(row["Close"]) >= float(row["Open"])


def candle_body(row: pd.Series) -> float:
    return abs(float(row["Close"]) - float(row["Open"]))


def candle_range(row: pd.Series) -> float:
    return float(row["High"]) - float(row["Low"])


def upper_wick(row: pd.Series) -> float:
    return float(row["High"]) - max(float(row["Open"]), float(row["Close"]))


def lower_wick(row: pd.Series) -> float:
    return min(float(row["Open"]), float(row["Close"])) - float(row["Low"])


def is_hammer(row: pd.Series, cfg: StrategyConfig) -> bool:
    rng = candle_range(row)
    if rng <= 0:
        return False
    body = candle_body(row)
    lw = lower_wick(row)
    uw = upper_wick(row)
    return lw >= body * cfg.wick_ratio_min and body <= rng * cfg.body_ratio_max and uw <= body


def is_inverted_hammer(row: pd.Series, cfg: StrategyConfig) -> bool:
    rng = candle_range(row)
    if rng <= 0:
        return False
    body = candle_body(row)
    lw = lower_wick(row)
    uw = upper_wick(row)
    return uw >= body * cfg.wick_ratio_min and body <= rng * cfg.body_ratio_max and lw <= body


def is_bullish_engulfing(prev_row: pd.Series, row: pd.Series) -> bool:
    prev_open, prev_close = float(prev_row["Open"]), float(prev_row["Close"])
    cur_open, cur_close = float(row["Open"]), float(row["Close"])
    return (prev_close < prev_open) and (cur_close > cur_open) and (cur_open <= prev_close) and (cur_close >= prev_open)


def is_bearish_engulfing(prev_row: pd.Series, row: pd.Series) -> bool:
    prev_open, prev_close = float(prev_row["Open"]), float(prev_row["Close"])
    cur_open, cur_close = float(row["Open"]), float(row["Close"])
    return (prev_close > prev_open) and (cur_close < cur_open) and (cur_open >= prev_close) and (cur_close <= prev_open)


def normalize_timezone_index(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC").tz_convert(tz_name)
    else:
        out.index = out.index.tz_convert(tz_name)
    return out


def session_bounds(trade_date: pd.Timestamp, cfg: StrategyConfig, tz_name: str) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    d = pd.Timestamp(trade_date).tz_localize(None)
    open_dt = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=cfg.session_open_hour, minute=cfg.session_open_minute).tz_localize(tz_name)
    first15_end = open_dt + pd.Timedelta(minutes=cfg.opening_range_minutes)
    reversal_end = open_dt + pd.Timedelta(minutes=cfg.reversal_window_minutes)
    return open_dt, first15_end, reversal_end


def get_recent_trade_dates(df_intraday_tz: pd.DataFrame) -> List[pd.Timestamp]:
    return list(pd.Index(pd.to_datetime(df_intraday_tz.index.date)).unique())


def analyze_day(symbol: str, df_5m_tz: pd.DataFrame, df_daily_tz: pd.DataFrame, trade_date: pd.Timestamp, cfg: StrategyConfig, tz_name: str) -> Optional[SignalResult]:
    open_dt, first15_end, reversal_end = session_bounds(trade_date, cfg, tz_name)
    first15 = df_5m_tz[(df_5m_tz.index >= open_dt) & (df_5m_tz.index < first15_end)]
    if len(first15) < 3:
        return None

    open15 = pd.Series({
        "Open": float(first15.iloc[0]["Open"]),
        "High": float(first15["High"].max()),
        "Low": float(first15["Low"].min()),
        "Close": float(first15.iloc[-1]["Close"]),
        "Volume": float(first15["Volume"].sum()) if "Volume" in first15.columns else np.nan,
    })

    box_high = float(open15["High"])
    box_low = float(open15["Low"])
    box_range = box_high - box_low
    direction = "green" if is_green(open15) else "red"

    daily_for_atr = df_daily_tz.copy()
    daily_for_atr["ATR14"] = compute_atr(daily_for_atr, cfg.atr_period)
    daily_idx = pd.Timestamp(trade_date).tz_localize(None)
    atr_row = daily_for_atr[daily_for_atr.index.tz_localize(None) <= daily_idx]
    if atr_row.empty or pd.isna(atr_row["ATR14"].iloc[-1]):
        return None

    atr14 = float(atr_row["ATR14"].iloc[-1])
    liquidity_threshold = atr14 * cfg.atr_threshold_pct
    is_liquidity = box_range >= liquidity_threshold
    notes: List[str] = []

    if not is_liquidity:
        notes.append(f"Opening range {box_range:.2f} is below threshold {liquidity_threshold:.2f}.")
        return SignalResult(symbol, pd.Timestamp(trade_date), direction, box_high, box_low, box_range, atr14, liquidity_threshold, False, None, None, None, None, None, None, None, " ".join(notes))

    after_first15 = df_5m_tz[(df_5m_tz.index >= first15_end) & (df_5m_tz.index < reversal_end)].copy()
    if len(after_first15) < 2:
        notes.append("No 5-minute data after the opening range within the first 90 minutes.")
        return SignalResult(symbol, pd.Timestamp(trade_date), direction, box_high, box_low, box_range, atr14, liquidity_threshold, True, None, None, None, None, None, None, None, " ".join(notes))

    rows = after_first15.reset_index().rename(columns={"index": "Datetime"})
    signal_type = None
    signal_side = None
    signal_time = None
    entry = stop = target = rr = None

    for i in range(1, len(rows)):
        prev_row = rows.iloc[i - 1]
        row = rows.iloc[i]
        ts = row["Datetime"]
        outside_above = float(row["High"]) > box_high
        outside_below = float(row["Low"]) < box_low

        if direction == "green" and outside_above:
            if is_inverted_hammer(row, cfg):
                signal_type = "Inverted Hammer"
                signal_side = "Short"
                signal_time = ts
                entry = float(row["Low"])
                stop = float(row["High"])
                target = box_low
                risk = stop - entry
                reward = entry - target
                rr = reward / risk if risk > 0 else None
                break
            if is_bearish_engulfing(prev_row, row):
                signal_type = "Bearish Engulfing"
                signal_side = "Short"
                signal_time = ts
                entry = float(prev_row["Low"])
                stop = float(row["High"])
                target = box_low
                risk = stop - entry
                reward = entry - target
                rr = reward / risk if risk > 0 else None
                break

        if direction == "red" and outside_below:
            if is_hammer(row, cfg):
                signal_type = "Hammer"
                signal_side = "Long"
                signal_time = ts
                entry = float(row["High"])
                stop = float(row["Low"])
                target = box_high
                risk = entry - stop
                reward = target - entry
                rr = reward / risk if risk > 0 else None
                break
            if is_bullish_engulfing(prev_row, row):
                signal_type = "Bullish Engulfing"
                signal_side = "Long"
                signal_time = ts
                entry = float(prev_row["High"])
                stop = float(row["Low"])
                target = box_high
                risk = entry - stop
                reward = target - entry
                rr = reward / risk if risk > 0 else None
                break

    if signal_type is None:
        notes.append("Liquidity candle found, but no valid reversal candle appeared outside the box within the first 90 minutes.")

    return SignalResult(symbol, pd.Timestamp(trade_date), direction, box_high, box_low, box_range, atr14, liquidity_threshold, True, signal_type, signal_side, signal_time, entry, stop, target, rr, " ".join(notes))


def prefilter_symbol(symbol: str, cfg: StrategyConfig, tz_name: str) -> Dict:
    symbol = symbol.strip().upper()
    if not symbol:
        return {"Symbol": symbol, "Status": "Skipped"}
    try:
        df_5m = download_intraday(symbol)
        df_daily = download_daily(symbol)
        if df_5m.empty or df_daily.empty:
            return {"Symbol": symbol, "Status": "No data"}

        df_5m = normalize_timezone_index(df_5m, tz_name)
        df_daily = normalize_timezone_index(df_daily, tz_name)
        dates = get_recent_trade_dates(df_5m)
        if not dates:
            return {"Symbol": symbol, "Status": "No trade dates found"}

        trade_date = pd.Timestamp(dates[-1])
        open_dt, first15_end, _ = session_bounds(trade_date, cfg, tz_name)
        first15 = df_5m[(df_5m.index >= open_dt) & (df_5m.index < first15_end)]
        if len(first15) < 3:
            return {"Symbol": symbol, "Status": "Insufficient data"}

        box_high = float(first15["High"].max())
        box_low = float(first15["Low"].min())
        box_range = box_high - box_low

        daily_for_atr = df_daily.copy()
        daily_for_atr["ATR14"] = compute_atr(daily_for_atr, cfg.atr_period)
        daily_idx = pd.Timestamp(trade_date).tz_localize(None)
        atr_row = daily_for_atr[daily_for_atr.index.tz_localize(None) <= daily_idx]
        if atr_row.empty or pd.isna(atr_row["ATR14"].iloc[-1]):
            return {"Symbol": symbol, "Status": "Insufficient ATR"}

        atr14 = float(atr_row["ATR14"].iloc[-1])
        liquidity_threshold = atr14 * cfg.atr_threshold_pct
        is_liquidity = box_range >= liquidity_threshold
        open15_direction = "green" if float(first15.iloc[-1]["Close"]) >= float(first15.iloc[0]["Open"]) else "red"

        return {
            "Symbol": symbol,
            "Trade Date": trade_date.date(),
            "Status": "Liquidity candidate" if is_liquidity else "No liquidity",
            "Open15 Dir": open15_direction,
            "Box High": round(box_high, 4),
            "Box Low": round(box_low, 4),
            "Range": round(box_range, 4),
            "ATR14": round(atr14, 4),
            "Threshold": round(liquidity_threshold, 4),
            "_is_liquidity": is_liquidity,
        }
    except Exception as exc:
        return {"Symbol": symbol, "Status": f"Error: {exc}"}


def analyze_candidate(symbol: str, cfg: StrategyConfig, tz_name: str) -> Tuple[Dict, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    symbol = symbol.strip().upper()
    try:
        df_5m = download_intraday(symbol)
        df_daily = download_daily(symbol)
        if df_5m.empty or df_daily.empty:
            return {"Symbol": symbol, "Status": "No data"}, None, None

        df_5m = normalize_timezone_index(df_5m, tz_name)
        df_daily = normalize_timezone_index(df_daily, tz_name)
        dates = get_recent_trade_dates(df_5m)
        if not dates:
            return {"Symbol": symbol, "Status": "No trade dates found"}, None, None

        trade_date = pd.Timestamp(dates[-1])
        result = analyze_day(symbol, df_5m, df_daily, trade_date, cfg, tz_name)
        if result is None:
            return {"Symbol": symbol, "Status": "Insufficient data"}, None, None

        status = "Signal" if result.signal_type else ("Liquidity only" if result.is_liquidity else "No liquidity")
        row = {
            "Symbol": result.symbol,
            "Trade Date": result.trade_date.date(),
            "Status": status,
            "Open15 Dir": result.open15_direction,
            "Box High": round(result.box_high, 4),
            "Box Low": round(result.box_low, 4),
            "Range": round(result.box_range, 4),
            "ATR14": round(result.atr14, 4),
            "Threshold": round(result.liquidity_threshold, 4),
            "Signal": result.signal_type or "",
            "Side": result.signal_side or "",
            "Signal Time": result.signal_time.strftime("%Y-%m-%d %H:%M") if result.signal_time is not None else "",
            "Entry": round(result.entry, 4) if result.entry is not None else np.nan,
            "Stop": round(result.stop, 4) if result.stop is not None else np.nan,
            "Target": round(result.target, 4) if result.target is not None else np.nan,
            "R/R": round(result.rr, 2) if result.rr is not None else np.nan,
            "Notes": result.notes,
        }
        return row, df_5m, df_daily
    except Exception as exc:
        return {"Symbol": symbol, "Status": f"Error: {exc}"}, None, None


def build_scanner(symbols: List[str], cfg: StrategyConfig, tz_name: str, keep_only_matches: bool, max_workers: int) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    rows: List[Dict] = []
    intraday_map: Dict[str, pd.DataFrame] = {}
    daily_map: Dict[str, pd.DataFrame] = {}
    candidates: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(prefilter_symbol, symbol, cfg, tz_name): symbol for symbol in symbols}
        for fut in as_completed(futures):
            row = fut.result()
            if row.get("_is_liquidity"):
                candidates.append(row["Symbol"])
            elif not keep_only_matches:
                rows.append({k: v for k, v in row.items() if not str(k).startswith("_")})

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_candidate, symbol, cfg, tz_name): symbol for symbol in candidates}
        for fut in as_completed(futures):
            row, df_5m, df_daily = fut.result()
            status = row.get("Status", "")
            if (not keep_only_matches) or status == "Signal":
                if df_5m is not None and df_daily is not None:
                    intraday_map[row["Symbol"]] = df_5m
                    daily_map[row["Symbol"]] = df_daily
            rows.append(row)

    result_df = pd.DataFrame(rows)
    return result_df, intraday_map, daily_map


def apply_ranking(scanner_df: pd.DataFrame, hot_df: pd.DataFrame, scanned_symbols: Optional[List[str]] = None, universe_choice: str = "MANUAL") -> pd.DataFrame:
    if scanner_df.empty:
        return scanner_df

    if scanned_symbols is None:
        scanned_symbols = []

    input_rank_df = pd.DataFrame({
        "Symbol": [str(s).upper() for s in scanned_symbols],
        "InputRank": np.arange(1, len(scanned_symbols) + 1),
    }).drop_duplicates(subset=["Symbol"], keep="first")
    if not input_rank_df.empty:
        scanner_df = scanner_df.merge(input_rank_df, on="Symbol", how="left")

    if not hot_df.empty:
        hot_rank_df = hot_df.reset_index(drop=True).copy()
        hot_rank_df["HotRank"] = np.arange(1, len(hot_rank_df) + 1)
        merge_cols = [c for c in ["Symbol", "HotRank", "HotScore", "Volume", "AvgVolume", "RelVolume", "PctChange", "Price", "Source", "Name"] if c in hot_rank_df.columns]
        scanner_df = scanner_df.merge(hot_rank_df[merge_cols], on="Symbol", how="left")

    if "InputRank" not in scanner_df.columns:
        scanner_df["InputRank"] = np.nan
    if "HotRank" not in scanner_df.columns:
        scanner_df["HotRank"] = np.nan
    if "HotScore" not in scanner_df.columns:
        scanner_df["HotScore"] = np.nan
    if "R/R" not in scanner_df.columns:
        scanner_df["R/R"] = np.nan

    scanner_df["_inputrank_sort"] = scanner_df["InputRank"].fillna(10**9)
    scanner_df["_hotrank_sort"] = scanner_df["HotRank"].fillna(10**9)

    if universe_choice == "HOT":
        scanner_df = (
            scanner_df
            .sort_values(["_inputrank_sort", "_hotrank_sort"], ascending=[True, True], kind="stable")
            .drop(columns=["_inputrank_sort", "_hotrank_sort"])
            .reset_index(drop=True)
        )
        return scanner_df

    if "Status" in scanner_df.columns:
        status_order = {"Signal": 0, "Liquidity only": 1, "Liquidity candidate": 2, "No liquidity": 3}
        scanner_df["_status_sort"] = scanner_df["Status"].map(status_order).fillna(9)
    else:
        scanner_df["_status_sort"] = 9

    scanner_df = (
        scanner_df
        .sort_values(["_status_sort", "_inputrank_sort", "_hotrank_sort", "HotScore", "R/R"], ascending=[True, True, True, False, False], kind="stable")
        .drop(columns=["_status_sort", "_inputrank_sort", "_hotrank_sort"])
        .reset_index(drop=True)
    )
    return scanner_df


def plot_symbol_chart(symbol: str, df_5m_tz: pd.DataFrame, df_daily_tz: pd.DataFrame, trade_date: pd.Timestamp, cfg: StrategyConfig, tz_name: str) -> Tuple[go.Figure, Optional[SignalResult]]:
    result = analyze_day(symbol, df_5m_tz, df_daily_tz, trade_date, cfg, tz_name)
    if result is None:
        fig = go.Figure()
        fig.update_layout(title=f"{symbol} — insufficient data")
        return fig, None

    open_dt, first15_end, reversal_end = session_bounds(trade_date, cfg, tz_name)
    chart_df = df_5m_tz[(df_5m_tz.index >= open_dt - pd.Timedelta(minutes=15)) & (df_5m_tz.index <= reversal_end + pd.Timedelta(minutes=30))].copy()

    fig = go.Figure(data=[go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"],
        high=chart_df["High"],
        low=chart_df["Low"],
        close=chart_df["Close"],
        name=symbol,
    )])

    fig.add_shape(
        type="rect",
        x0=first15_end,
        x1=reversal_end,
        y0=result.box_low,
        y1=result.box_high,
        line=dict(width=1),
        fillcolor="LightSkyBlue",
        opacity=0.18,
    )
    fig.add_hline(y=result.box_high, line_dash="dot", annotation_text="Box High")
    fig.add_hline(y=result.box_low, line_dash="dot", annotation_text="Box Low")

    if result.signal_time is not None and result.entry is not None:
        marker_symbol = "triangle-up" if result.signal_side == "Long" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[result.signal_time],
            y=[result.entry],
            mode="markers+text",
            text=[result.signal_type],
            textposition="top center",
            marker=dict(size=14, symbol=marker_symbol),
            name="Signal",
        ))
        fig.add_hline(y=result.entry, line_dash="solid", annotation_text="Entry")
        fig.add_hline(y=result.stop, line_dash="dash", annotation_text="Stop")
        fig.add_hline(y=result.target, line_dash="dash", annotation_text="Target")

    fig.update_layout(
        title=f"{symbol} — Quick Flip Scalper",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=720,
    )
    return fig, result


def highlight_status(val: str) -> str:
    if val == "Signal":
        return "background-color: #d1fae5"
    if val == "Liquidity only":
        return "background-color: #fef3c7"
    if val == "No liquidity":
        return "background-color: #fee2e2"
    return ""


def select_symbols(universe_choice: str, watchlist_text: str, max_scan: int, hot_candidates: int, min_price: float, min_rel_volume: float) -> Tuple[List[str], pd.DataFrame, Optional[str]]:
    hot_df = pd.DataFrame()

    if universe_choice == "MANUAL":
        symbols = [s.strip().upper() for s in watchlist_text.split(",") if s.strip()]
        return symbols, hot_df, None

    if universe_choice == "HOT":
        hot_df = fetch_hot_universe()
        if hot_df.empty:
            fallback = fetch_exchange_universe("NASDAQ")
            if fallback:
                fallback_symbols = ["SPY"] + [s for s in fallback if s != "SPY"]
                return fallback_symbols[: min(int(max_scan), 200)], pd.DataFrame(), None
            return [], hot_df, "Could not load the HOT universe right now."

        filtered_hot = hot_df.copy()
        if "Price" in filtered_hot.columns:
            filtered_hot = filtered_hot[filtered_hot["Price"].fillna(0) >= float(min_price)]
        if "RelVolume" in filtered_hot.columns:
            filtered_hot = filtered_hot[(filtered_hot["RelVolume"].fillna(0) >= float(min_rel_volume)) | (filtered_hot["RelVolume"].isna())]

        spy_row = pd.DataFrame([{"Symbol": "SPY", "Name": "SPDR S&P 500 ETF Trust", "HotScore": 9999.0, "Source": "manual_spy"}])
        if "SPY" in filtered_hot["Symbol"].astype(str).str.upper().tolist():
            filtered_hot = filtered_hot[filtered_hot["Symbol"].astype(str).str.upper() != "SPY"].copy()
        filtered_hot = pd.concat([spy_row, filtered_hot], ignore_index=True)
        filtered_hot["Symbol"] = filtered_hot["Symbol"].astype(str).str.upper().str.strip()
        filtered_hot = filtered_hot.drop_duplicates(subset=["Symbol"], keep="first")

        filtered_hot = filtered_hot.head(int(hot_candidates)).copy()
        if filtered_hot.empty:
            return [], hot_df, "HOT filtering removed all symbols. Lower the minimum price or relative volume filter."

        if len(filtered_hot) > int(max_scan):
            filtered_hot = filtered_hot.head(int(max_scan)).copy()

        symbols = filtered_hot["Symbol"].dropna().astype(str).str.upper().tolist()
        return symbols, filtered_hot, None

    symbols = fetch_exchange_universe(universe_choice)
    if not symbols:
        return [], hot_df, f"Could not load the selected exchange universe: {universe_choice}."
    return symbols[: int(max_scan)], hot_df, None


def render_symbol_buttons(symbols: List[str], key_prefix: str = "symbtn") -> None:
    if not symbols:
        return
    cols_per_row = 8
    for start in range(0, len(symbols), cols_per_row):
        row_symbols = symbols[start:start + cols_per_row]
        cols = st.columns(cols_per_row)
        for idx, sym in enumerate(row_symbols):
            if cols[idx].button(sym, key=f"{key_prefix}_{sym}_{start}_{idx}", use_container_width=True):
                st.session_state["selected_symbol"] = sym


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Quick Flip Scalper scanner with HOT-market ranking, two-stage filtering, and click-to-chart symbol selection."
    )

    with st.sidebar:
        st.header("Settings")
        universe_choice = st.selectbox("Universe", ["MANUAL", "HOT", "NASDAQ", "NYSE", "NASDAQ+NYSE"], index=1)
        keep_only_matches = st.checkbox("Keep only symbols with active signals", value=True)
        watchlist_text = st.text_area("Watchlist (comma-separated)", value=DEFAULT_WATCHLIST, height=120, disabled=(universe_choice != "MANUAL"))
        max_scan = st.number_input("Max symbols to scan this run", min_value=25, max_value=5000, value=DEFAULT_MAX_SCAN, step=25)
        hot_candidates = st.number_input("HOT candidate pool size", min_value=25, max_value=1000, value=DEFAULT_HOT_CANDIDATES, step=25)
        min_price = st.number_input("Minimum stock price", min_value=0.0, max_value=10000.0, value=DEFAULT_MIN_PRICE, step=0.5)
        min_rel_volume = st.number_input("Minimum relative volume", min_value=0.0, max_value=100.0, value=DEFAULT_MIN_REL_VOLUME, step=0.1)
        max_workers = st.slider("Parallel workers", min_value=1, max_value=24, value=8)
        tz_name = st.text_input("Session timezone", value=DEFAULT_TIMEZONE)
        session_open = st.text_input("Market open (HH:MM)", value="09:30")
        atr_threshold_pct = st.slider("Liquidity threshold (% of ATR14)", min_value=0.10, max_value=0.50, value=0.25, step=0.01)
        wick_ratio_min = st.slider("Hammer wick/body minimum", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
        body_ratio_max = st.slider("Max body as % of candle range", min_value=0.10, max_value=0.70, value=0.45, step=0.05)
        run_scan = st.button("Run scanner", type="primary")
        if run_scan:
            st.session_state["scanner_ran"] = True

    hour, minute = parse_session_open(session_open)
    cfg = StrategyConfig(
        session_open_hour=hour,
        session_open_minute=minute,
        atr_threshold_pct=float(atr_threshold_pct),
        wick_ratio_min=float(wick_ratio_min),
        body_ratio_max=float(body_ratio_max),
    )

    if "scanner_ran" not in st.session_state:
        st.session_state["scanner_ran"] = False

    if not st.session_state["scanner_ran"]:
        st.info("Adjust the settings in the sidebar and click Run scanner.")
        st.stop()

    with st.spinner("Loading selected universe..."):
        symbols, hot_df, selection_error = select_symbols(
            universe_choice,
            watchlist_text,
            int(max_scan),
            int(hot_candidates),
            float(min_price),
            float(min_rel_volume),
        )

    if selection_error:
        st.error(selection_error)
        st.stop()
    if not symbols:
        st.error("No symbols available to scan.")
        st.stop()

    scan_signature = {
        "universe_choice": universe_choice,
        "watchlist_text": watchlist_text,
        "max_scan": int(max_scan),
        "hot_candidates": int(hot_candidates),
        "min_price": float(min_price),
        "min_rel_volume": float(min_rel_volume),
        "keep_only_matches": bool(keep_only_matches),
        "max_workers": int(max_workers),
        "tz_name": tz_name,
        "session_open": session_open,
        "atr_threshold_pct": float(atr_threshold_pct),
        "wick_ratio_min": float(wick_ratio_min),
        "body_ratio_max": float(body_ratio_max),
    }

    should_scan = run_scan or (st.session_state.get("scan_signature") != scan_signature) or ("scanner_df" not in st.session_state)
    if should_scan:
        with st.spinner("Scanning symbols..."):
            scanner_df, intraday_map, daily_map = build_scanner(
                symbols=symbols,
                cfg=cfg,
                tz_name=tz_name,
                keep_only_matches=keep_only_matches,
                max_workers=int(max_workers),
            )
            scanner_df = apply_ranking(scanner_df, hot_df, scanned_symbols=symbols, universe_choice=universe_choice)

        st.session_state["scanner_df"] = scanner_df
        st.session_state["intraday_map"] = intraday_map
        st.session_state["daily_map"] = daily_map
        st.session_state["hot_df"] = hot_df
        st.session_state["scan_symbols"] = symbols
        st.session_state["scan_signature"] = scan_signature
    else:
        scanner_df = st.session_state.get("scanner_df", pd.DataFrame())
        intraday_map = st.session_state.get("intraday_map", {})
        daily_map = st.session_state.get("daily_map", {})
        hot_df = st.session_state.get("hot_df", pd.DataFrame())
        symbols = st.session_state.get("scan_symbols", symbols)

    st.subheader("Scanner results")
    if scanner_df.empty:
        st.warning("No results returned.")
        st.stop()

    signal_count = int((scanner_df["Status"] == "Signal").sum()) if "Status" in scanner_df.columns else 0
    liquidity_count = int(scanner_df["Status"].isin(["Signal", "Liquidity only", "Liquidity candidate"]).sum()) if "Status" in scanner_df.columns else 0

    if universe_choice == "HOT":
        a, b, c, d = st.columns(4)
        a.metric("Symbols scanned", len(symbols))
        b.metric("HOT candidates", len(symbols))
        c.metric("Liquidity candidates", liquidity_count)
        d.metric("Active signals", signal_count)
    else:
        a, b, c = st.columns(3)
        a.metric("Symbols scanned", len(symbols))
        b.metric("Liquidity candidates", liquidity_count)
        c.metric("Active signals", signal_count)

    preferred_cols = [
        "Symbol", "Trade Date", "Status", "InputRank", "HotRank", "HotScore", "PctChange", "Volume", "AvgVolume", "RelVolume", "Price",
        "Open15 Dir", "Box High", "Box Low", "Range", "ATR14", "Threshold",
        "Signal", "Side", "Signal Time", "Entry", "Stop", "Target", "R/R", "Source", "Notes"
    ]
    show_cols = [c for c in preferred_cols if c in scanner_df.columns] + [c for c in scanner_df.columns if c not in preferred_cols]
    scanner_show = scanner_df[show_cols].copy()

    st.caption("HOT mode preserves the exact ranked scan-list order. Click a symbol button below the grid to load its chart.")
    if "Status" in scanner_show.columns:
        st.dataframe(scanner_show.style.map(highlight_status, subset=["Status"]), use_container_width=True, hide_index=True)
    else:
        st.dataframe(scanner_show, use_container_width=True, hide_index=True)

    valid_symbols = [s for s in scanner_df.get("Symbol", pd.Series(dtype=str)).tolist() if s in intraday_map and s in daily_map]
    if keep_only_matches and not valid_symbols and not scanner_df.empty:
        st.info("No symbols produced a full signal on this run.")
        st.stop()
    if not valid_symbols:
        st.stop()

    render_symbol_buttons(valid_symbols[:120])

    if st.session_state.get("selected_symbol") not in valid_symbols:
        st.session_state["selected_symbol"] = valid_symbols[0]

    selected_symbol = st.session_state["selected_symbol"]

    st.subheader(f"Chart — {selected_symbol}")
    df_5m = intraday_map[selected_symbol]
    df_daily = daily_map[selected_symbol]
    trade_dates = get_recent_trade_dates(df_5m)
    trade_date = pd.Timestamp(trade_dates[-1])
    fig, result = plot_symbol_chart(selected_symbol, df_5m, df_daily, trade_date, cfg, tz_name)
    st.plotly_chart(fig, use_container_width=True)

    if result is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Opening range", f"{result.box_range:.2f}")
        c2.metric("ATR14", f"{result.atr14:.2f}")
        c3.metric("Liquidity threshold", f"{result.liquidity_threshold:.2f}")
        c4.metric("Signal", result.signal_type or "None")

        st.json({
            "Trade date": str(result.trade_date.date()),
            "Opening candle direction": result.open15_direction,
            "Liquidity candle": "Yes" if result.is_liquidity else "No",
            "Signal side": result.signal_side,
            "Signal type": result.signal_type,
            "Signal time": result.signal_time.strftime("%Y-%m-%d %H:%M %Z") if result.signal_time is not None else None,
            "Entry": result.entry,
            "Stop": result.stop,
            "Target": result.target,
            "Reward/Risk": result.rr,
            "Notes": result.notes,
        })

    if universe_choice == "HOT" and not hot_df.empty:
        with st.expander("HOT universe preview"):
            hot_preview_cols = [c for c in ["Symbol", "Name", "HotScore", "PctChange", "Volume", "AvgVolume", "RelVolume", "Price", "Source"] if c in hot_df.columns]
            st.dataframe(hot_df[hot_preview_cols].head(min(len(hot_df), 100)), use_container_width=True, hide_index=True)

    with st.expander("How this app interprets the transcript"):
        st.markdown(
            """
            - Boxes the first **15-minute regular-session candle**.
            - Confirms it as a liquidity candle when its range is at least **ATR(14) × threshold**.
            - Searches the next **75 minutes** so the full trade window is the **first 90 minutes** from open.
            - If the opening candle is **green**, it looks **above the box** for:
              - inverted hammer
              - bearish engulfing
            - If the opening candle is **red**, it looks **below the box** for:
              - hammer
              - bullish engulfing
            - Uses the opposite side of the box as the default target.
            - HOT mode approximates the kind of names TradingView surfaces first by ranking most active stocks, gainers, and losers.
            """
        )

    st.warning(
        "This is a strategy prototype. Backtest it carefully before using real money, and validate session rules, data quality, and execution handling."
    )
    st.info(
        "HOT mode is an approximation of the kind of names TradingView surfaces first when no symbol is typed. It does not use TradingView's private ranking model."
    )
    st.info(
        "If the HOT feed is temporarily unavailable, the app falls back to a NASDAQ symbol list so the scanner can still run."
    )
    st.info(
        "Full-market scans on free Yahoo intraday data can still be slow or rate-limited. A paid real-time market data API is better for production use."
    )


if __name__ == "__main__":
    main()
