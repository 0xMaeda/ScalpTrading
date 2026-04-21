import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Quick Flip Scalper", page_icon="📈", layout="wide")

# =========================================================
# CONFIG
# =========================================================
DEFAULT_MARKET_OPEN = time(9, 30)
MAX_REVERSAL_WINDOW_MINUTES = 90
LIQUIDITY_MAIN_THRESHOLD = 25.0
EASTERN_TZ = "America/New_York"
REQUEST_TIMEOUT = 20


# =========================================================
# DATA MODELS
# =========================================================
@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0


@dataclass
class SignalResult:
    ticker: str
    trading_day: date
    direction: str
    atr14: float
    opening_range: float
    liquidity_pct: float
    passed_liquidity: bool
    liquidity_grade: str
    box_top: float
    box_bottom: float
    reversal_type: Optional[str]
    reversal_time: Optional[datetime]
    entry: Optional[float]
    stop_loss: Optional[float]
    target: Optional[float]
    status: str
    success: Optional[bool] = None


# =========================================================
# UTILS
# =========================================================
def format_money(x: Optional[float]) -> str:
    return "—" if x is None or pd.isna(x) else f"${x:,.2f}"


def parse_ticker_list(raw: str) -> List[str]:
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def safe_get_json(url: str, headers: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def candle_direction(bar: Bar) -> str:
    return "bullish" if bar.close >= bar.open else "bearish"


def candle_range(bar: Bar) -> float:
    return abs(bar.high - bar.low)


def real_body(bar: Bar) -> float:
    return abs(bar.close - bar.open)


def upper_wick(bar: Bar) -> float:
    return bar.high - max(bar.open, bar.close)


def lower_wick(bar: Bar) -> float:
    return min(bar.open, bar.close) - bar.low


def compute_atr14(daily_df: pd.DataFrame) -> pd.Series:
    df = daily_df.copy().sort_values("date")
    prev_close = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()),
    )
    df["atr14"] = df["tr"].rolling(14).mean()
    return df["atr14"]


def is_hammer(bar: Bar) -> bool:
    body = max(real_body(bar), 0.0001)
    return lower_wick(bar) >= body * 2 and upper_wick(bar) <= body * 1.25


def is_inverted_hammer(bar: Bar) -> bool:
    body = max(real_body(bar), 0.0001)
    return upper_wick(bar) >= body * 2 and lower_wick(bar) <= body * 1.25


def is_bullish_engulfing(prev_bar: Bar, curr_bar: Bar) -> bool:
    prev_bear = prev_bar.close < prev_bar.open
    curr_bull = curr_bar.close > curr_bar.open
    body_engulf = curr_bar.open <= prev_bar.close and curr_bar.close >= prev_bar.open
    return prev_bear and curr_bull and body_engulf


def is_bearish_engulfing(prev_bar: Bar, curr_bar: Bar) -> bool:
    prev_bull = prev_bar.close > prev_bar.open
    curr_bear = curr_bar.close < curr_bar.open
    body_engulf = curr_bar.open >= prev_bar.close and curr_bar.close <= prev_bar.open
    return prev_bull and curr_bear and body_engulf


def detect_reversal(prev_bar: Optional[Bar], curr_bar: Bar, direction: str, box_top: float, box_bottom: float) -> Optional[str]:
    fully_outside_top = curr_bar.low > box_top
    fully_outside_bottom = curr_bar.high < box_bottom

    if direction == "bullish" and fully_outside_top:
        if is_inverted_hammer(curr_bar):
            return "inverted_hammer"
        if prev_bar and is_bearish_engulfing(prev_bar, curr_bar):
            return "bearish_engulfing"

    if direction == "bearish" and fully_outside_bottom:
        if is_hammer(curr_bar):
            return "hammer"
        if prev_bar and is_bullish_engulfing(prev_bar, curr_bar):
            return "bullish_engulfing"

    return None


def liquidity_grade(liquidity_pct: float) -> Tuple[bool, str]:
    if liquidity_pct >= LIQUIDITY_MAIN_THRESHOLD:
        return True, "Pass"
    return False, "Fail"


def bars_from_df(df: pd.DataFrame) -> List[Bar]:
    rows = []
    for _, r in df.iterrows():
        rows.append(
            Bar(
                ts=pd.to_datetime(r["timestamp"]).to_pydatetime(),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r.get("volume", 0)),
            )
        )
    return rows


def to_eastern_timestamps(df: pd.DataFrame, source_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[source_col] = pd.to_datetime(out[source_col], utc=True).dt.tz_convert(EASTERN_TZ).dt.tz_localize(None)
    return out


# =========================================================
# CORE STRATEGY LOGIC
# =========================================================
def find_opening_15m_bar(intraday_5m: pd.DataFrame, trading_day: date) -> Optional[Bar]:
    start_dt = datetime.combine(trading_day, DEFAULT_MARKET_OPEN)
    end_dt = start_dt + timedelta(minutes=15)
    first_window = intraday_5m[
        (intraday_5m["timestamp"] >= pd.Timestamp(start_dt)) &
        (intraday_5m["timestamp"] < pd.Timestamp(end_dt))
    ].copy()

    if len(first_window) < 3:
        return None

    return Bar(
        ts=start_dt,
        open=float(first_window.iloc[0]["open"]),
        high=float(first_window["high"].max()),
        low=float(first_window["low"].min()),
        close=float(first_window.iloc[-1]["close"]),
        volume=float(first_window["volume"].sum()) if "volume" in first_window.columns else 0,
    )


def evaluate_strategy_for_day(ticker: str, trading_day: date, intraday_5m: pd.DataFrame, atr14_value: float) -> Optional[SignalResult]:
    open15 = find_opening_15m_bar(intraday_5m, trading_day)
    if open15 is None or pd.isna(atr14_value) or atr14_value <= 0:
        return None

    direction = candle_direction(open15)
    box_top = open15.high
    box_bottom = open15.low
    opening_range = candle_range(open15)
    liquidity_pct = (opening_range / atr14_value) * 100
    passed_liq, grade = liquidity_grade(liquidity_pct)

    if not passed_liq:
        return SignalResult(
            ticker=ticker,
            trading_day=trading_day,
            direction=direction,
            atr14=atr14_value,
            opening_range=opening_range,
            liquidity_pct=liquidity_pct,
            passed_liquidity=False,
            liquidity_grade=grade,
            box_top=box_top,
            box_bottom=box_bottom,
            reversal_type=None,
            reversal_time=None,
            entry=None,
            stop_loss=None,
            target=None,
            status="failed_liquidity",
            success=None,
        )

    scan_start = datetime.combine(trading_day, DEFAULT_MARKET_OPEN) + timedelta(minutes=15)
    scan_end = datetime.combine(trading_day, DEFAULT_MARKET_OPEN) + timedelta(minutes=MAX_REVERSAL_WINDOW_MINUTES)

    scan_df = intraday_5m[
        (intraday_5m["timestamp"] >= pd.Timestamp(scan_start)) &
        (intraday_5m["timestamp"] <= pd.Timestamp(scan_end))
    ].copy()
    bars = bars_from_df(scan_df)

    signal_bar: Optional[Bar] = None
    reversal_name: Optional[str] = None

    for idx, bar in enumerate(bars):
        prev_bar = bars[idx - 1] if idx > 0 else None
        reversal = detect_reversal(prev_bar, bar, direction, box_top, box_bottom)
        if reversal:
            signal_bar = bar
            reversal_name = reversal
            break

    if signal_bar is None:
        return SignalResult(
            ticker=ticker,
            trading_day=trading_day,
            direction=direction,
            atr14=atr14_value,
            opening_range=opening_range,
            liquidity_pct=liquidity_pct,
            passed_liquidity=True,
            liquidity_grade=grade,
            box_top=box_top,
            box_bottom=box_bottom,
            reversal_type=None,
            reversal_time=None,
            entry=None,
            stop_loss=None,
            target=None,
            status="watching_reversal",
            success=None,
        )

    entry = signal_bar.close
    stop_loss = signal_bar.high if direction == "bullish" else signal_bar.low
    target = box_bottom if direction == "bullish" else box_top

    post_signal_df = intraday_5m[intraday_5m["timestamp"] > pd.Timestamp(signal_bar.ts)].copy()
    success = None
    if not post_signal_df.empty:
        if direction == "bullish":
            success = bool((post_signal_df["low"] <= box_bottom).any())
        else:
            success = bool((post_signal_df["high"] >= box_top).any())

    return SignalResult(
        ticker=ticker,
        trading_day=trading_day,
        direction=direction,
        atr14=atr14_value,
        opening_range=opening_range,
        liquidity_pct=liquidity_pct,
        passed_liquidity=True,
        liquidity_grade=grade,
        box_top=box_top,
        box_bottom=box_bottom,
        reversal_type=reversal_name,
        reversal_time=signal_bar.ts,
        entry=entry,
        stop_loss=stop_loss,
        target=target,
        status="setup_found",
        success=success,
    )


# =========================================================
# REAL DATA PROVIDERS
# =========================================================
@st.cache_data(ttl=120)
def fetch_tradingview_hotlist_iframe(theme: str = "dark") -> str:
    return f"""
    <div class=\"tradingview-widget-container\">
      <div id=\"tradingview_stock_market\"></div>
      <script type=\"text/javascript\" src=\"https://s3.tradingview.com/external-embedding/embed-widget-stock-market.js\" async>
      {{
      \"colorTheme\": \"{theme}\",
      \"dateRange\": \"1D\",
      \"exchange\": \"US\",
      \"showChart\": false,
      \"locale\": \"en\",
      \"isTransparent\": true,
      \"showSymbolLogo\": true,
      \"width\": \"100%\",
      \"height\": 700
      }}
      </script>
    </div>
    """


@st.cache_data(ttl=60)
def fetch_alpaca_most_active(api_key: str, secret_key: str, top: int = 25, by: str = "volume") -> List[str]:
    url = "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    data = safe_get_json(url, headers=headers, params={"top": top, "by": by})

    rows = data.get("most_actives", []) or data.get("data", []) or []
    symbols = []
    for row in rows:
        symbol = row.get("symbol")
        if symbol:
            symbols.append(symbol.upper())
    return symbols


@st.cache_data(ttl=60)
def fetch_alpaca_movers(api_key: str, secret_key: str) -> Tuple[List[str], List[str]]:
    url = "https://data.alpaca.markets/v1beta1/screener/stocks/movers"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    data = safe_get_json(url, headers=headers)

    gainers = [row.get("symbol", "").upper() for row in data.get("gainers", []) if row.get("symbol")]
    losers = [row.get("symbol", "").upper() for row in data.get("losers", []) if row.get("symbol")]
    return gainers, losers


@st.cache_data(ttl=300)
def fetch_polygon_intraday_5m_data(api_key: str, ticker: str, trading_day: date) -> pd.DataFrame:
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/5/minute/{trading_day}/{trading_day}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    data = safe_get_json(url, params=params)
    rows = data.get("results", [])
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = to_eastern_timestamps(df, "timestamp")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()


@st.cache_data(ttl=300)
def fetch_polygon_daily_data_for_atr(api_key: str, ticker: str, trading_day: date) -> pd.DataFrame:
    start_date = trading_day - timedelta(days=60)
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start_date}/{trading_day}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": api_key,
    }
    data = safe_get_json(url, params=params)
    rows = data.get("results", [])
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "atr14"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(EASTERN_TZ).dt.date
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    df = df[["date", "open", "high", "low", "close"]].copy().sort_values("date")
    df["atr14"] = compute_atr14(df)
    return df


# =========================================================
# CHARTS / TABLES
# =========================================================
def build_chart_df(intraday_5m: pd.DataFrame, result: SignalResult) -> pd.DataFrame:
    df = intraday_5m.copy().sort_values("timestamp")
    df["box_top"] = result.box_top
    df["box_bottom"] = result.box_bottom
    df["entry"] = result.entry
    df["stop_loss"] = result.stop_loss
    df["target"] = result.target
    return df


def create_setup_chart(chart_df: pd.DataFrame, result: SignalResult) -> go.Figure:
    plot_df = chart_df.copy().sort_values("timestamp")
    market_open = datetime.combine(result.trading_day, DEFAULT_MARKET_OPEN)
    box_end = market_open + timedelta(minutes=MAX_REVERSAL_WINDOW_MINUTES)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=plot_df["timestamp"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name=result.ticker,
        )
    )

    fig.add_shape(
        type="rect",
        x0=market_open,
        x1=box_end,
        y0=result.box_bottom,
        y1=result.box_top,
        xref="x",
        yref="y",
        fillcolor="rgba(102, 178, 255, 0.18)",
        line=dict(width=0),
        layer="below",
    )

    fig.add_hline(y=result.box_top, line_width=1.5, line_dash="dot", annotation_text=f"Box High {result.box_top:.4f}", annotation_position="right")
    fig.add_hline(y=result.box_bottom, line_width=1.5, line_dash="dot", annotation_text=f"Box Low {result.box_bottom:.4f}", annotation_position="right")

    if result.entry is not None:
        fig.add_hline(y=result.entry, line_width=2, annotation_text=f"Entry {result.entry:.4f}", annotation_position="right")
    if result.stop_loss is not None:
        fig.add_hline(y=result.stop_loss, line_width=2, line_dash="dash", annotation_text=f"Stop {result.stop_loss:.4f}", annotation_position="right")
    if result.target is not None:
        fig.add_hline(y=result.target, line_width=2, line_dash="dashdot", annotation_text=f"Target {result.target:.4f}", annotation_position="right")

    if result.reversal_time is not None:
        marker_symbol = "triangle-down" if result.direction == "bullish" else "triangle-up"
        fig.add_trace(
            go.Scatter(
                x=[result.reversal_time],
                y=[result.entry if result.entry is not None else result.box_top],
                mode="markers+text",
                marker=dict(symbol=marker_symbol, size=12),
                text=[result.reversal_type.replace("_", " ").title() if result.reversal_type else "Signal"],
                textposition="top center",
                name="Signal",
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def results_to_dataframe(results: List[SignalResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Ticker": r.ticker,
                "Date": r.trading_day,
                "Direction": r.direction,
                "ATR14": round(r.atr14, 2),
                "Opening Range": round(r.opening_range, 2),
                "Liquidity %": round(r.liquidity_pct, 2),
                "Liquidity Grade": r.liquidity_grade,
                "Passed Liquidity": r.passed_liquidity,
                "Reversal Type": r.reversal_type,
                "Reversal Time": r.reversal_time.strftime("%H:%M") if r.reversal_time else None,
                "Entry": round(r.entry, 4) if r.entry is not None else None,
                "Stop Loss": round(r.stop_loss, 4) if r.stop_loss is not None else None,
                "Target": round(r.target, 4) if r.target is not None else None,
                "Status": r.status,
                "Success": r.success,
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# EXECUTION HELPERS
# =========================================================
def scan_one_ticker_polygon(api_key: str, ticker: str, trading_day: date) -> Optional[SignalResult]:
    intraday = fetch_polygon_intraday_5m_data(api_key, ticker, trading_day)
    if intraday.empty:
        return None

    daily = fetch_polygon_daily_data_for_atr(api_key, ticker, trading_day)
    if daily.empty or daily["atr14"].dropna().empty:
        return None

    atr = float(daily["atr14"].dropna().iloc[-1])
    return evaluate_strategy_for_day(ticker, trading_day, intraday, atr)


def run_live_scanner_polygon(api_key: str, tickers: List[str], trading_day: date) -> List[SignalResult]:
    results = []
    progress = st.progress(0, text="Scanning symbols...")
    for idx, ticker in enumerate(tickers, start=1):
        result = scan_one_ticker_polygon(api_key, ticker, trading_day)
        if result is not None:
            results.append(result)
        progress.progress(idx / max(len(tickers), 1), text=f"Scanning {ticker} ({idx}/{len(tickers)})")
    progress.empty()
    return results


def run_backtest_polygon(api_key: str, ticker: str, start_date: date, end_date: date) -> List[SignalResult]:
    trading_days = pd.date_range(start=start_date, end=end_date, freq="B")
    results: List[SignalResult] = []
    progress = st.progress(0, text="Running backtest...")
    for idx, ts in enumerate(trading_days, start=1):
        day = ts.date()
        result = scan_one_ticker_polygon(api_key, ticker, day)
        if result and result.status == "setup_found":
            results.append(result)
        progress.progress(idx / max(len(trading_days), 1), text=f"Backtesting {ticker} on {day}")
    progress.empty()
    return results


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Quick Flip Scalper")
st.sidebar.caption("API keys loaded from secrets/env")

page_mode = st.sidebar.radio("Page", ["Live Scanner", "Historical Backtesting"])
provider = st.sidebar.selectbox("Primary Data Provider", ["Polygon"], index=0)
# Load API keys securely from Streamlit secrets (preferred) or environment variables
polygon_api_key = st.secrets.get("POLYGON_API_KEY", os.getenv("POLYGON_API_KEY", ""))
alpaca_api_key = st.secrets.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY", ""))
alpaca_secret_key = st.secrets.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", ""))

st.sidebar.caption("API keys loaded from secrets/env")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Strategy Rules**
- First 15-minute candle defines the opening range box
- Liquidity threshold = 25% of ATR(14)
- Reversal must form on the 5-minute chart
- Reversal must be fully outside the box
- Reversal must happen within the first 90 minutes after the open
- Target = opposite side of the opening range box
"""
)


# =========================================================
# HEADER
# =========================================================
st.title("📈 Quick Flip Scalper")
st.write(
    "This version uses real market data for 5-minute bars and ATR(14) calculations when a Polygon API key is supplied. TradingView remains available for visual hotlists, and Alpaca can supply dynamic market-mover symbol lists."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Liquidity Threshold", "25% ATR")
with c2:
    st.metric("Reversal Window", "90 min", "after market open")
with c3:
    st.metric("Reversal Chart", "5 min")
with c4:
    st.metric("Opening Box", "First 15 min")


# =========================================================
# LIVE SCANNER
# =========================================================
if page_mode == "Live Scanner":
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Scanner Inputs")
        ticker_source = st.radio(
            "Ticker Source",
            ["Manual Entry", "Alpaca Most Active", "Alpaca Movers", "TradingView Hotlist"],
            index=0,
        )
        trading_day = st.date_input("Trading Day", value=date.today())

        tickers: List[str] = []
        hotlist_html = None

        if ticker_source == "Manual Entry":
            raw_tickers = st.text_area("Tickers", value="AAPL, NVDA, TSLA, AMD", height=100)
            tickers = parse_ticker_list(raw_tickers)
        elif ticker_source == "Alpaca Most Active":
            top_n = st.slider("How many most-active tickers", min_value=1, max_value=100, value=25)
            rank_by = st.selectbox("Rank by", ["volume", "trades"], index=0)
            if alpaca_api_key and alpaca_secret_key:
                try:
                    tickers = fetch_alpaca_most_active(alpaca_api_key, alpaca_secret_key, top=top_n, by=rank_by)
                except Exception as exc:
                    st.error(f"Unable to load Alpaca most-active list: {exc}")
            else:
                st.warning("Add Alpaca API credentials in the sidebar to load a real most-active list.")
        elif ticker_source == "Alpaca Movers":
            mover_group = st.selectbox("Mover list", ["Gainers", "Losers"], index=0)
            top_n = st.slider("How many movers", min_value=1, max_value=50, value=20)
            if alpaca_api_key and alpaca_secret_key:
                try:
                    gainers, losers = fetch_alpaca_movers(alpaca_api_key, alpaca_secret_key)
                    tickers = (gainers if mover_group == "Gainers" else losers)[:top_n]
                except Exception as exc:
                    st.error(f"Unable to load Alpaca movers: {exc}")
            else:
                st.warning("Add Alpaca API credentials in the sidebar to load real gainers/losers.")
        else:
            hotlist_html = fetch_tradingview_hotlist_iframe("dark")
            st.caption("TradingView hotlist is shown on the right for review. To auto-scan dynamic lists, use Alpaca Most Active or Alpaca Movers.")

        if tickers:
            st.caption(f"Loaded {len(tickers)} symbols")
            st.code(", ".join(tickers[:50]) + (" ..." if len(tickers) > 50 else ""), language=None)

        run_scan = st.button(
            "Run Live Scan",
            width="stretch",
            disabled=(not polygon_api_key or (ticker_source != "TradingView Hotlist" and len(tickers) == 0)),
        )
        if not polygon_api_key:
            st.info("Add a Polygon API key to run the scanner and backtester on real bars.")

    with right:
        st.subheader("Live Scanner Results")

        if ticker_source == "TradingView Hotlist":
            components.html(hotlist_html, height=760)
            st.info("Use the TradingView hotlist for visual review. For actual automated scanning, choose Manual Entry, Alpaca Most Active, or Alpaca Movers.")
        else:
            if run_scan:
                try:
                    st.session_state.live_results = run_live_scanner_polygon(polygon_api_key, tickers, trading_day)
                except Exception as exc:
                    st.error(f"Live scan failed: {exc}")
                    st.session_state.live_results = []

            results: List[SignalResult] = st.session_state.get("live_results", [])
            df = results_to_dataframe(results)

            if df.empty:
                st.info("No results to display yet.")
            else:
                total = len(df)
                setup_count = int((df["Status"] == "setup_found").sum())
                pass_count = int(df["Passed Liquidity"].sum())

                m1, m2, m3 = st.columns(3)
                m1.metric("Tickers Scanned", total)
                m2.metric("Liquidity Passes", pass_count)
                m3.metric("Setups Found", setup_count)

                selection_event = st.dataframe(
                    df,
                    width="stretch",
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="live_results_table",
                )

                selected_rows = selection_event.get("selection", {}).get("rows", []) if selection_event else []
                if selected_rows:
                    chosen = results[selected_rows[0]]
                else:
                    chosen = next((r for r in results if r.status == "setup_found"), results[0] if results else None)

                if chosen:
                    st.markdown("## Selected Setup Details")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Direction", chosen.direction.title())
                    d2.metric("ATR14", f"{chosen.atr14:.2f}")
                    d3.metric("Opening Range", f"{chosen.opening_range:.2f}")
                    d4.metric("Liquidity", f"{chosen.liquidity_pct:.1f}%", chosen.liquidity_grade)

                    d5, d6, d7 = st.columns(3)
                    d5.metric("Entry", format_money(chosen.entry))
                    d6.metric("Stop Loss", format_money(chosen.stop_loss))
                    d7.metric("Target", format_money(chosen.target))

                    try:
                        intraday = fetch_polygon_intraday_5m_data(polygon_api_key, chosen.ticker, chosen.trading_day)
                        chart_df = build_chart_df(intraday, chosen)
                        st.plotly_chart(create_setup_chart(chart_df, chosen), width="stretch")
                    except Exception as exc:
                        st.error(f"Unable to render chart: {exc}")

                    st.markdown("### Rule Check")
                    st.markdown("### Rule Check")

                    rule_text = "\n".join([
                        f"- First 15-minute candle direction: **{chosen.direction}**",
                        f"- Opening range box: **{chosen.box_bottom:.2f} to {chosen.box_top:.2f}**",
                        f"- Liquidity: **{chosen.liquidity_pct:.1f}% of ATR14** ({chosen.liquidity_grade})",
                        f"- Reversal detected: **{chosen.reversal_type or 'None'}**",
                        f"- Status: **{chosen.status}**"
                    ])

                    st.markdown(rule_text)

# =========================================================
# HISTORICAL BACKTESTING
# =========================================================
else:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Backtest Inputs")
        ticker = st.text_input("Ticker", value="AAPL").upper().strip()
        preset = st.selectbox("Date Range", ["1 week", "1 month", "1 year", "YTD", "3 years", "All time", "Custom"])

        today = date.today()
        if preset == "1 week":
            start_date = today - timedelta(days=7)
            end_date = today
        elif preset == "1 month":
            start_date = today - timedelta(days=30)
            end_date = today
        elif preset == "1 year":
            start_date = today - timedelta(days=365)
            end_date = today
        elif preset == "YTD":
            start_date = date(today.year, 1, 1)
            end_date = today
        elif preset == "3 years":
            start_date = today - timedelta(days=365 * 3)
            end_date = today
        elif preset == "All time":
            start_date = date(2018, 1, 1)
            end_date = today
        else:
            start_date = st.date_input("Start Date", value=today - timedelta(days=30))
            end_date = st.date_input("End Date", value=today)

        run_backtest_btn = st.button("Run Historical Review", width="stretch", disabled=not polygon_api_key)
        if not polygon_api_key:
            st.info("Add a Polygon API key to run the backtester.")

    with right:
        st.subheader("Historical Results")

        if run_backtest_btn:
            try:
                st.session_state.backtest_results = run_backtest_polygon(polygon_api_key, ticker, start_date, end_date)
            except Exception as exc:
                st.error(f"Historical review failed: {exc}")
                st.session_state.backtest_results = []

        results: List[SignalResult] = st.session_state.get("backtest_results", [])
        df = results_to_dataframe(results)

        if df.empty:
            st.info("No historical setups found for the selected period.")
        else:
            success_rate = float(df["Success"].fillna(False).mean() * 100)
            b1, b2, b3 = st.columns(3)
            b1.metric("Historical Setups", len(df))
            b2.metric("Success Rate", f"{success_rate:.1f}%")
            b3.metric("Ticker", ticker)

            selection_event = st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="historical_results_table",
            )
            selected_rows = selection_event.get("selection", {}).get("rows", []) if selection_event else []

            st.markdown("### Setup Success Definition")
            st.write(
                "A setup is counted as successful only when the first 15-minute candle passes the 25%-of-ATR liquidity test, a valid 5-minute reversal forms completely outside the opening-range box within the first 90 minutes, and price later reaches the opposite side of the opening-range box."
            )

            if selected_rows:
                chosen_hist = results[selected_rows[0]]
            else:
                chosen_hist = next((r for r in results if r.success), results[0] if results else None)

            if chosen_hist:
                st.markdown("## Selected Historical Setup")
                try:
                    intraday = fetch_polygon_intraday_5m_data(polygon_api_key, chosen_hist.ticker, chosen_hist.trading_day)
                    chart_df = build_chart_df(intraday, chosen_hist)
                    st.plotly_chart(create_setup_chart(chart_df, chosen_hist), width="stretch")
                except Exception as exc:
                    st.error(f"Unable to render historical chart: {exc}")

                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Direction", chosen_hist.direction.title())
                h2.metric("Reversal", chosen_hist.reversal_type or "—")
                h3.metric("Entry", format_money(chosen_hist.entry))
                h4.metric("Success", "Yes" if chosen_hist.success else "No")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
### Notes
- Polygon is used here for real intraday bars and daily ATR data.
- Alpaca is optional and used here for dynamic market-mover symbol lists.
- TradingView remains embedded for visual hotlist review, but the scanner logic runs on provider data rather than widget output.
- To run locally, install dependencies and set environment variables or paste your API keys in the sidebar.
"""
)
