import json
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock AI Agent | Professional Analysis", page_icon="📈", layout="wide")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TIMEFRAMES = {
    "1 मिनिट": ("1m", "5d", None),
    "5 मिनिट": ("5m", "60d", None),
    "15 मिनिट": ("15m", "60d", None),
    "30 मिनिट": ("30m", "60d", None),
    "1 तास": ("60m", "1y", None),
    "4 तास": ("60m", "1y", "4h"),
    "1 दिवस": ("1d", "5y", None),
    "1 आठवडा": ("1wk", "10y", None),
}

COMMODITIES = {
    "GOLD": "GC=F", "XAUUSD": "GC=F", "SILVER": "SI=F", "XAGUSD": "SI=F",
    "PLATINUM": "PL=F", "COPPER": "HG=F", "CRUDEOIL": "CL=F", "CRUDE": "CL=F",
    "BRENT": "BZ=F", "NATURALGAS": "NG=F", "COTTON": "CT=F", "SUGAR": "SB=F",
    "COFFEE": "KC=F", "WHEAT": "ZW=F", "CORN": "ZC=F", "SOYBEAN": "ZS=F",
    "BTCUSD": "BTC-USD", "BTC": "BTC-USD", "ETHUSD": "ETH-USD", "ETH": "ETH-USD",
    "SOLUSD": "SOL-USD", "XRPUSD": "XRP-USD", "DOGEUSD": "DOGE-USD", "BNBUSD": "BNB-USD",
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "USDINR": "USDINR=X",
}

TV_MAP = {
    "XAUUSD": "OANDA:XAUUSD", "GOLD": "OANDA:XAUUSD", "XAGUSD": "OANDA:XAGUSD", "SILVER": "OANDA:XAGUSD",
    "CRUDEOIL": "TVC:USOIL", "CRUDE": "TVC:USOIL", "BRENT": "TVC:UKOIL",
    "NATURALGAS": "CAPITALCOM:NATURALGAS", "COPPER": "CAPITALCOM:COPPER",
    "BTCUSD": "BINANCE:BTCUSDT", "BTC": "BINANCE:BTCUSDT", "ETHUSD": "BINANCE:ETHUSDT", "ETH": "BINANCE:ETHUSDT",
    "SOLUSD": "BINANCE:SOLUSDT", "XRPUSD": "BINANCE:XRPUSDT", "DOGEUSD": "BINANCE:DOGEUSDT", "BNBUSD": "BINANCE:BNBUSDT",
    "EURUSD": "FX:EURUSD", "GBPUSD": "FX:GBPUSD", "USDJPY": "FX:USDJPY", "USDINR": "FX_IDC:USDINR",
}

TF_WEIGHTS = {
    "1 मिनिट": 0.5, "5 मिनिट": 1.0, "15 मिनिट": 1.15, "30 मिनिट": 1.25,
    "1 तास": 1.5, "4 तास": 1.8, "1 दिवस": 2.0, "1 आठवडा": 2.2,
}


def normalize_name(name: str) -> str:
    return name.strip().upper().replace(" ", "").replace("_", "")


def resolve_symbol(name: str, market: str) -> str:
    key = normalize_name(name)
    if key in COMMODITIES:
        return COMMODITIES[key]
    if not key or any(x in key for x in (".", "=", "-")) or key.startswith("^"):
        return key
    if market == "NSE (भारत)":
        return key + ".NS"
    if market == "BSE (भारत)":
        return key + ".BO"
    return key


def tv_symbol(name: str, market: str) -> str:
    key = normalize_name(name)
    if key in TV_MAP:
        return TV_MAP[key]
    if ":" in key:
        return key
    if market == "NSE (भारत)":
        return "NSE:" + key
    if market == "BSE (भारत)":
        return "BSE:" + key
    return key


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
@st.cache_data(ttl=45, show_spinner=False)
def load_data(symbol: str, interval: str, period: str, resample: Optional[str] = None) -> pd.DataFrame:
    data = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if data is None or data.empty:
        return pd.DataFrame()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(data.columns):
        return pd.DataFrame()
    data = data[needed].copy().dropna(subset=["Open", "High", "Low", "Close"])
    if resample:
        data = data.resample(resample, origin="start_day").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna(subset=["Open", "High", "Low", "Close"])
    return data


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"].fillna(0)

    for length in (20, 50, 200):
        df[f"EMA{length}"] = close.ewm(span=length, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_SIG"] = macd.ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIG"]

    middle = close.rolling(20).mean()
    deviation = close.rolling(20).std()
    df["BB_UP"] = middle + 2 * deviation
    df["BB_LOW"] = middle - 2 * deviation

    previous_close = close.shift(1)
    tr = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)
    df["PLUS_DI"] = plus_di.fillna(0)
    df["MINUS_DI"] = minus_di.fillna(0)

    df["VOL_MA"] = volume.rolling(20).mean()
    df["SUPPORT"] = low.rolling(20).min()
    df["RESIST"] = high.rolling(20).max()
    return df.replace([np.inf, -np.inf], np.nan)


# -----------------------------------------------------------------------------
# Signal engine: rule-based technical scoring, not a prediction probability
# -----------------------------------------------------------------------------
def make_signal(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["EMA20", "EMA50", "RSI", "MACD", "MACD_SIG", "ATR"])
    if valid.empty:
        raise ValueError("Technical indicators साठी पुरेसा डेटा उपलब्ध नाही.")
    last = valid.iloc[-1]
    score = 0.0
    max_score = 0.0
    reasons = []

    def add(points: float, weight: float, positive_text: str, negative_text: str):
        nonlocal score, max_score
        score += points * weight
        max_score += weight
        reasons.append(("🟢 " + positive_text) if points > 0 else ("🔴 " + negative_text) if points < 0 else "⚪ Neutral")

    close = float(last["Close"])
    # Trend structure
    add(1 if close > last["EMA50"] else -1, 1.4,
        "भाव EMA50 च्या वर आहे", "भाव EMA50 च्या खाली आहे")
    add(1 if last["EMA20"] > last["EMA50"] else -1, 1.2,
        "EMA20 > EMA50: short-term trend bullish", "EMA20 < EMA50: short-term trend bearish")
    if len(valid) >= 200:
        add(1 if close > last["EMA200"] else -1, 1.0,
            "भाव EMA200 च्या वर आहे", "भाव EMA200 च्या खाली आहे")
    else:
        reasons.append("⚪ EMA200 साठी 200 candles उपलब्ध नाहीत")

    # RSI momentum: neutral zone does not force a direction
    rsi = float(last["RSI"])
    if 52 <= rsi <= 68:
        add(1, 1.0, f"RSI {rsi:.1f}: bullish momentum", "")
    elif 32 <= rsi <= 48:
        add(-1, 1.0, "", f"RSI {rsi:.1f}: bearish momentum")
    elif rsi > 75:
        add(-0.5, 0.6, "", f"RSI {rsi:.1f}: overbought; pullback risk")
    elif rsi < 25:
        add(0.5, 0.6, f"RSI {rsi:.1f}: oversold; bounce possible", "")
    else:
        reasons.append(f"⚪ RSI {rsi:.1f}: neutral zone")

    add(1 if last["MACD"] > last["MACD_SIG"] else -1, 1.0,
        "MACD signal line च्या वर", "MACD signal line च्या खाली")

    # Directional strength
    if float(last["ADX"]) >= 20:
        di_up = last["PLUS_DI"] > last["MINUS_DI"]
        add(1 if di_up else -1, 1.0,
            f"ADX {last['ADX']:.1f}: bullish directional strength", f"ADX {last['ADX']:.1f}: bearish directional strength")
    else:
        reasons.append(f"⚪ ADX {last['ADX']:.1f}: trend strength कमी")

    # Volume is not reliable for every FX/index/futures feed; use only when meaningful.
    vol_ma = last["VOL_MA"]
    if pd.notna(vol_ma) and vol_ma > 0 and last["Volume"] > 1.5 * vol_ma:
        up_candle = last["Close"] >= last["Open"]
        add(1 if up_candle else -1, 0.7,
            "जास्त volume सोबत bullish candle", "जास्त volume सोबत bearish candle")
    else:
        reasons.append("⚪ Volume confirmation उपलब्ध नाही किंवा सामान्य आहे")

    normalized = score / max_score if max_score else 0
    strength = int(round(abs(normalized) * 100))
    if normalized >= 0.35:
        action = "BUY"
    elif normalized <= -0.35:
        action = "SELL"
    else:
        action = "HOLD"

    if action == "BUY" and strength >= 65:
        label = "STRONG BUY"
    elif action == "SELL" and strength >= 65:
        label = "STRONG SELL"
    elif action == "BUY":
        label = "BUY"
    elif action == "SELL":
        label = "SELL"
    else:
        label = "WAIT / HOLD"

    atr = float(last["ATR"])
    entry_low = entry_high = sl = t1 = t2 = t3 = None
    risk_reward = None
    if action in ("BUY", "SELL") and np.isfinite(atr) and atr > 0:
        # Entry zone is a planning range around the latest close, not a guaranteed fill price.
        if action == "BUY":
            entry_low, entry_high = close - 0.20 * atr, close + 0.10 * atr
            entry_mid = (entry_low + entry_high) / 2
            sl = close - 1.50 * atr
            risk = max(entry_mid - sl, 0)
            t1, t2, t3 = entry_mid + risk, entry_mid + 2 * risk, entry_mid + 3 * risk
        else:
            entry_low, entry_high = close - 0.10 * atr, close + 0.20 * atr
            entry_mid = (entry_low + entry_high) / 2
            sl = close + 1.50 * atr
            risk = max(sl - entry_mid, 0)
            t1, t2, t3 = entry_mid - risk, entry_mid - 2 * risk, entry_mid - 3 * risk
        risk_reward = "1:1 / 1:2 / 1:3"

    return {
        "action": action, "label": label, "score": score, "strength": strength,
        "normalized": normalized, "reasons": reasons, "close": close, "rsi": rsi,
        "atr": atr, "adx": float(last["ADX"]), "support": float(last["SUPPORT"]),
        "resist": float(last["RESIST"]), "ema20": float(last["EMA20"]),
        "ema50": float(last["EMA50"]), "ema200": float(last["EMA200"]) if len(valid) >= 200 else None,
        "entry_low": entry_low, "entry_high": entry_high, "sl": sl,
        "t1": t1, "t2": t2, "t3": t3, "risk_reward": risk_reward,
        "timestamp": str(valid.index[-1]),
    }


def build_chart(df: pd.DataFrame, sig: dict, title: str) -> go.Figure:
    plot_df = df.tail(220).copy()
    x = plot_df.index.strftime("%d %b %H:%M")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.55, 0.12, 0.16, 0.17])
    fig.add_trace(go.Candlestick(x=x, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"],
                                 close=plot_df["Close"], name="Price"), row=1, col=1)
    for n, color in ((20, "#f5a623"), (50, "#4a90e2"), (200, "#9b59b6")):
        fig.add_trace(go.Scatter(x=x, y=plot_df[f"EMA{n}"], name=f"EMA{n}",
                                 line=dict(width=1.2, color=color)), row=1, col=1)
    for col in ("BB_UP", "BB_LOW"):
        fig.add_trace(go.Scatter(x=x, y=plot_df[col], name=col, showlegend=False,
                                 line=dict(width=0.8, dash="dot", color="gray")), row=1, col=1)

    if sig["action"] in ("BUY", "SELL"):
        lines = [(sig["entry_low"], "Entry Zone Low", "#1e88e5"),
                 (sig["entry_high"], "Entry Zone High", "#1e88e5"),
                 (sig["sl"], "Stop Loss", "#e53935"), (sig["t1"], "Target 1", "#43a047"),
                 (sig["t2"], "Target 2", "#2e7d32"), (sig["t3"], "Target 3", "#1b5e20")]
        for value, label, color in lines:
            fig.add_hline(y=value, line=dict(color=color, dash="dash", width=1),
                          annotation_text=f"{label}: {value:.4f}", annotation_position="right", row=1, col=1)
    fig.add_hline(y=sig["support"], line=dict(color="#26a69a", dash="dot", width=1),
                  annotation_text="Support", row=1, col=1)
    fig.add_hline(y=sig["resist"], line=dict(color="#ef5350", dash="dot", width=1),
                  annotation_text="Resistance", row=1, col=1)

    volume_colors = np.where(plot_df["Close"] >= plot_df["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=x, y=plot_df["Volume"], marker_color=volume_colors, name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=plot_df["RSI"], name="RSI", line=dict(color="#8e44ad")), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="red", dash="dot", width=0.8), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="green", dash="dot", width=0.8), row=3, col=1)
    hist_colors = np.where(plot_df["MACD_HIST"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=x, y=plot_df["MACD_HIST"], marker_color=hist_colors, name="MACD Histogram", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=plot_df["MACD"], name="MACD", line=dict(color="#1e88e5", width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=plot_df["MACD_SIG"], name="Signal", line=dict(color="#f5a623", width=1)), row=4, col=1)
    fig.update_xaxes(type="category", nticks=7, rangeslider_visible=False)
    fig.update_layout(title=title, height=850, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.04), dragmode="pan")
    return fig


# -----------------------------------------------------------------------------
# TradingView widgets
# -----------------------------------------------------------------------------
def tv_chart_html(symbol: str) -> str:
    cfg = {"autosize": True, "symbol": symbol, "interval": "5", "timezone": "Asia/Kolkata",
           "theme": "light", "style": "1", "locale": "en", "allow_symbol_change": True,
           "support_host": "https://www.tradingview.com"}
    return ('<div class="tradingview-widget-container" style="height:640px;width:100%">'
            '<div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>'
            '<script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>'
            + json.dumps(cfg) + '</script></div>')


def tv_rating_html(symbol: str) -> str:
    cfg = {"interval": "1h", "width": "100%", "height": 480, "isTransparent": False,
           "symbol": symbol, "showIntervalTabs": True, "displayMode": "single", "locale": "en", "colorTheme": "light"}
    return ('<div class="tradingview-widget-container"><div class="tradingview-widget-container__widget"></div>'
            '<script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>'
            + json.dumps(cfg) + '</script></div>')


def fmt(value, digits=4):
    return "—" if value is None or not np.isfinite(value) else f"{value:,.{digits}f}"


def weighted_consensus(results: dict) -> tuple[str, int, int, int, int]:
    weighted = 0.0
    total = 0.0
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for label, result in results.items():
        if not result:
            continue
        action = result["action"]
        counts[action] += 1
        weight = TF_WEIGHTS[label]
        weighted += (1 if action == "BUY" else -1 if action == "SELL" else 0) * weight
        total += weight
    pct = int(round(100 * abs(weighted) / total)) if total else 0
    if total and weighted / total >= 0.30:
        bias = "BULLISH BIAS"
    elif total and weighted / total <= -0.30:
        bias = "BEARISH BIAS"
    else:
        bias = "MIXED / WAIT"
    return bias, counts["BUY"], counts["SELL"], counts["HOLD"], pct


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("📈 Stock AI Agent – Professional Trading Analysis")
st.caption("Multi-timeframe technical analysis • Manual trading assistant • Marathi explanation")

with st.sidebar:
    st.header("⚙️ Market Settings")
    name = st.text_input("Stock / Asset Symbol", value="XAUUSD",
                         help="उदा. RELIANCE, TCS, XAUUSD, GOLD, BTCUSD, EURUSD")
    market = st.selectbox("Market", ["NSE (भारत)", "BSE (भारत)", "US"], index=0,
                          help="Commodity, Forex आणि Crypto साठी हे selection symbol mapping बदलत नाही.")
    selected_tfs = st.multiselect("Analysis Timeframes", list(TIMEFRAMES.keys()), default=list(TIMEFRAMES.keys()))
    if not selected_tfs:
        selected_tfs = list(TIMEFRAMES.keys())
    st.divider()
    auto_refresh = st.toggle("Auto refresh", value=True)
    refresh_seconds = st.slider("Refresh interval (seconds)", 30, 180, 60, disabled=not auto_refresh)
    closed_candle = st.toggle("फक्त पूर्ण झालेली candle वापरा", value=True,
                              help="Intraday timeframe मध्ये चालू candle काढून टाकते. Signal स्थिर ठेवण्यास मदत होते.")
    st.caption("डेटा स्रोत: Yahoo Finance (yfinance). हा broker-grade real-time feed नाही.")

symbol = resolve_symbol(name, market)

if not name.strip():
    st.info("Sidebar मध्ये Stock किंवा Asset चे नाव टाका.")
    st.stop()


@st.fragment(run_every=refresh_seconds if auto_refresh else None)
def render_analysis():
    st.subheader(f"🎯 {normalize_name(name)}  |  Data symbol: {symbol}")
    results = {}
    errors = {}
    for label in selected_tfs:
        interval, period, resample = TIMEFRAMES[label]
        try:
            raw = load_data(symbol, interval, period, resample)
            if raw.empty:
                results[label] = None
                errors[label] = "डेटा मिळाला नाही"
                continue
            if closed_candle and interval in ("1m", "5m", "15m", "30m", "60m") and len(raw) > 35:
                raw = raw.iloc[:-1]
            if len(raw) < 30:
                results[label] = None
                errors[label] = f"पुरेशा candles नाहीत ({len(raw)})"
                continue
            enriched = add_indicators(raw)
            results[label] = {"df": enriched, "signal": make_signal(enriched)}
        except Exception as exc:
            results[label] = None
            errors[label] = str(exc)

    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        st.error(f"{symbol} साठी कोणत्याही timeframe चा डेटा मिळाला नाही.")
        with st.expander("तपशील / संभाव्य कारण"):
            st.write(errors)
            st.write("Symbol योग्य आहे का, market निवड बरोबर आहे का आणि Yahoo Finance वर त्या timeframe चा डेटा उपलब्ध आहे का ते तपासा.")
        return

    bias, buy_count, sell_count, hold_count, agreement = weighted_consensus(
        {k: v["signal"] for k, v in valid_results.items()}
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall Market Bias", bias)
    c2.metric("BUY timeframes", buy_count)
    c3.metric("SELL timeframes", sell_count)
    c4.metric("HOLD timeframes", hold_count)
    c5.metric("Directional agreement", f"{agreement}%", help="Weighted directional agreement; probability of profit नाही.")

    st.markdown("#### Multi-Timeframe Signal Board")
    ordered = [(tf, valid_results[tf]["signal"]) for tf in selected_tfs if tf in valid_results]
    for start in range(0, len(ordered), 4):
        cols = st.columns(4)
        for col, (tf, sig) in zip(cols, ordered[start:start + 4]):
            icon = "🟢" if sig["action"] == "BUY" else "🔴" if sig["action"] == "SELL" else "🟡"
            col.metric(tf, f"{icon} {sig['label']}", f"Strength {sig['strength']}% | {fmt(sig['close'])}")

    if bias == "BULLISH BIAS":
        st.success("मोठ्या timeframe ला जास्त weight देऊन एकूण कल bullish दिसतो. Entry आधी लहान timeframe confirmation तपासा.")
    elif bias == "BEARISH BIAS":
        st.error("मोठ्या timeframe ला जास्त weight देऊन एकूण कल bearish दिसतो. Entry आधी लहान timeframe confirmation तपासा.")
    else:
        st.warning("Timeframes मध्ये स्पष्ट एकमत नाही. घाईने trade घेण्याऐवजी setup स्पष्ट होण्याची वाट पाहा.")

    chosen = st.radio("Detailed Analysis साठी timeframe निवडा", [x for x in selected_tfs if x in valid_results],
                      horizontal=True, key="detail_tf")
    df = valid_results[chosen]["df"]
    sig = valid_results[chosen]["signal"]

    st.divider()
    st.subheader("📌 Detailed Trade Setup")
    a, b, c = st.columns(3)
    a.metric("Signal", sig["label"])
    b.metric("Setup Strength", f"{sig['strength']}%", help="Indicator agreement score; profit probability नाही.")
    c.metric("Last Analyzed Price", fmt(sig["close"]))

    if sig["action"] in ("BUY", "SELL"):
        st.markdown("**Entry Zone**")
        e1, e2, e3 = st.columns(3)
        e1.metric("Entry Zone Low", fmt(sig["entry_low"]))
        e2.metric("Entry Zone High", fmt(sig["entry_high"]))
        e3.metric("Stop Loss", fmt(sig["sl"]))
        t1, t2, t3 = st.columns(3)
        t1.metric("Target 1 (1R)", fmt(sig["t1"]))
        t2.metric("Target 2 (2R)", fmt(sig["t2"]))
        t3.metric("Target 3 (3R)", fmt(sig["t3"]))
        st.caption("Entry Zone ही ATR-आधारित planning range आहे; guaranteed execution price नाही. Targets हे अंदाजे risk multiples आहेत.")
    else:
        st.info("सध्या HOLD / WAIT. स्पष्ट BUY किंवा SELL setup नाही; त्यामुळे Entry, SL आणि Targets दाखवलेले नाहीत.")

    st.markdown("**Market Structure & Indicators**")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RSI (14)", fmt(sig["rsi"], 1))
    m2.metric("ADX (14)", fmt(sig["adx"], 1))
    m3.metric("ATR (14)", fmt(sig["atr"]))
    m4.metric("Support", fmt(sig["support"]))
    m5.metric("Resistance", fmt(sig["resist"]))

    with st.expander("Signal कसा तयार झाला?", expanded=True):
        for reason in sig["reasons"]:
            st.write(reason)
        st.caption(f"शेवटची वापरलेली candle: {sig['timestamp']}")

    st.plotly_chart(build_chart(df, sig, f"{normalize_name(name)} • {chosen}"), use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False})

    st.subheader("🧭 Trading Plan – Manual Checklist")
    st.markdown("- [ ] 4H / 1H trend आणि निवडलेल्या entry timeframe ची दिशा जुळते का ते तपासा.\n"
                "- [ ] Entry zone मध्ये price आल्यावर candle close / price action confirmation तपासा.\n"
                "- [ ] प्रत्येक trade आधी position size आणि कमाल नुकसान ठरवा.\n"
                "- [ ] Stop Loss हलवून नुकसान वाढवू नका.\n"
                "- [ ] महत्त्वाच्या news / market volatility वेळी सावध राहा.")

    st.caption("महत्त्वाचे: हा rule-based technical analysis assistant आहे; तो नफा किंवा signal accuracy ची हमी देत नाही. yfinance data delayed/limited असू शकतो. आधी backtesting आणि paper trading करा. हा वैयक्तिक गुंतवणूक सल्ला नाही.")

    if errors:
        with st.expander("काही timeframe चा डेटा का आला नाही?"):
            for tf, message in errors.items():
                st.write(f"**{tf}:** {message}")


render_analysis()

# -----------------------------------------------------------------------------
# TradingView comparison
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📺 TradingView Comparison")
tv_sym = st.text_input("TradingView Symbol", value=tv_symbol(name, market),
                       help="उदा. OANDA:XAUUSD, BINANCE:BTCUSDT, NSE:RELIANCE, MCX:GOLD1!")
tab_chart, tab_rating = st.tabs(["📈 TradingView Chart", "🎯 TradingView Technical Rating"])
with tab_chart:
    components.html(tv_chart_html(tv_sym), height=660, scrolling=False)
with tab_rating:
    st.caption("TradingView widget मधील उपलब्ध timeframe tabs वापरून rating तपासा. हे आपल्या yfinance signals पेक्षा वेगळ्या data feed वर असू शकते.")
    components.html(tv_rating_html(tv_sym), height=500, scrolling=False)

st.caption("Data source difference: XAUUSD साठी आपल्या analysis मध्ये Yahoo Finance Gold Futures (GC=F) वापरले जातात; TradingView widget मध्ये OANDA Spot Gold असू शकते. त्यामुळे price आणि indicators तंतोतंत जुळतीलच असे नाही.")
