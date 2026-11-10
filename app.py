import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock AI Agent", page_icon="📈", layout="wide")

TIMEFRAMES = {
    "5 मिनिट": ("5m", "1mo", None),
    "1 तास": ("60m", "1y", None),
    "4 तास": ("60m", "1y", "4h"),  # 1 तासाच्या डेटावरून तयार केला जातो
}


# ---------------------------------------------------------------- data
COMMODITIES = {
    "GOLD": "GC=F", "SILVER": "SI=F", "PLATINUM": "PL=F", "COPPER": "HG=F",
    "CRUDEOIL": "CL=F", "CRUDE": "CL=F", "BRENT": "BZ=F", "NATURALGAS": "NG=F",
    "COTTON": "CT=F", "SUGAR": "SB=F", "COFFEE": "KC=F",
    "WHEAT": "ZW=F", "CORN": "ZC=F", "SOYBEAN": "ZS=F",
    # Gold / Silver (USD)
    "XAUUSD": "GC=F", "XAGUSD": "SI=F",
    # Crypto
    "BTCUSD": "BTC-USD", "BTC": "BTC-USD", "ETHUSD": "ETH-USD", "ETH": "ETH-USD",
    "SOLUSD": "SOL-USD", "XRPUSD": "XRP-USD", "DOGEUSD": "DOGE-USD", "BNBUSD": "BNB-USD",
    # Forex
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "USDINR": "USDINR=X",
}


def resolve_symbol(name: str, market: str) -> str:
    name = name.strip().upper().replace(" ", "").replace("_", "")
    if name in COMMODITIES:
        return COMMODITIES[name]
    if not name or "." in name or "=" in name or "-" in name or name.startswith("^"):
        return name
    if market == "NSE (भारत)":
        return name + ".NS"
    if market == "BSE (भारत)":
        return name + ".BO"
    return name  # US


@st.cache_data(ttl=30, show_spinner=False)
def load_data(symbol: str, interval: str, period: str, resample: str = None) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if resample:
        df = df.resample(resample).agg({"Open": "first", "High": "max", "Low": "min",
                                        "Close": "last", "Volume": "sum"}).dropna()
    return df


# ---------------------------------------------------------------- indicators
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    for n in (20, 50, 200):
        df[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()

    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_SIG"] = macd.ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIG"]

    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["BB_UP"], df["BB_LOW"] = ma + 2 * sd, ma - 2 * sd

    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["VOL_MA"] = v.rolling(20).mean()
    df["SUPPORT"] = l.rolling(20).min()
    df["RESIST"] = h.rolling(20).max()
    return df


# ---------------------------------------------------------------- signal
def make_signal(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    score, reasons = 0, []

    def add(points: int, text: str):
        nonlocal score
        score += points
        icon = "🟢" if points > 0 else "🔴" if points < 0 else "⚪"
        reasons.append(f"{icon} {text}")

    close = last["Close"]

    # Trend
    add(1 if close > last["EMA50"] else -1,
        "भाव EMA50 च्या वर (तेजी)" if close > last["EMA50"] else "भाव EMA50 च्या खाली (मंदी)")
    add(1 if last["EMA20"] > last["EMA50"] else -1,
        "EMA20 > EMA50 (short-term तेजी)" if last["EMA20"] > last["EMA50"] else "EMA20 < EMA50 (short-term मंदी)")
    if len(df) >= 200:
        add(1 if close > last["EMA200"] else -1,
            "भाव EMA200 च्या वर (long-term तेजी)" if close > last["EMA200"] else "भाव EMA200 च्या खाली (long-term मंदी)")

    # RSI
    rsi = last["RSI"]
    if rsi < 30:
        add(1, f"RSI {rsi:.0f}: oversold, उसळी येऊ शकते")
    elif rsi > 70:
        add(-1, f"RSI {rsi:.0f}: overbought, घसरण येऊ शकते")
    else:
        add(0, f"RSI {rsi:.0f}: neutral")

    # MACD
    if last["MACD"] > last["MACD_SIG"]:
        add(1, "MACD signal line च्या वर (momentum तेजीचा)")
    else:
        add(-1, "MACD signal line च्या खाली (momentum मंदीचा)")

    # Bollinger
    if close < last["BB_LOW"]:
        add(1, "भाव Bollinger खालच्या band च्या खाली")
    elif close > last["BB_UP"]:
        add(-1, "भाव Bollinger वरच्या band च्या वर")

    # Volume confirmation
    if last["VOL_MA"] and last["Volume"] > 1.5 * last["VOL_MA"]:
        up = last["Close"] >= last["Open"]
        add(1 if up else -1, "जास्त volume सोबत " + ("हिरवी candle" if up else "लाल candle"))

    if score >= 3:
        action = "BUY"
    elif score <= -3:
        action = "SELL"
    else:
        action = "HOLD"

    atr = last["ATR"]
    levels = {}
    if action == "BUY":
        levels = {"entry": close, "sl": close - 1.5 * atr, "target": close + 3 * atr}
    elif action == "SELL":
        levels = {"entry": close, "sl": close + 1.5 * atr, "target": close - 3 * atr}

    return {"action": action, "score": score, "reasons": reasons, "levels": levels,
            "close": close, "rsi": rsi, "atr": atr,
            "support": last["SUPPORT"], "resist": last["RESIST"]}


# ---------------------------------------------------------------- chart
def build_chart(df: pd.DataFrame, sig: dict, title: str) -> go.Figure:
    x = df.index.strftime("%d %b %H:%M")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.55, 0.12, 0.16, 0.17])

    fig.add_trace(go.Candlestick(x=x, open=df["Open"], high=df["High"], low=df["Low"],
                                 close=df["Close"], name="Price"), row=1, col=1)
    for n, color in ((20, "#f5a623"), (50, "#4a90e2"), (200, "#9b59b6")):
        fig.add_trace(go.Scatter(x=x, y=df[f"EMA{n}"], name=f"EMA{n}",
                                 line=dict(width=1.2, color=color)), row=1, col=1)
    for col in ("BB_UP", "BB_LOW"):
        fig.add_trace(go.Scatter(x=x, y=df[col], name=col, showlegend=False,
                                 line=dict(width=0.8, dash="dot", color="gray")), row=1, col=1)

    lv = sig["levels"]
    if lv:
        for key, color, label in (("entry", "#1e88e5", "Entry"), ("sl", "#e53935", "Stop-loss"),
                                  ("target", "#43a047", "Target")):
            fig.add_hline(y=lv[key], line=dict(color=color, dash="dash", width=1.2),
                          annotation_text=f"{label} {lv[key]:.2f}", annotation_position="right",
                          row=1, col=1)

    colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=x, y=df["Volume"], marker_color=colors, name="Volume",
                         showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["RSI"], name="RSI", line=dict(color="#8e44ad")), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="red", dash="dot", width=0.8), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="green", dash="dot", width=0.8), row=3, col=1)

    hist_colors = np.where(df["MACD_HIST"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=x, y=df["MACD_HIST"], marker_color=hist_colors, showlegend=False,
                         name="Hist"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["MACD"], name="MACD", line=dict(color="#1e88e5", width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["MACD_SIG"], name="Signal", line=dict(color="#f5a623", width=1)), row=4, col=1)

    fig.update_xaxes(type="category", nticks=6, rangeslider_visible=False)
    fig.update_layout(title=title, height=820, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.04), xaxis_rangeslider_visible=False,
                      dragmode="pan")
    return fig


# ---------------------------------------------------------------- AI explanation
def get_api_key(typed: str) -> str:
    if typed:
        return typed
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return ""


def ai_explain(api_key: str, symbol: str, tf: str, sig: dict, others: str = "") -> str:
    import anthropic

    lv = sig["levels"]
    prompt = f"""तू एक अनुभवी शेअर बाजार विश्लेषक आहेस. खालील technical डेटा वरून सोप्या मराठीत (नवशिक्याला समजेल असं) विश्लेषण लिही.

Stock: {symbol}, Timeframe: {tf}
Signal: {sig['action']} (score {sig['score']})
सर्व timeframes: {others}
Last price: {sig['close']:.2f}, RSI: {sig['rsi']:.1f}, ATR: {sig['atr']:.2f}
Support: {sig['support']:.2f}, Resistance: {sig['resist']:.2f}
Entry/SL/Target: {lv if lv else 'लागू नाही'}
कारणे:
""" + "\n".join(sig["reasons"]) + """

हे लिही: 1) सध्याची परिस्थिती 2) हा signal का आला 3) मुख्य धोके 4) risk management सल्ला (position size, stop-loss पाळणे).
शेवटी एक ओळ: हा आर्थिक सल्ला नाही, फक्त शैक्षणिक विश्लेषण आहे."""
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(model="claude-sonnet-5-5", max_tokens=900,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if b.type == "text")


# ---------------------------------------------------------------- UI
st.title("📈 Stock AI Agent")
st.caption("स्टॉकचं नाव टाका, 5 मिनिट / 1 तास / 4 तास तिन्ही timeframe चे Buy/Sell signals मिळतील.")

with st.sidebar:
    st.header("⚙️ Settings")
    name = st.text_input("Stock चं नाव", value="XAUUSD",
                         help="Stock: RELIANCE, TCS, AAPL | Commodity: GOLD, XAUUSD, SILVER, CRUDEOIL | Crypto: BTCUSD, ETHUSD | Forex: EURUSD, USDINR")
    market = st.selectbox("Market", ["NSE (भारत)", "BSE (भारत)", "US"],
                          help="Commodity/Crypto/Forex साठी कोणतंही चालेल")
    auto = st.toggle("Auto refresh (live)", value=True)
    every = st.slider("Refresh दर (सेकंद)", 15, 120, 30, disabled=not auto)
    typed_key = st.text_input("Anthropic API key (AI explanation साठी)", type="password")

symbol = resolve_symbol(name, market)


def analyze(tf_label: str):
    interval, period, resample = TIMEFRAMES[tf_label]
    raw = load_data(symbol, interval, period, resample)
    if raw is None or raw.empty or len(raw) < 30:
        return None, None
    df = add_indicators(raw)
    return df, make_signal(df)


@st.fragment(run_every=every if auto else None)
def dashboard():
    results = {}
    for tf_label in TIMEFRAMES:
        try:
            results[tf_label] = analyze(tf_label)
        except Exception:
            results[tf_label] = (None, None)

    if all(sig is None for _, sig in results.values()):
        st.warning(f"'{symbol}' साठी डेटा मिळाला नाही. नाव तपासा. "
                   "(Stock/Forex/Gold चं मार्केट बंद असेल तर डेटा कमी येऊ शकतो.)")
        return

    # ---- तिन्ही timeframe चा सारांश
    st.subheader(f"🎯 {symbol} : Multi-timeframe Signals")
    icons = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    cols = st.columns(len(TIMEFRAMES))
    actions = []
    for col, (tf_label, (df, sig)) in zip(cols, results.items()):
        if sig is None:
            col.metric(tf_label, "डेटा नाही")
            continue
        actions.append(sig["action"])
        strong = " (मजबूत)" if abs(sig["score"]) >= 5 else ""
        col.metric(tf_label, f"{icons[sig['action']]} {sig['action']}{strong}",
                   f"score {sig['score']:+d} | भाव {sig['close']:.2f}")

    if len(actions) == len(TIMEFRAMES):
        if all(a == "BUY" for a in actions):
            st.success("✅ तिन्ही timeframe वर BUY: सहमती जास्त, signal अधिक विश्वासार्ह.")
        elif all(a == "SELL" for a in actions):
            st.error("✅ तिन्ही timeframe वर SELL: सहमती जास्त, signal अधिक विश्वासार्ह.")
        else:
            st.warning("⚠️ Timeframes मध्ये एकमत नाही. मोठ्या timeframe (4 तास/1 तास) ची दिशा बघा आणि "
                       "5 मिनिटाचा signal त्याच दिशेचा असेल तेव्हाच entry घ्या.")

    # ---- निवडलेल्या timeframe ची सविस्तर माहिती
    chosen = st.radio("📊 Chart आणि Entry/SL/Target कोणत्या timeframe चा?", list(TIMEFRAMES.keys()),
                      horizontal=True, index=0)
    df, sig = results[chosen]
    if sig is None:
        st.warning(f"{chosen} साठी पुरेसा डेटा मिळाला नाही.")
        return

    lv = sig["levels"]
    if lv:
        a, b, c = st.columns(3)
        a.metric("Entry", f"{lv['entry']:.2f}")
        b.metric("Stop-loss", f"{lv['sl']:.2f}")
        c.metric("Target", f"{lv['target']:.2f}", help="Risk:Reward = 1:2")
    else:
        st.info(f"{chosen}: HOLD, स्पष्ट संधी नाही. Support {sig['support']:.2f} | Resistance {sig['resist']:.2f}")

    st.plotly_chart(build_chart(df, sig, f"{symbol} • {chosen}"), use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False})

    with st.expander(f"📋 {chosen} signal ची कारणे", expanded=True):
        for r in sig["reasons"]:
            st.write(r)

    st.subheader("🤖 AI विश्लेषण (मराठी)")
    if st.button("AI ला समजावून सांगायला सांगा"):
        key = get_api_key(typed_key)
        if not key:
            st.warning("Sidebar मध्ये Anthropic API key टाका.")
        else:
            others = ", ".join(f"{k}: {v[1]['action']} ({v[1]['score']:+d})"
                               for k, v in results.items() if v[1] is not None)
            with st.spinner("विश्लेषण सुरू आहे..."):
                try:
                    st.session_state["ai_text"] = ai_explain(key, symbol, chosen, sig, others)
                except Exception as e:
                    st.session_state["ai_text"] = f"Error: {e}"
    if st.session_state.get("ai_text"):
        st.markdown(st.session_state["ai_text"])

    st.caption("⚠️ हे अॅप फक्त शिक्षण आणि निर्णयाला मदत यासाठी आहे. हा SEBI-registered आर्थिक सल्ला नाही. "
               "yfinance चा डेटा थोडा delayed असू शकतो. आधी paper trading करा.")


dashboard()
