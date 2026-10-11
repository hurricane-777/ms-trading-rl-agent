"""Offline Streamlit replay dashboard for Part D.

Loads a saved Part C checkpoint and replays it on the chosen evaluation window.
No live market connection and no real trades.
"""

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from run_partc_trial import (
    DEVICE,
    INITIAL_CAPITAL,
    QNet,
    ROOT,
    TradingEnv,
    build_market_features,
    download_aapl,
    greedy_policy,
    metrics_from_values,
    split_mask,
)

st.set_page_config(page_title="RL Trading Agent", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 0.6rem;}
    h1 {font-size: 1.55rem; margin-bottom: 0.15rem;}
    [data-testid="stHeader"] {background: transparent;}
    div[data-testid="stMetric"] {background: #f4f4f4; padding: 0.35rem 0.55rem; border-radius: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)

MODELS = {
    "Double + Dueling DQN": (ROOT / "Double_Dueling_DQN.pt", True),
    "DQN": (ROOT / "DQN.pt", False),
}

PERIODS = {
    "2020-2022 validation": ("2020-01-01", "2022-12-31"),
    "2023-2024 test": ("2023-01-01", "2024-12-31"),
}

# Chart shows a sample of trades so markers stay readable.
# The true count is still reported in Key results.
MAX_MARKERS = 20


def thin_markers(idx, max_n=MAX_MARKERS):
    if len(idx) <= max_n:
        return idx
    pick = np.linspace(0, len(idx) - 1, max_n).round().astype(int)
    return np.unique(idx[pick])


@st.cache_data(show_spinner="Loading market data...")
def load_dataset():
    raw = download_aapl()
    feats = build_market_features(raw)
    data = raw.join(feats).dropna()
    market_raw = data[feats.columns].to_numpy(dtype=np.float32)
    train_m = split_mask(data.index, "2010-01-01", "2019-12-31")
    scaler = StandardScaler()
    scaler.fit(market_raw[train_m])
    market = np.clip(scaler.transform(market_raw).astype(np.float32), -10.0, 10.0)
    return data, market


def buy_hold_curve(opens, closes, start, n_steps):
    cash = INITIAL_CAPITAL
    fill = float(opens[start + 1])
    qty = int(cash / (fill * 1.001))
    fee = 0.001 * qty * fill
    cash -= qty * fill + fee
    values = [INITIAL_CAPITAL]
    for t in range(start + 1, start + 1 + n_steps):
        values.append(cash + qty * closes[t])
    return np.asarray(values), 1 if qty else 0, fee


def replay(model_name, period_name):
    data, market = load_dataset()
    opens = data["Open"].to_numpy()
    closes = data["Close"].to_numpy()
    start_str, end_str = PERIODS[period_name]
    mask = split_mask(data.index, start_str, end_str)
    idxs = np.arange(len(data))[mask]
    start = int(idxs[0])
    horizon = int(idxs[-1] - start - 1)

    path, dueling = MODELS[model_name]
    net = QNet(dueling).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state)
    policy = greedy_policy(net)

    env = TradingEnv(opens, closes, market, np.array([start]))
    obs, _ = env.reset(options={"start": start, "horizon": horizon})
    dates = [data.index[env.t]]
    prices = [float(env.closes[env.t])]
    values = [env.prev_value]
    actions = [0]
    cash_hist = [env.cash]
    shares_hist = [env.shares]

    done = False
    while not done:
        action = policy(obs)
        obs, _, term, trunc, info = env.step(action)
        dates.append(data.index[env.t])
        prices.append(float(env.closes[env.t]))
        values.append(info["value"])
        actions.append(action)
        cash_hist.append(info["cash"])
        shares_hist.append(info["shares"])
        done = term or trunc

    values = np.asarray(values, dtype=np.float64)
    bh, bh_trades, bh_cost = buy_hold_curve(opens, closes, start, len(values) - 1)
    stats = metrics_from_values(values, env.n_trades, env.costs)
    bh_stats = metrics_from_values(bh, bh_trades, bh_cost)
    last_action = {0: "Hold", 1: "Buy", 2: "Sell"}[int(actions[-1])]
    return {
        "dates": pd.DatetimeIndex(dates),
        "prices": np.asarray(prices),
        "values": values,
        "bh": bh,
        "actions": np.asarray(actions),
        "stats": stats,
        "bh_stats": bh_stats,
        "cash": cash_hist[-1],
        "shares": shares_hist[-1],
        "last_action": last_action,
    }


st.title("RL Trading Agent")
st.caption(
    "Offline replay dashboard. Loads a saved model and replays decisions. "
    "Does not connect to live markets or place real trades."
)

c1, c2 = st.columns(2)
with c1:
    period_name = st.selectbox("Evaluation period", list(PERIODS.keys()), index=0)
with c2:
    model_name = st.selectbox("Saved model", list(MODELS.keys()), index=0)

out = replay(model_name, period_name)
dates = out["dates"]
actions = out["actions"]
buy_idx = thin_markers(np.where(actions == 1)[0])
sell_idx = thin_markers(np.where(actions == 2)[0])
n_buy = int((actions == 1).sum())
n_sell = int((actions == 2).sum())

left, right = st.columns([2.15, 0.95])

with left:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 5.0), sharex=True)

    ax1.set_title("Stock price over time", loc="left", fontsize=11, pad=4)
    ax1.plot(dates, out["prices"], color="black", linewidth=1.15)
    if len(buy_idx):
        ax1.scatter(dates[buy_idx], out["prices"][buy_idx], marker="^", s=22, color="black", label="Buy", zorder=3)
    if len(sell_idx):
        ax1.scatter(dates[sell_idx], out["prices"][sell_idx], marker="v", s=22, color="dimgray", label="Sell", zorder=3)
    ax1.legend(frameon=False, loc="upper left", fontsize=8)
    ax1.set_ylabel("Close")
    ax1.tick_params(labelsize=8)

    ax2.set_title("Portfolio value over time", loc="left", fontsize=11, pad=4)
    ax2.plot(dates, out["values"], color="black", linewidth=1.3, label="Agent")
    ax2.plot(dates, out["bh"], color="gray", linewidth=1.2, linestyle="--", label="Buy-and-hold")
    ax2.legend(frameon=False, loc="upper left", fontsize=8)
    ax2.set_ylabel("Portfolio ($)")
    ax2.tick_params(labelsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        f"Buy/sell markers are thinned to about {MAX_MARKERS} of each for readability "
        f"({n_buy} buys and {n_sell} sells actually occurred). "
        "The full trade count is in Key results."
    )

with right:
    st.subheader("Key results")
    s = out["stats"]
    m1, m2 = st.columns(2)
    m1.metric("Total return", f"{100.0 * s['total_return']:.2f}%")
    m2.metric("Sharpe", f"{s['sharpe']:.3f}")
    m3, m4 = st.columns(2)
    m3.metric("Max drawdown", f"{100.0 * s['max_drawdown']:.2f}%")
    m4.metric("Trades", f"{s['n_trades']}")
    st.metric("Transaction costs", f"${s['transaction_costs']:.2f}")

    st.subheader("Current position")
    st.write(f"Cash: ${out['cash']:.2f}")
    st.write(f"Shares held: {int(out['shares'])}")
    st.write(f"Last action: {out['last_action']}")
    st.caption(
        "Buy-and-hold return on this window: "
        f"{100.0 * out['bh_stats']['total_return']:.2f}%"
    )
