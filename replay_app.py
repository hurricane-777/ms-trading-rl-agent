"""Offline Streamlit replay dashboard for Part D.

Loads a saved Part C checkpoint and replays it on the chosen evaluation window.
No live market connection and no real trades.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.preprocessing import StandardScaler

from run_partc_trial import (
    CFG,
    DEVICE,
    EPISODE_LEN,
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

MODELS = {
    "Double + Dueling DQN": (ROOT / "Double_Dueling_DQN.pt", True),
    "DQN": (ROOT / "DQN.pt", False),
}

PERIODS = {
    "2020-2022 validation": ("2020-01-01", "2022-12-31"),
    "2023-2024 test": ("2023-01-01", "2024-12-31"),
}


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
    "Offline replay dashboard · Local Streamlit demo · "
    "Loads a saved model and replays decisions · "
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
buy_idx = np.where(actions == 1)[0]
sell_idx = np.where(actions == 2)[0]

left, right = st.columns([1.7, 1.0])

with left:
    st.subheader("Stock price over time")
    st.caption("Buy and sell markers")
    fig1, ax1 = plt.subplots(figsize=(8.2, 3.2))
    ax1.plot(dates, out["prices"], color="black", linewidth=1.2)
    if len(buy_idx):
        ax1.scatter(dates[buy_idx], out["prices"][buy_idx], marker="^", s=28, color="black", label="Buy")
    if len(sell_idx):
        ax1.scatter(dates[sell_idx], out["prices"][sell_idx], marker="v", s=28, color="dimgray", label="Sell")
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_ylabel("Close")
    fig1.autofmt_xdate()
    fig1.tight_layout()
    st.pyplot(fig1, width="stretch")
    plt.close(fig1)

    st.subheader("Portfolio value over time")
    st.caption("Agent vs buy-and-hold")
    fig2, ax2 = plt.subplots(figsize=(8.2, 3.2))
    ax2.plot(dates, out["values"], color="black", linewidth=1.4, label="Agent")
    ax2.plot(dates, out["bh"], color="gray", linewidth=1.3, linestyle="--", label="Buy-and-hold")
    ax2.legend(frameon=False, loc="upper left")
    ax2.set_ylabel("Portfolio ($)")
    fig2.autofmt_xdate()
    fig2.tight_layout()
    st.pyplot(fig2, width="stretch")
    plt.close(fig2)

with right:
    st.subheader("Key results")
    s = out["stats"]
    st.metric("Total return", f"{100.0 * s['total_return']:.2f}%")
    st.metric("Sharpe ratio", f"{s['sharpe']:.3f}")
    st.metric("Maximum drawdown", f"{100.0 * s['max_drawdown']:.2f}%")
    st.metric("Number of trades", f"{s['n_trades']}")
    st.metric("Transaction costs", f"${s['transaction_costs']:.2f}")

    st.subheader("Current position")
    st.write(f"Cash: ${out['cash']:.2f}")
    st.write(f"Shares held: {int(out['shares'])}")
    st.write(f"Current action: {out['last_action']}")
    st.caption("Buy-and-hold return on this window: "
               f"{100.0 * out['bh_stats']['total_return']:.2f}%")
