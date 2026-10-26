"""Replay one selected checkpoint on the price file that sits next to final_results.json.

The chart is a fresh rollout. Headline metrics stay in final_results.json.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from results_loader import artifact_root, selected_checkpoint
from run_partc_trial import (
    DEVICE,
    QNet,
    TradingEnv,
    build_market_features,
    greedy_policy,
    split_mask,
)

PERIODS = ("Validation", "Test")


def load_market(root: Path | None = None):
    root = root or artifact_root()
    csv_path = root / "aapl_ohlcv.csv"
    raw = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
    feats = build_market_features(raw)
    data = raw.join(feats).dropna()
    market_raw = data[feats.columns].to_numpy(dtype=np.float32)
    train_m = split_mask(data.index, "2010-01-01", "2019-12-31")
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(market_raw[train_m])
    market = np.clip(scaler.transform(market_raw).astype(np.float32), -10.0, 10.0)
    return data, market


def window(data: pd.DataFrame, start: str, end: str) -> tuple[int, int]:
    """Draft 2 horizon: the episode ends on the split's last day."""
    idxs = np.arange(len(data))[split_mask(data.index, start, end)]
    start_i = int(idxs[0])
    horizon = int(idxs[-1] - start_i)
    return start_i, horizon


def replay_selected(variant: str, seed: int, dueling: bool, start: str, end: str, root: Path | None = None):
    root = root or artifact_root()
    path = selected_checkpoint(variant, seed, root)
    if not path.is_file():
        raise FileNotFoundError(path)
    data, market = load_market(root)
    start_i, horizon = window(data, start, end)
    opens = data["Open"].to_numpy()
    closes = data["Close"].to_numpy()

    net = QNet(dueling).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state)
    policy = greedy_policy(net)

    env = TradingEnv(opens, closes, market, np.array([start_i]))
    obs, _ = env.reset(options={"start": start_i, "horizon": horizon})
    dates = [data.index[env.t]]
    prices = [float(closes[env.t])]
    values = [float(env.prev_value)]
    actions = []
    done = False
    while not done:
        action = int(policy(obs))
        obs, _, term, trunc, info = env.step(action)
        dates.append(data.index[env.t])
        prices.append(float(closes[env.t]))
        values.append(float(info["value"]))
        actions.append(action)
        done = term or trunc

    # Buy-and-hold on the same window, for the chart only.
    cash = 10_000.0
    fill = float(opens[start_i + 1])
    qty = int(cash / (fill * 1.001))
    fee = 0.001 * qty * fill
    cash -= qty * fill + fee
    bh = [10_000.0]
    for t in range(start_i + 1, start_i + 1 + len(actions)):
        bh.append(cash + qty * closes[t])

    return {
        "dates": pd.DatetimeIndex(dates),
        "prices": np.asarray(prices, dtype=np.float64),
        "values": np.asarray(values, dtype=np.float64),
        "buy_hold": np.asarray(bh, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.int64),
        "cash": float(env.cash),
        "shares": int(env.shares),
        "n_trades": int(env.n_trades),
        "costs": float(env.costs),
        "checkpoint": path,
    }
