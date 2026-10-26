"""Replay one saved checkpoint on the price file that sits next to final_results.json.

The chart is a fresh rollout of runs/*_selected.pt or runs/*_final.pt.
Headline metrics stay in final_results.json.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from results_loader import artifact_root, checkpoint_path
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


def replay_checkpoint(
    variant: str,
    seed: int,
    dueling: bool,
    start: str,
    end: str,
    weights: str = "selected",
    root: Path | None = None,
    initial_capital: float = 10_000.0,
    cost_rate: float = 0.001,
):
    root = root or artifact_root()
    path = checkpoint_path(variant, seed, weights, root)
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
    cash = float(initial_capital)
    fill = float(opens[start_i + 1])
    qty = int(cash / (fill * (1.0 + cost_rate)))
    fee = cost_rate * qty * fill
    cash -= qty * fill + fee
    bh = [float(initial_capital)]
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
