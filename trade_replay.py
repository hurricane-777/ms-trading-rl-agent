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
    execute,
    greedy_policy,
    lot_shares,
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


def last_price_row(root: Path | None = None) -> pd.Series:
    """Last row of aapl_ohlcv.csv. This repo has no later bar and no live feed."""
    root = root or artifact_root()
    raw = pd.read_csv(root / "aapl_ohlcv.csv", parse_dates=["Date"]).sort_values("Date")
    if raw.empty:
        raise ValueError("aapl_ohlcv.csv has no rows")
    return raw.iloc[-1]


def apply_bar_action(action: int, cash: float, shares: int, close: float, avg_cost: float) -> dict:
    """Same fill rule as TradingEnv: one lot, whole shares, no short, no leverage.

    `close` is the price used for the lot. The training env fills at the next open;
    the price file has no bar after the last row, so the latest-bar page passes that close.
    """
    action = int(action)
    if action not in (0, 1, 2):
        raise ValueError(f"unknown action {action}")
    cash = float(cash)
    shares = int(shares)
    close = float(close)
    if cash < 0 or shares < 0:
        raise ValueError("cash and shares must be zero or positive")
    if close <= 0:
        raise ValueError("close must be positive")
    new_cash, new_shares, _, penalty, fee, traded = execute(
        action, cash, shares, float(avg_cost), close
    )
    if action == 1 and traded:
        qty = int(new_shares) - shares
    elif action == 2 and traded:
        qty = shares - int(new_shares)
    else:
        qty = 0
    return {
        "action": action,
        "traded": bool(traded),
        "qty": qty,
        "fee": float(fee),
        "penalty": float(penalty),
        "cash": float(new_cash),
        "shares": int(new_shares),
        "lot_shares": int(lot_shares(close)),
    }


def predict_latest_bar(
    variant: str,
    seed: int,
    dueling: bool,
    weights: str = "selected",
    cash: float = 10_000.0,
    shares: int = 0,
    root: Path | None = None,
) -> dict:
    """One greedy pass of a saved checkpoint on the last feature row.

    Account features match TradingEnv: cash and position are scaled by the
    training capital ($10,000), then clipped. When shares are already held,
    cost basis is this bar's close, so unrealized gain in the state is zero.
    """
    if float(cash) < 0 or int(shares) < 0:
        raise ValueError("cash and shares must be zero or positive")
    if abs(float(shares) - int(shares)) > 1e-9:
        raise ValueError("shares must be a whole number")
    shares = int(shares)
    root = root or artifact_root()
    path = checkpoint_path(variant, seed, weights, root)
    if not path.is_file():
        raise FileNotFoundError(path)

    data, market = load_market(root)
    csv_last = last_price_row(root)
    bar_day = pd.Timestamp(data.index[-1])
    csv_day = pd.Timestamp(csv_last["Date"])
    if bar_day.normalize() != csv_day.normalize():
        raise RuntimeError(
            f"feature row {bar_day.date()} is not the last csv row {csv_day.date()}"
        )

    idx = len(data) - 1
    close = float(data["Close"].iloc[idx])
    avg_cost = close if shares else 0.0
    env = TradingEnv(
        data["Open"].to_numpy(),
        data["Close"].to_numpy(),
        market,
        np.array([idx]),
    )
    env.cash = float(cash)
    env.shares = shares
    env.avg_cost = avg_cost
    obs = env._obs(idx)

    net = QNet(dueling).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state)
    action = int(greedy_policy(net)(obs))
    filled = apply_bar_action(action, cash, shares, close, avg_cost)
    filled["date"] = bar_day
    filled["close"] = close
    filled["open"] = float(data["Open"].iloc[idx])
    filled["checkpoint"] = path
    filled["cost_basis"] = avg_cost
    return filled
