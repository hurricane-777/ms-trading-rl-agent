"""Part C trial: AAPL trading env + DQN vs Double+Dueling DQN.

Validation only. The 2023-2024 test split is loaded and then left unused.

Sections
1. Trial settings and constants
2. Download AAPL and build the 10 market features
3. Trading environment (fill at next open)
4. DQN / Double+Dueling networks, replay, training
5. Baselines and the trial run
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

# Works as a .py file or inside a notebook (%run)
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CACHE = ROOT / "aapl_ohlcv.csv"
RESULTS = ROOT / "partc_trial_results.json"
DEVICE = torch.device("cpu")  # tiny MLP; GPU is not helpful here

# --- 1. Trial settings (Part B values, smaller step budget for Part C) ---
INITIAL_CAPITAL = 10_000.0
LOT_NOTIONAL = 1_000.0       # one lot ~ $1000 of stock, whole shares only
COST_RATE = 0.001            # 0.1% cost on buys and sells
EPISODE_LEN = 252            # one trading year
INVALID_PENALTY = -0.01      # if a buy/sell cannot be filled
OBS_DIM = 14                 # 10 market + 4 account features
N_ACTIONS = 3                # 0=hold, 1=buy, 2=sell


@dataclass
class TrialConfig:
    ticker: str = "AAPL"
    train_years: str = "2010-2019"
    val_years: str = "2020-2022"
    test_years: str = "2023-2024 (unused in this trial)"
    steps: int = 50_000
    seed: int = 42
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    replay_size: int = 100_000
    warmup: int = 5_000
    tau: float = 0.005
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 25_000
    hidden: int = 128
    eval_every_episodes: int = 50
    random_baseline_seeds: int = 100


CFG = TrialConfig()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --- 2. Price data and indicators ---
def download_aapl() -> pd.DataFrame:
    if CACHE.exists():
        df = pd.read_csv(CACHE, parse_dates=["Date"], index_col="Date")
        return df.sort_index()
    try:
        import yfinance as yf

        df = yf.download(
            "AAPL",
            start="2010-01-01",
            end="2025-01-01",
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        df = pd.read_csv(
            "https://stooq.com/q/d/l/?s=aapl.us&i=d",
            parse_dates=["Date"],
            index_col="Date",
        ).sort_index()
        df = df.loc["2010-01-01":"2024-12-31", ["Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna()
    df.to_csv(CACHE)
    return df


def rsi_14(close: pd.Series) -> pd.Series:
    # RSI(14) scaled to [0, 1]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))) / 100.0


def macd_pair(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal


def build_market_features(df: pd.DataFrame) -> pd.DataFrame:
    # Stationary market features (no raw price level in the state)
    close = df["Close"]
    volume = df["Volume"].astype(float)
    log_ret_1 = np.log(close / close.shift(1))
    log_ret_5 = np.log(close / close.shift(5))
    sma10_ratio = close.rolling(10).mean() / close
    sma30_ratio = close.rolling(30).mean() / close
    rsi = rsi_14(close)
    macd_line, macd_signal = macd_pair(close)
    macd_line_n = macd_line / close
    macd_signal_n = macd_signal / close
    vol20 = log_ret_1.rolling(20).std()
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    vol_z = (volume - vol_mean) / vol_std.replace(0.0, np.nan)
    close_max60 = close / close.rolling(60).max()
    feats = pd.DataFrame(
        {
            "log_ret_1": log_ret_1,
            "log_ret_5": log_ret_5,
            "sma10_ratio": sma10_ratio,
            "sma30_ratio": sma30_ratio,
            "rsi14": rsi,
            "macd_line_n": macd_line_n,
            "macd_signal_n": macd_signal_n,
            "vol20": vol20,
            "vol_z": vol_z,
            "close_max60": close_max60,
        },
        index=df.index,
    )
    return feats


def split_mask(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return (index >= start) & (index <= end)


# --- 3. Environment: decide at close t, fill at open t+1 ---
def lot_shares(price: float) -> int:
    if price <= 0:
        return 0
    return int(LOT_NOTIONAL / price)


def execute(action: int, cash: float, shares: int, avg_cost: float, fill: float):
    # Invalid buy/sell is treated as hold plus INVALID_PENALTY
    penalty = 0.0
    cost_paid = 0.0
    traded = False
    if action == 1:
        qty = lot_shares(fill)
        if qty <= 0:
            penalty = INVALID_PENALTY
        else:
            qty_afford = int(cash / (fill * (1.0 + COST_RATE)))
            if qty_afford <= 0:
                penalty = INVALID_PENALTY
                qty = 0
            else:
                qty = min(qty, qty_afford)
            if qty > 0:
                notional = qty * fill
                fee = COST_RATE * notional
                avg_cost = (avg_cost * shares + fill * qty) / (shares + qty)
                shares += qty
                cash -= notional + fee
                cost_paid = fee
                traded = True
    elif action == 2:
        if shares <= 0:
            penalty = INVALID_PENALTY
        else:
            qty = min(lot_shares(fill), shares)
            if qty <= 0:
                penalty = INVALID_PENALTY
            else:
                notional = qty * fill
                fee = COST_RATE * notional
                cash += notional - fee
                shares -= qty
                if shares == 0:
                    avg_cost = 0.0
                cost_paid = fee
                traded = True
    cash = max(cash, 0.0)
    return cash, shares, avg_cost, penalty, cost_paid, traded


class TradingEnv(gym.Env):
    # Observation: 14 floats. Action: hold / buy one lot / sell one lot.
    metadata = {"render_modes": []}

    def __init__(self, opens, closes, market, start_idxs, episode_len=EPISODE_LEN):
        super().__init__()
        self.opens = np.asarray(opens, dtype=np.float64)
        self.closes = np.asarray(closes, dtype=np.float64)
        self.market = np.asarray(market, dtype=np.float32)
        self.start_idxs = np.asarray(start_idxs, dtype=np.int64)
        self.episode_len = episode_len
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.t = 0
        self.steps_left = 0
        self.cash = INITIAL_CAPITAL
        self.shares = 0
        self.avg_cost = 0.0
        self.prev_value = INITIAL_CAPITAL
        self.costs = 0.0
        self.n_trades = 0
        np.random.default_rng(0)

    def _value(self, idx: int) -> float:
        return float(self.cash + self.shares * self.closes[idx])

    def _obs(self, idx: int) -> np.ndarray:
        v = self._value(idx)
        pos = self.shares * self.closes[idx]
        unreal = (self.closes[idx] - self.avg_cost) * self.shares if self.shares else 0.0
        acct = np.array(
            [
                self.cash / INITIAL_CAPITAL,
                pos / v if v > 0 else 0.0,
                pos / INITIAL_CAPITAL,
                unreal / INITIAL_CAPITAL,
            ],
            dtype=np.float32,
        )
        x = np.concatenate([self.market[idx], acct])
        return np.clip(x, -10.0, 10.0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start = int(self.start_idxs[self.np_random.integers(0, len(self.start_idxs))])
        if options and "start" in options:
            start = int(options["start"])
        self.t = start
        self.steps_left = self.episode_len
        if options and "horizon" in options:
            self.steps_left = int(options["horizon"])
        self.cash = INITIAL_CAPITAL
        self.shares = 0
        self.avg_cost = 0.0
        self.costs = 0.0
        self.n_trades = 0
        self.prev_value = self._value(self.t)
        return self._obs(self.t), {}

    def step(self, action):
        action = int(action)
        fill = float(self.opens[self.t + 1])
        self.cash, self.shares, self.avg_cost, penalty, fee, traded = execute(
            action, self.cash, self.shares, self.avg_cost, fill
        )
        self.costs += fee
        if traded:
            self.n_trades += 1
        self.t += 1
        self.steps_left -= 1
        value = self._value(self.t)
        ret = (value - self.prev_value) / max(self.prev_value, 1e-8)
        reward = float(np.clip(100.0 * ret, -10.0, 10.0) + penalty)
        self.prev_value = value
        terminated = False
        truncated = self.steps_left <= 0 or self.t >= len(self.closes) - 1
        info = {
            "value": value,
            "cash": self.cash,
            "shares": self.shares,
            "costs": self.costs,
            "n_trades": self.n_trades,
        }
        return self._obs(self.t), reward, terminated, truncated, info


# --- 4. DQN (optional Double + dueling head) ---
class QNet(nn.Module):
    def __init__(self, dueling: bool):
        super().__init__()
        self.dueling = dueling
        self.fc1 = nn.Linear(OBS_DIM, CFG.hidden)
        self.fc2 = nn.Linear(CFG.hidden, CFG.hidden)
        if dueling:
            self.v = nn.Linear(CFG.hidden, 1)
            self.a = nn.Linear(CFG.hidden, N_ACTIONS)
        else:
            self.q = nn.Linear(CFG.hidden, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.dueling:
            v = self.v(x)
            a = self.a(x)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.q(x)


class Replay:
    # Stores past transitions so the agent can learn from them in a random order
    def __init__(self, cap: int):
        self.s = np.zeros((cap, OBS_DIM), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.ns = np.zeros((cap, OBS_DIM), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.cap = cap
        self.i = 0
        self.n = 0

    def add(self, s, a, r, ns, done):
        self.s[self.i] = s
        self.a[self.i] = a
        self.r[self.i] = r
        self.ns[self.i] = ns
        self.d[self.i] = done
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, size=batch)
        return (
            torch.from_numpy(self.s[idx]).to(DEVICE),
            torch.from_numpy(self.a[idx]).to(DEVICE),
            torch.from_numpy(self.r[idx]).to(DEVICE),
            torch.from_numpy(self.ns[idx]).to(DEVICE),
            torch.from_numpy(self.d[idx]).to(DEVICE),
        )


def metrics_from_values(values: np.ndarray, n_trades: int, costs: float) -> dict:
    # Equity curve -> return, Sharpe, max drawdown, trades, costs
    values = np.asarray(values, dtype=np.float64)
    rets = np.diff(values) / np.clip(values[:-1], 1e-8, None)
    total_return = float(values[-1] / values[0] - 1.0)
    if rets.std() < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float(rets.mean() / rets.std() * math.sqrt(252.0))
    peak = np.maximum.accumulate(values)
    dd = values / np.clip(peak, 1e-8, None) - 1.0
    max_dd = float(dd.min())
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": int(n_trades),
        "transaction_costs": float(costs),
        "final_value": float(values[-1]),
    }


def rollout(env: TradingEnv, policy_fn, start: int, horizon: int) -> dict:
    obs, _ = env.reset(options={"start": start, "horizon": horizon})
    values = [env.prev_value]
    done = False
    while not done:
        a = policy_fn(obs)
        obs, _, term, trunc, info = env.step(a)
        values.append(info["value"])
        done = term or trunc
    return metrics_from_values(np.asarray(values), env.n_trades, env.costs)


def greedy_policy(net: QNet):
    # Evaluation policy: always take the highest Q-value
    net.eval()

    def _fn(obs):
        with torch.no_grad():
            q = net(torch.from_numpy(obs).unsqueeze(0).to(DEVICE))
            return int(q.argmax(dim=1).item())

    return _fn


def train_agent(env_train, env_val, val_start, val_horizon, double: bool, dueling: bool, label: str):
    # Train one agent. If double=True, use Double DQN targets. If dueling=True, use a dueling head.
    set_seed(CFG.seed)
    online = QNet(dueling).to(DEVICE)
    target = QNet(dueling).to(DEVICE)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=CFG.lr)
    replay = Replay(CFG.replay_size)
    obs, _ = env_train.reset(seed=CFG.seed)
    eps = CFG.eps_start
    best_sharpe = -1e9
    best_state = {k: v.cpu().clone() for k, v in online.state_dict().items()}
    episodes = 0
    ep_steps = 0

    for step in trange(CFG.steps, desc=label):
        if np.random.rand() < eps:
            action = int(np.random.randint(0, N_ACTIONS))
        else:
            action = greedy_policy(online)(obs)
        nxt, reward, term, trunc, _ = env_train.step(action)
        done = float(term or trunc)
        replay.add(obs, action, reward, nxt, done)
        obs = nxt
        ep_steps += 1
        if done:
            obs, _ = env_train.reset()
            episodes += 1
            ep_steps = 0
            if episodes % CFG.eval_every_episodes == 0:
                stats = rollout(env_val, greedy_policy(online), val_start, val_horizon)
                if stats["sharpe"] > best_sharpe:
                    best_sharpe = stats["sharpe"]
                    best_state = {k: v.cpu().clone() for k, v in online.state_dict().items()}

        frac = min(1.0, step / max(CFG.eps_decay_steps, 1))
        eps = CFG.eps_start + frac * (CFG.eps_end - CFG.eps_start)

        if replay.n >= CFG.warmup:
            s, a, r, ns, d = replay.sample(CFG.batch_size)
            with torch.no_grad():
                if double:
                    # online net chooses the action, target net scores it
                    na = online(ns).argmax(dim=1)
                    q_next = target(ns).gather(1, na.unsqueeze(1)).squeeze(1)
                else:
                    q_next = target(ns).max(dim=1).values
                y = r + CFG.gamma * (1.0 - d) * q_next
            q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q, y)
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(online.parameters(), 10.0)
            opt.step()
            with torch.no_grad():
                for tp, p in zip(target.parameters(), online.parameters()):
                    tp.data.mul_(1.0 - CFG.tau).add_(CFG.tau * p.data)

    online.load_state_dict(best_state)
    final = rollout(env_val, greedy_policy(online), val_start, val_horizon)
    torch.save(best_state, ROOT / f"{label.replace(' ', '_').replace('+', '')}.pt")
    final["best_val_sharpe_during_train"] = float(best_sharpe if best_sharpe > -1e8 else final["sharpe"])
    return final


# --- 5. Baselines on the same validation window ---
def baseline_buy_hold(opens, closes, start, horizon):
    cash = INITIAL_CAPITAL
    fill = float(opens[start + 1])
    qty = int(cash / (fill * (1.0 + COST_RATE)))
    fee = COST_RATE * qty * fill
    cash -= qty * fill + fee
    shares = qty
    values = [INITIAL_CAPITAL]
    for t in range(start + 1, start + 1 + horizon):
        values.append(cash + shares * closes[t])
    return metrics_from_values(np.asarray(values), 1 if qty else 0, fee)


def baseline_all_cash(horizon):
    values = np.full(horizon + 1, INITIAL_CAPITAL)
    return metrics_from_values(values, 0, 0.0)


def baseline_random(env_val, start, horizon, n_seeds: int):
    rows = []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)

        def policy(_obs, rng=rng):
            return int(rng.integers(0, N_ACTIONS))

        rows.append(rollout(env_val, policy, start, horizon))
    keys = ["total_return", "sharpe", "max_drawdown", "n_trades", "transaction_costs", "final_value"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["n_seeds"] = n_seeds
    return out


def fmt_pct(x):
    return f"{100.0 * x:.2f}%"


def fmt_num(x):
    return f"{x:.3f}"


def main():
    # Load data, train both agents, evaluate on validation only, write JSON
    set_seed(CFG.seed)
    raw = download_aapl()
    feats = build_market_features(raw)
    data = raw.join(feats).dropna()
    opens = data["Open"].to_numpy()
    closes = data["Close"].to_numpy()
    market_raw = data[feats.columns].to_numpy(dtype=np.float32)
    idx = data.index

    train_m = split_mask(idx, "2010-01-01", "2019-12-31")
    val_m = split_mask(idx, "2020-01-01", "2022-12-31")
    test_m = split_mask(idx, "2023-01-01", "2024-12-31")
    assert test_m.sum() > 0, "test rows exist but are not used"

    # Normalisation stats from the training split only
    scaler = StandardScaler()
    scaler.fit(market_raw[train_m])
    market = scaler.transform(market_raw).astype(np.float32)
    market = np.clip(market, -10.0, 10.0)

    i_all = np.arange(len(data))
    train_idx = i_all[train_m]
    val_idx = i_all[val_m]
    train_starts = train_idx[train_idx + EPISODE_LEN + 1 < train_idx[-1]]
    val_start = int(val_idx[0])
    val_horizon = int(val_idx[-1] - val_start - 1)
    assert val_horizon > 100

    env_train = TradingEnv(opens, closes, market, train_starts)
    env_val = TradingEnv(opens, closes, market, np.array([val_start]))

    print("rows", len(data), "train starts", len(train_starts), "val horizon", val_horizon)

    dqn = train_agent(env_train, env_val, val_start, val_horizon, False, False, "DQN")
    ddqn = train_agent(env_train, env_val, val_start, val_horizon, True, True, "Double_Dueling_DQN")
    bnh = baseline_buy_hold(opens, closes, val_start, val_horizon)
    cash = baseline_all_cash(val_horizon)
    rnd = baseline_random(env_val, val_start, val_horizon, CFG.random_baseline_seeds)

    payload = {
        "config": asdict(CFG),
        "notes": {
            "split": "validation 2020-2022 only",
            "test_split": "loaded but not evaluated",
            "fill": "decision at close t, fill at open t+1",
            "device": "cpu",
        },
        "results": {
            "DQN": dqn,
            "Double + Dueling DQN": ddqn,
            "Buy-and-hold": bnh,
            "Random": rnd,
            "All-cash": cash,
        },
        "data_rows": {
            "train": int(train_m.sum()),
            "validation": int(val_m.sum()),
            "test_unused": int(test_m.sum()),
        },
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["results"], indent=2))
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
