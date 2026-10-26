"""Offline dashboard for the Draft 2 AAPL runs.

Metrics are read from final_results.json. The trade chart replays runs/*_selected.pt
on aapl_ohlcv.csv. It does not train, and it does not place trades.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from results_loader import (
    METRIC_KEYS,
    artifact_root,
    baseline_rows,
    figures,
    find_run,
    format_metric,
    load_results,
    metric_block,
    per_run_rows,
    selected_checkpoint,
    summary_rows,
)
from trade_replay import replay_selected

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

LABELS = {
    "total_return": "Total return",
    "sharpe": "Sharpe",
    "max_drawdown": "Max drawdown",
    "n_trades": "Trades",
    "transaction_costs": "Costs ($)",
    "final_value": "Final value ($)",
    "avg_exposure": "Avg. invested",
}
MAX_MARKERS = 20


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _show_metrics(metrics: dict) -> None:
    m1, m2 = st.columns(2)
    m1.metric("Total return", _pct(metrics["total_return"]))
    m2.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    m3, m4 = st.columns(2)
    m3.metric("Max drawdown", _pct(metrics["max_drawdown"]))
    m4.metric("Trades", f"{metrics['n_trades']:.0f}")
    m5, m6 = st.columns(2)
    m5.metric("Transaction costs", f"${metrics['transaction_costs']:,.0f}")
    m6.metric("Final value", f"${metrics['final_value']:,.2f}")
    st.caption(f"Average invested: {100.0 * metrics['avg_exposure']:.0f}% of the portfolio.")


def _present(rows: list[dict]) -> list[dict]:
    shown = []
    for row in rows:
        item = {}
        for key, value in row.items():
            item[key] = format_metric(key, value) if key in METRIC_KEYS else value
        shown.append(item)
    return shown


def _frame(rows: list[dict], formatted: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(_present(rows) if formatted else rows)
    return frame.rename(columns={key: LABELS.get(key, key) for key in frame.columns})


def thin_markers(idx, max_n=MAX_MARKERS):
    import numpy as np

    if len(idx) <= max_n:
        return idx
    pick = np.linspace(0, len(idx) - 1, max_n).round().astype(int)
    return np.unique(idx[pick])


root = artifact_root()
try:
    results = load_results(root / "final_results.json")
except (OSError, ValueError) as exc:
    st.error(f"Could not read final_results.json from {root}: {exc}")
    st.stop()

cfg = results["config"]
st.title("RL Trading Agent")
st.caption(
    "Draft 2 final results for four variants and seeds 0, 1 and 2. "
    f"Numbers are read from final_results.json ({results['data_source']}). "
    "No live market and no retraining."
)

weights_for = {
    "Train": ["selected"],
    "Validation": ["selected", "final"],
    "Test": ["selected", "final"],
}
left, right = st.columns(2)
split = left.selectbox("Split", list(weights_for), index=1)
weights = right.selectbox(
    "Weights",
    weights_for[split],
    format_func=lambda name: "Selected checkpoint" if name == "selected" else "Final weights",
)

per_run = per_run_rows(results, split, weights)
st.subheader(f"{split} · {weights} weights")
st.dataframe(_frame(per_run, formatted=True), width="stretch", hide_index=True)
st.caption(
    "Each row is one saved run. Total return, drawdown and average invested are fractions in the file "
    "and shown here as percents. Selection score is the validation Sharpe used to keep the checkpoint."
)

st.subheader("Mean ± sample std across the stored seeds")
st.dataframe(_frame(summary_rows(results, split, weights)), width="stretch", hide_index=True)

baselines = baseline_rows(results, split)
if baselines:
    st.subheader("Baselines from the same file")
    st.dataframe(_frame(baselines), width="stretch", hide_index=True)
    st.caption(
        "Buy-and-hold on the training split is one stored dict. "
        "Validation and test baselines are stored as lists; a list of one is shown as that row, "
        "and Random is the mean ± sample std of the 100 stored seeds."
    )

st.subheader("Trade view")
st.caption("The chart replays the selected checkpoint. The metrics beside it are the saved ones, not a new score.")
variants = list(results["variants"])
pick_left, pick_mid, pick_right = st.columns(3)
variant = pick_left.selectbox("Variant", variants, index=variants.index("Double + Dueling DQN"))
seed = int(pick_mid.selectbox("Seed", [int(s) for s in results["seeds"]]))
period = pick_right.selectbox("Chart period", ["Validation", "Test"])
period_start = cfg["val_start"] if period == "Validation" else cfg["test_start"]
period_end = cfg["val_end"] if period == "Validation" else cfg["test_end"]
run = find_run(results, variant, seed)
saved = metric_block(run, split, weights)
ckpt = selected_checkpoint(variant, seed, root)

chart_col, stat_col = st.columns([2.15, 0.95])
with stat_col:
    st.markdown(f"**{variant}, seed {seed}**")
    st.write(
        f"Selected at episode {run['selected_episode']} "
        f"(selection score {run['selection_score']:.4f}). "
        f"Trained for {run['episodes']} episodes, {run['minutes']:.1f} min."
    )
    st.markdown(f"Saved {split.lower()} metrics, {weights} weights")
    _show_metrics(saved)
    if not (split == period and weights == "selected"):
        st.caption(f"The chart is the selected checkpoint on {period} ({period_start} to {period_end}).")
    if not ckpt.is_file():
        st.warning(f"Selected checkpoint is not in the repo: {ckpt.name}")

with chart_col:
    if ckpt.is_file():
        try:
            out = replay_selected(
                variant,
                seed,
                bool(results["variants"][variant]["dueling"]),
                period_start,
                period_end,
                root,
            )
        except Exception as exc:
            st.error(f"Could not replay {ckpt.name}: {exc}")
        else:
            import numpy as np

            dates = out["dates"]
            actions = out["actions"]
            buy_idx = thin_markers(np.where(actions == 1)[0])
            sell_idx = thin_markers(np.where(actions == 2)[0])
            n_buy = int((actions == 1).sum())
            n_sell = int((actions == 2).sum())
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
            ax2.plot(dates, out["buy_hold"], color="gray", linewidth=1.2, linestyle="--", label="Buy-and-hold")
            ax2.legend(frameon=False, loc="upper left", fontsize=8)
            ax2.set_ylabel("Portfolio ($)")
            ax2.tick_params(labelsize=8)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption(
                f"Buy/sell markers are thinned to about {MAX_MARKERS} of each "
                f"({n_buy} buys and {n_sell} sells in this replay of {ckpt.name}). "
                f"End of replay: ${out['cash']:.2f} cash, {out['shares']} shares."
            )
    else:
        st.info("Add the selected checkpoint under runs/ to draw the trade chart.")

saved_figs = figures(root)
if saved_figs:
    st.subheader("Figures saved with the runs")
    for path in saved_figs:
        st.image(str(path), caption=path.name, width="stretch")
