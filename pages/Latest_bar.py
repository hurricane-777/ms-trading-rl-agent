"""One greedy action from a saved checkpoint on the last row of aapl_ohlcv.csv.

The 2023–2024 numbers are read from final_results.json. This page does not train,
does not call a market, and does not place an order.
"""

from __future__ import annotations

import streamlit as st

from results_loader import (
    artifact_root,
    find_run,
    format_metric,
    load_results,
    mean_std_text,
    metric_block,
)
from run_partc_trial import lot_shares
from trade_replay import last_price_row, predict_latest_bar

st.set_page_config(page_title="Latest bar", layout="wide")
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

ACTION_LABELS = {0: "Hold", 1: "Buy one lot", 2: "Sell one lot"}
WEIGHT_LABELS = {"selected": "Selected checkpoint", "final": "Final weights"}


def _usd(value: float) -> str:
    # Streamlit treats $...$ as math. A backslash keeps the dollar sign as text.
    return f"\\${value:,.2f}"


def _plain_dollars(text: str) -> str:
    return text.replace("$", "\\$")


def _fill_sentence(pred: dict) -> str:
    action = int(pred["action"])
    lot = int(pred["lot_shares"])
    qty = int(pred["qty"])
    price = _usd(float(pred["close"]))
    fee = _usd(float(pred["fee"]))
    cash_left = _usd(float(pred["cash"]))
    if action == 0:
        return (
            f"Hold leaves cash and shares unchanged. "
            f"One lot at this close is {lot} shares ({price})."
        )
    if action == 1 and not pred["traded"]:
        return (
            "Buy one lot is unfilled. Cash does not cover one share at this close plus the 0.1% cost. "
            "The trading rules do not borrow."
        )
    if action == 1 and qty < lot:
        return (
            f"Buy one lot is short of cash for {lot} shares, so only {qty} shares are bought at {price} "
            f"(cost {fee}). Cash would be {cash_left} and shares {pred['shares']}."
        )
    if action == 1:
        return (
            f"Buy one lot would take {qty} shares at {price} (cost {fee}). "
            f"Cash would be {cash_left} and shares {pred['shares']}."
        )
    if not pred["traded"]:
        return "Sell one lot is unfilled. The account has no shares. The trading rules do not short."
    if qty < lot:
        return (
            f"Sell one lot sells the {qty} shares on hand (a full lot is {lot} at {price}). "
            f"After the 0.1% cost, cash would be {cash_left}."
        )
    return (
        f"Sell one lot would sell {qty} shares at {price} (cost {fee}). "
        f"Cash would be {cash_left} and shares {pred['shares']}."
    )


def _profit(metrics: dict, capital: float) -> float:
    return float(metrics["final_value"]) - float(capital)


root = artifact_root()
try:
    results = load_results(root / "final_results.json")
    row = last_price_row(root)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Could not read the saved results or price file from {root}: {exc}")
    st.stop()

capital = float(results["environment"]["initial_capital"])
date_label = row["Date"].strftime("%Y-%m-%d")
close = float(row["Close"])
lot = lot_shares(close)
cfg = results["config"]

st.title("Latest bar")
st.markdown(
    f"Last row of `aapl_ohlcv.csv`: **{date_label}**. Close **{_usd(close)}**."
)
st.caption(
    f"No live market feed. One lot is about $1,000 in whole shares: {lot} shares at this close, "
    "before the 0.1% cost. The cash and shares check uses this close. "
    "In training, a decision on a bar fills at the next open, and this file ends on this date."
)

variants = list(results["variants"])
c1, c2, c3 = st.columns(3)
variant = c1.selectbox("Model", variants, index=variants.index("DQN") if "DQN" in variants else 0)
seed = int(c2.selectbox("Seed", [int(s) for s in results["seeds"]]))
weights = c3.selectbox("Weights", ["selected", "final"], format_func=lambda name: WEIGHT_LABELS[name])

a1, a2 = st.columns(2)
cash = float(a1.number_input("Cash ($)", min_value=0.0, value=10000.0, step=100.0, format="%.2f"))
shares = int(a2.number_input("Apple shares", min_value=0, value=0, step=1))
st.caption(
    "Shares already held use this bar's close as the cost basis, so unrealized gain in the network state is zero."
)

sig = (variant, seed, weights, round(cash, 2), shares)
if st.button("Run saved network", type="primary"):
    try:
        pred = predict_latest_bar(
            variant,
            seed,
            bool(results["variants"][variant]["dueling"]),
            weights,
            cash,
            shares,
            root,
        )
    except Exception as exc:
        st.session_state["bar_result"] = {"sig": sig, "error": f"Could not run the checkpoint: {exc}"}
    else:
        st.session_state["bar_result"] = {"sig": sig, "pred": pred}

saved = st.session_state.get("bar_result")
pred = None
if isinstance(saved, dict) and saved.get("sig") == sig:
    if "error" in saved:
        st.error(saved["error"])
    else:
        pred = saved["pred"]

if pred is None:
    st.caption("The button runs the saved network once on this bar.")
else:
    shown_day = pred["date"].strftime("%Y-%m-%d")
    st.subheader(ACTION_LABELS[int(pred["action"])])
    st.markdown(f"Action on **{shown_day}**.")
    st.write(_fill_sentence(pred))
    st.caption(f"Checkpoint `{pred['checkpoint'].name}`.")

run = find_run(results, variant, seed)
block = metric_block(run, "Test", weights)
bh_raw = results["baselines"]["test"]["Buy-and-hold"]
bh = bh_raw[0] if isinstance(bh_raw, list) else bh_raw
run_return = float(block["total_return"])
bh_return = float(bh["total_return"])
run_profit = _profit(block, capital)
bh_profit = _profit(bh, capital)

st.subheader(f"Stored test · {cfg['test_start']} to {cfg['test_end']}")
st.caption(
    "These figures are the saved backtest in final_results.json. "
    "Profit is the stored final value minus the starting capital. The button does not recompute them."
)
left, right = st.columns(2)
with left:
    st.markdown(f"**{variant}, seed {seed}, {WEIGHT_LABELS[weights].lower()}**")
    st.metric("Total return", format_metric("total_return", run_return))
    st.metric("Profit", f"${run_profit:,.2f}")
with right:
    st.markdown("**Buy-and-hold**")
    st.metric("Total return", format_metric("total_return", bh_return))
    st.metric("Profit", f"${bh_profit:,.2f}")

dqn_blocks = [
    metric_block(item, "Test", "selected")
    for item in results["runs"]
    if item["variant"] == "DQN"
]
dqn_return = mean_std_text([item["total_return"] for item in dqn_blocks], "total_return")
dqn_profit = mean_std_text([_profit(item, capital) for item in dqn_blocks], "final_value")
st.caption(
    f"DQN, selected checkpoints, mean of seeds {', '.join(str(int(s)) for s in results['seeds'])}: "
    f"{dqn_return} return, profit {_plain_dollars(dqn_profit)} on {_plain_dollars(f'${capital:,.0f}')}."
)

if run_return < bh_return:
    st.markdown(
        "This is the saved model's action, not an order and not advice, "
        "because it did not beat buy-and-hold."
    )
else:
    st.markdown("This is the saved model's action, not an order and not advice.")
