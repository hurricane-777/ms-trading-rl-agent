# M&S Trading — RL stock trading dashboard

Offline Streamlit app for UTS 43008 Assignment 3. It reads the Draft 2 final results (four variants, seeds 0, 1 and 2) from `final_results.json` and replays `runs/*_selected.pt` or `runs/*_final.pt` on the cached AAPL prices, using the same split and weights as the table. It does not train, connect to a live market, or place real trades.

## Run locally

```
pip install -r requirements.txt
streamlit run replay_app.py
```

Then open http://localhost:8501

The sidebar has two pages. **Replay app** is the trade chart and the stored tables. **Latest bar** runs one saved checkpoint on the last row of `aapl_ohlcv.csv` (there is no live feed; the date on that row is shown on the page). Pick a model, seed 0, 1 or 2, and selected or final weights, enter cash and Apple shares, then press **Run saved network**. The page shows Hold, Buy one lot, or Sell one lot, and whether that lot would be unfilled under the same cash and share rules as the training env. Under the action it shows that run's stored 2023–2024 test result from `final_results.json` next to buy-and-hold. The action is the saved model's output, not an order and not advice.

`final_results.json`, `aapl_ohlcv.csv`, `figures/`, `runs/*_selected.pt`, and `runs/*_final.pt` sit next to `replay_app.py`. To point at another copy of that folder, set `TRADING_AGENT_ROOT`.

`DQN.pt` and `Double_Dueling_DQN.pt` are the earlier Part C trial. The dashboard does not load them.

## Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Create app → this repository.
3. Main file: `replay_app.py`
4. Python version: 3.12 (or 3.11).
