# M&S Trading — RL stock trading dashboard

Offline Streamlit app for UTS 43008 Assignment 3. It reads the Draft 2 final results (four variants, seeds 0, 1 and 2) from `final_results.json` and replays each run's selected checkpoint on the cached AAPL prices. It does not train, connect to a live market, or place real trades.

## Run locally

```
pip install -r requirements.txt
streamlit run replay_app.py
```

Then open http://localhost:8501

`final_results.json`, `aapl_ohlcv.csv`, `figures/`, and `runs/*_selected.pt` sit next to `replay_app.py`. To point at another copy of that folder, set `TRADING_AGENT_ROOT`.

`DQN.pt` and `Double_Dueling_DQN.pt` are the earlier Part C trial. The dashboard does not load them.

## Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Create app → this repository.
3. Main file: `replay_app.py`
4. Python version: 3.12 (or 3.11).
