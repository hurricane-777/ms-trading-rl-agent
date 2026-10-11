# M&S Trading — RL stock trading dashboard

Offline Streamlit app for UTS 43008 Assignment 3. It loads a saved DQN checkpoint and replays buy / sell / hold decisions on AAPL. It does not connect to a live market or place real trades.

## Run locally

```
pip install -r requirements.txt
streamlit run replay_app.py
```

Then open http://localhost:8501

## Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Create app → this repository.
3. Main file: `replay_app.py`
4. Python version: 3.12 (or 3.11).

The 2023–2024 test split is available in the period dropdown but was not used to train or select the Part C checkpoints.
