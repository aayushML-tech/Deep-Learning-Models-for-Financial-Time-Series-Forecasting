# Deep Learning Models for Financial Time-Series Forecasting

Forecasting the **next-day closing price** of stocks using three deep learning architectures and comparing their performance on the same data pipeline:

1. **GRU-Transformer** — a GRU encoder feeding a Transformer encoder
2. **LSTM-Transformer** — an LSTM encoder feeding a Transformer encoder
3. **Transformer (standalone)** — a pure self-attention model with no recurrent layers

A single model is trained across **multiple stocks at once**. Each stock is given a learnable embedding, so one shared network can specialise its predictions per ticker while still learning patterns common to all of them.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. Nothing here is financial advice, and these predictions should not be used to make trading or investment decisions.

---

## Table of Contents

- [Key Features](#key-features)
- [Models](#models)
- [Data Pipeline](#data-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Key Features

- **Multi-stock training with stock embeddings** — one model handles all tickers; a learned `nn.Embedding` distinguishes each stock.
- **Technical-indicator feature engineering** — SMA, EMA, RSI, MACD, and Bollinger Bands added on top of OHLCV data.
- **Outlier removal** — daily returns are filtered with a rolling 252-day Z-score (|Z| ≤ 3) before training.
- **Reproducible** — fixed random seeds across NumPy and PyTorch (including CUDA/cuDNN deterministic mode).
- **Next-day prediction + visualization** — for each stock, the model plots 3 years of actual vs. predicted prices and marks the predicted next-day close.

## Models

All three models share the same input projection, positional encoding, and final linear head. They differ only in the encoder stage:

| Model | Architecture | Notebook / Script |
|-------|--------------|-------------------|
| **GRU-Transformer** | Input projection → 2-layer GRU (dropout 0.1) → positional encoding → Transformer encoder → linear head | GRU model |
| **LSTM-Transformer** | Input projection → 2-layer LSTM (dropout 0.1) → positional encoding → Transformer encoder → linear head | LSTM-Transformer model |
| **Transformer (standalone)** | Input projection → positional encoding → Transformer encoder → linear head (no recurrent layer) | Transformer standalone |

**Shared hyperparameters:**

- Model dimension: `128`
- Recurrent hidden size (GRU/LSTM models): `128`
- Attention heads: `8`
- Transformer encoder layers: `3`
- Stock embedding dimension: `32`
- Activation: GELU

> Note: each model uses an `input_proj` linear layer that concatenates the input features with the stock embedding before encoding. The standalone Transformer applies positional encoding directly to the projected input, while the hybrid models apply it to the recurrent layer's output.

## Data Pipeline

The same preprocessing is used across all three models:

- **Source:** Yahoo Finance via the `yfinance` library
- **Tickers:** `AAPL`, `MSFT`, `GOOGL`, `JPM`, `V`, `JNJ`, `TSLA`
- **Date range:** January 1, 2015 → today
- **Lookback window (`seq_len`):** 60 trading days
- **Target:** next-day scaled `Close` price
- **Base features:** Open, High, Low, Close, Volume
- **Engineered features:** `sma_10`, `ema_10`, `rsi` (14), `macd`, `bb_bbm` (Bollinger middle band), `day_of_week`
- **Outlier removal:** rows where the rolling 252-day Z-score of daily returns exceeds 3 (absolute) are dropped
- **Scaling:** `MinMaxScaler` (a separate close-price scaler is kept per stock for inverse-transforming predictions back to price)
- **Split:** sequences from all stocks are pooled, shuffled, and split 80% train / 20% test

## Evaluation Metrics

After training, each model is evaluated on the held-out test set (predictions are inverse-scaled to real prices first). The following are reported:

- R² (coefficient of determination)
- Explained Variance
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- SMAPE (Symmetric MAPE)

## Project Structure

```
Deep-Learning-Models-for-Financial-Time-Series-Forecasting/
├── GRU.ipynb                          # GRU-Transformer model
├── LSTM_Transformer.ipynb             # LSTM-Transformer model
├── Transformer standalone final.ipynb # Standalone Transformer model
├── requirements.txt
└── README.md
```

> Rename the entries above to match your actual notebook filenames if they differ.

## Installation

Clone the repository:

```bash
git clone https://github.com/aayushML-tech/Deep-Learning-Models-for-Financial-Time-Series-Forecasting.git
cd Deep-Learning-Models-for-Financial-Time-Series-Forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A suitable `requirements.txt` for this project:

```
yfinance
torch
numpy
pandas
scikit-learn
matplotlib
ta
```

## Usage

Each model lives in its own notebook. Launch Jupyter:

```bash
jupyter notebook
```

Open any of the three notebooks and run all cells in order. Each notebook will:

1. Download and preprocess data for all seven tickers
2. Train the model for 30 epochs
3. Print test-set evaluation metrics
4. Generate a next-day close prediction and a 3-year comparison plot for **every** trained stock

To predict a single stock instead of all of them, call the prediction helper directly (it's defined inside each notebook):

```python
predict_next_day("AAPL")
```

Only the seven tickers the model was trained on are supported; passing any other symbol returns an error message.

## Configuration

Common settings near the top of each script:

| Setting | Value | Where |
|---------|-------|-------|
| Tickers | 7 large-cap stocks | `tickers` list |
| Lookback window | 60 days | `seq_len` |
| Train/test split | 80 / 20 | `train_size` |
| Batch size | 64 | `DataLoader` |
| Optimizer | Adam, lr = 5e-4 | `optimizer` |
| LR scheduler | StepLR, gamma 0.5 (step 10 for GRU/LSTM, 15 for Transformer) | `scheduler` |
| Epochs | 30 | `epochs` |
| Loss | MSE | `criterion` |
| Gradient clipping | max-norm 1.0 | training loop |

To experiment, change the `tickers` list, adjust `seq_len`, or edit the model-construction arguments (`model_dim`, `n_heads`, `num_layers`, `embedding_dim`).

## Results

Fill in the metrics printed by each notebook's evaluation block to compare the models:

| Model | R² | Explained Var. | MAE | RMSE | MAPE | SMAPE |
|-------|-----|----------------|-----|------|------|-------|
| GRU-Transformer | — | — | — | — | — | — |
| LSTM-Transformer | — | — | — | — | — | — |
| Transformer (standalone) | — | — | — | — | — | — |

You can also embed a prediction plot, e.g.:

```
![AAPL prediction](results/aapl_prediction.png)
```

## Future Work

- Multi-step (multi-day horizon) forecasting rather than single next-day prediction
- Add macro/sentiment features alongside the technical indicators
- Hyperparameter tuning and an ensemble of the three encoders
- Walk-forward / time-based validation instead of a shuffled split, to better reflect real deployment

