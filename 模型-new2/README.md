# Visual Servo Control with One-Frame Delay Compensation

## Project Structure

```
模型-new2/
├── prepare_data.py          # Data preparation with causal moving average
├── train.py                 # Training script
├── evaluate.py              # Evaluation metrics
├── fusion.py                # Fixed and adaptive fusion strategies
├── loss.py                  # Combined loss function
├── models/
│   ├── causal_cnn_gru.py   # Main model (Causal CNN + GRU)
│   ├── gru_baseline.py     # GRU baseline
│   ├── lstm_baseline.py    # LSTM baseline
│   └── mlp_baseline.py     # MLP baseline
├── configs/
│   ├── default.yaml        # Default configuration
│   └── ablation_w3.yaml    # Window size ablation
├── plot_curves.py          # Time series visualization
└── plot_results.py         # Results comparison plots
```

## Usage

1. Prepare data:
```python
from prepare_data import prepare_sequences, split_data, normalize_data
X, Y, e_ref = prepare_sequences(e_raw, window_size=2, history_len=8)
data_dict = split_data(X, Y)
data_dict, stats = normalize_data(data_dict)
```

2. Train model:
```bash
python train.py
```

3. Evaluate:
```bash
python evaluate.py
```

## Key Parameters

- `window_size`: 2 or 3 (causal moving average window)
- `history_len`: 8 (temporal history for prediction)
- `alpha`: 0.4 (fusion weight)
- `delta`: 10.0 (clipping threshold)

## Paper

Paper location: `paper/仪器与测量汇刊/new2/main.tex`

Compile with: `pdflatex main.tex`
