"""
Example: Backtesting a trained Lightning model
"""
import os
import torch

from backtesting import BacktestEngine
from backtesting.strategy import MLStrategy, SimpleMAStrategy, EMACrossStrategy, MACDStrategy, MAcrossover
from model import MInterface


# ============================================================
# 1. Model config (MUST match training)
# ============================================================
model_config = {
    "modelname": "transformer_encoder",
    "d_input": 5,
    "d_model": 32,
    "nhead": 2,
    "num_layers": 1,
    "dim_feedforward": 64,
    "dropout": 0.3,
    "num_classes": 3,
    "lr": 1e-3,
}

# ============================================================
# 2. Locate Lightning checkpoint
# ============================================================
CKPT_DIR = "lightning_logs/best_msft/checkpoints"

ckpt_files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".ckpt")]
if not ckpt_files:
    raise FileNotFoundError(f"No checkpoint found in {CKPT_DIR}")

ckpt_path = os.path.join(CKPT_DIR, sorted(ckpt_files)[-1])
print(f"[INFO] Loading checkpoint: {ckpt_path}")

# ============================================================
# 3. Build model via factory
# ============================================================
model = MInterface(model_config)

# ============================================================
# 4. Load Lightning checkpoint (state_dict only)
# ============================================================
ckpt = torch.load(ckpt_path, map_location="cpu")

# Lightning checkpoints always store weights here
model.load_state_dict(ckpt["state_dict"])
model.eval()

# ============================================================
# 5. Setup backtest engine
# ============================================================
engine = BacktestEngine(
    data_path="./data/data/msft/MSFT.csv",
    initial_cash=10000,
    commission=0.001,
    fromdate="2024-01-01",
    todate="2025-12-01",
)

# ============================================================
# 6. Add ML strategy
# ============================================================
# engine.add_strategy(
#     MLStrategy,
#     model=model,
#     lookback=60,     # MUST match sliding_window
#     hold_period=2,   # MUST match k
#     size=25,
#     threshold=0.0,   # optional
# )

# engine.add_strategy(
#     SimpleMAStrategy,
#     fast_period=10,
#     slow_period=30,
#     size=20,
# )

# engine.add_strategy(
#     MACDStrategy,
#     size=20,
# )
engine.add_strategy(
    MAcrossover,
)



# ============================================================
# 7. Run backtest
# ============================================================
results = engine.run()

print("\n" + "=" * 50)
print("BACKTEST RESULTS")
print("=" * 50)
for k, v in results.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

# ============================================================
# 8. Plot
# ============================================================
engine.plot(save_path="backtest_results.png")
