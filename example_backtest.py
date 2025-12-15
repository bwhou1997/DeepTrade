"""
Example: Backtesting a trained model
"""
import torch
from backtesting import BacktestEngine, MLStrategy
from model import MInterface

# Load trained model (example)
# model_config = {
#     "modelname": "lstm",
#     "d_input": 5,
#     "hidden_size": 64,
#     "num_layers": 2,
#     "num_classes": 3,
#     "lr": 1e-3,
# }
# model = MInterface(model_config)
# model.load_state_dict(torch.load("checkpoints/best_model.ckpt"))

# For demonstration, create a dummy model
# In practice, load your trained model here
print("Note: Load your trained model here")
print("Example: model.load_state_dict(torch.load('checkpoints/best_model.ckpt'))")

# Setup backtest engine
engine = BacktestEngine(
    data_path="data/AAPL.csv",
    initial_cash=10000,
    commission=0.001,
    fromdate="2020-01-01",
    todate="2023-12-31"
)

# Create strategy with model
# strategy = MLStrategy(
#     model=model,
#     lookback=60,
#     size=1,
#     threshold=0.6  # Only trade if confidence > 60%
# )

# For demonstration, use SimpleMAStrategy instead
from backtesting.strategy import SimpleMAStrategy
strategy = SimpleMAStrategy(fast_period=10, slow_period=30)

# Add strategy and run
engine.add_strategy(strategy)
results = engine.run()

# Print results
print("\n" + "="*50)
print("BACKTEST RESULTS")
print("="*50)
for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Plot results
engine.plot(save_path="backtest_results.png")


