"""Root entrypoint. Real training logic lives in
src/hydra_torch/scripts/train.py so it's callable both manually
(`python main.py`) and from DVC (`python -m src.hydra_torch.scripts.train`)
without duplicating code.
"""
from src.hydra_torch.scripts.train import main as train_main

if __name__ == "__main__":
    train_main()