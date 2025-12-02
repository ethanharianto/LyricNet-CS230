import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available.")
        x = torch.ones(1, device=device)
        print(f"Tensor on MPS: {x}")
    else:
        print("MPS not available.")
except Exception as e:
    print(f"MPS Error: {e}")

from transformers import AutoTokenizer
print("Transformers imported.")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer loaded.")
