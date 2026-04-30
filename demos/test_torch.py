import sys
print("Python started", flush=True)
print(f"Python: {sys.executable}", flush=True)
print(f"Path: {sys.path[:3]}", flush=True)

print("Importing torch...", flush=True)
import torch
print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
