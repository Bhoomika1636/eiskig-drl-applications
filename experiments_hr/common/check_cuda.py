import torch

if torch.cuda.is_available():
    print("[info] CUDA is available.")
    print("[info] GPU used:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("[info] CUDA is not available or not installed.")
