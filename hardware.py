# HARDWARE TEST
# Used to identify hardware employed to train and evaluate model configurations. 
# This code displays information for CPU, RAM, and GPU.

#!/usr/bin/env python3

# Import Libraries
import platform, psutil, os, torch, time

start_total = time.time()
# CPU
print("CPU:", platform.processor())
print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical processors:", psutil.cpu_count(logical=True))

# RAM
ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"RAM: {ram_gb:.2f} GB")

# GPU
try:
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU Memory:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
    else:
        print("GPU: None")
except ImportError:
    print("PyTorch not installed; cannot detect GPU")

end_total = time.time()
hardware_time = end_total - start_total
print(f"Total hardware runtime: {hardware_time:.2f} sec")