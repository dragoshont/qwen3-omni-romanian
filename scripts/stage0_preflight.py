import sys
import torch
import platform
import subprocess

def run_preflight():
    py_ver = sys.version
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if cuda_avail else "N/A"
    
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
    else:
        gpu_name = "N/A"
        capability = "N/A"
        vram_gb = 0

    print("=== STAGE 0 PREFLIGHT REPORT ===")
    print(f"Python version: {py_ver.split()[0]}")
    print(f"PyTorch version: {torch_ver}")
    print(f"CUDA runtime version: {cuda_ver}")
    print(f"GPU name: {gpu_name}")
    print(f"compute capability: {capability}")
    print(f"total VRAM: {vram_gb:.2f} GB ({vram_bytes} bytes)")
    
    # Run test tensor operation
    x = torch.randn(2000, 2000, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    print(f"BFloat16 Matrix Mul on CUDA: SUCCESS (shape={y.shape})")
    
    env_content = f"""OS: {platform.system()} {platform.release()} ({platform.version()})
Machine: {platform.machine()}
Python: {sys.version}
PyTorch: {torch_ver}
CUDA available: {cuda_avail}
CUDA version: {cuda_ver}
GPU: {gpu_name}
Compute capability: {capability}
Total VRAM: {vram_gb:.2f} GB ({vram_bytes} bytes)
Initial allocated VRAM: {torch.cuda.memory_allocated() / (1024**2):.2f} MB
Initial reserved VRAM: {torch.cuda.memory_reserved() / (1024**2):.2f} MB
"""
    with open("logs/environment.txt", "w", encoding="utf-8") as f:
        f.write(env_content)
    print("Saved logs/environment.txt")

if __name__ == "__main__":
    run_preflight()
