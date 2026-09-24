import os
import sys
import json
import time
import hashlib
import subprocess
import shutil

sys.stdout.reconfigure(encoding="utf-8")

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def make_snapshot():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    snap_dir = os.path.join("audit_snapshots", timestamp)
    hashes_dir = os.path.join(snap_dir, "hashes")
    os.makedirs(hashes_dir, exist_ok=True)
    print(f"Creating master audit snapshot at: {snap_dir}")
    
    # 1. file_tree.txt
    print("Recording file tree...")
    with open(os.path.join(snap_dir, "file_tree.txt"), "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk("."):
            if ".venv" in root or ".git" in root or "__pycache__" in root:
                continue
            level = root.replace(".", "").count(os.sep)
            indent = " " * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = " " * 4 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")
                
    # 2. git_status.txt
    print("Recording git status...")
    with open(os.path.join(snap_dir, "git_status.txt"), "w", encoding="utf-8") as f:
        git_cmd = shutil.which("git")
        if git_cmd:
            try:
                res1 = subprocess.run([git_cmd, "status"], capture_output=True, text=True, check=False)
                res2 = subprocess.run([git_cmd, "log", "-n", "5", "--oneline"], capture_output=True, text=True, check=False)
                f.write("=== GIT STATUS ===\n" + res1.stdout + "\n" + res1.stderr)
                f.write("\n=== GIT LOG ===\n" + res2.stdout + "\n" + res2.stderr)
            except Exception as e:
                f.write(f"Git execution notice: {e}\n")
        else:
            f.write("Git was not found on PATH.\n")
            
    # 3. environment.txt
    print("Recording environment...")
    with open(os.path.join(snap_dir, "environment.txt"), "w", encoding="utf-8") as f:
        f.write(f"Python: {sys.version}\n")
        f.write(f"Executable: {sys.executable}\n")
        f.write(f"Platform: {sys.platform}\n")
        try:
            import torch
            f.write(f"PyTorch: {torch.__version__}\n")
            f.write(f"CUDA available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA device: {torch.cuda.get_device_name(0)}\n")
                f.write(f"CUDA capability: {torch.cuda.get_device_capability(0)}\n")
                f.write(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")
        except Exception as e:
            f.write(f"Torch note: {e}\n")
            
    # 4. nvidia_smi.txt
    print("Recording nvidia-smi...")
    with open(os.path.join(snap_dir, "nvidia_smi.txt"), "w", encoding="utf-8") as f:
        try:
            res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            f.write(res.stdout + "\n" + res.stderr)
        except Exception as e:
            f.write(f"nvidia-smi note: {e}\n")
            
    # 5. checkpoints_manifest.json
    print("Cataloging and hashing checkpoints...")
    checkpoints = []
    checkpoint_roots = ["models/romanian_mtp_lora", "models/romanian_mtp_lora_ro150", "models/romanian_mtp_lora_ro500", "models/romanian_mtp_lora_cycle2_full15", "models/dry_run_lora_checkpoint"]
    for cr in checkpoint_roots:
        if not os.path.exists(cr):
            continue
        for root, dirs, files in os.walk(cr):
            for file in files:
                if file.endswith(".safetensors") or file.endswith(".bin") or file.endswith(".json"):
                    fp = os.path.join(root, file)
                    st = os.stat(fp)
                    sha = get_sha256(fp)
                    checkpoints.append({
                        "path": fp.replace("\\", "/"),
                        "size_bytes": st.st_size,
                        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                        "sha256": sha
                    })
                    
    with open(os.path.join(snap_dir, "checkpoints_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, indent=2)
        
    # 6. reports_manifest.json
    print("Cataloging reports and generated audio...")
    reports = []
    for root, dirs, files in os.walk("reports"):
        for file in files:
            fp = os.path.join(root, file)
            st = os.stat(fp)
            reports.append({
                "path": fp.replace("\\", "/"),
                "size_bytes": st.st_size,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                "sha256": get_sha256(fp)
            })
    with open(os.path.join(snap_dir, "reports_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
        
    print(f"Master snapshot complete at: {snap_dir}")
    return snap_dir

if __name__ == "__main__":
    make_snapshot()
