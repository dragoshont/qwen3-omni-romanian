from huggingface_hub import hf_hub_download
import os

REPO_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
LOCAL_DIR = "models/qwen3-omni-partial"

shards_to_download = [
    "model-00013-of-00015.safetensors",
    "model-00014-of-00015.safetensors"
]

print("=== STAGE 9: Downloading Talker/MTP shards ===")
for shard in shards_to_download:
    dest_path = os.path.join(LOCAL_DIR, shard)
    if os.path.exists(dest_path):
        print(f"Shard {shard} already exists locally ({os.path.getsize(dest_path)/(1024**3):.2f} GB)")
    else:
        print(f"Downloading {shard} into {LOCAL_DIR}...")
        hf_hub_download(repo_id=REPO_ID, filename=shard, local_dir=LOCAL_DIR)
        print(f"Successfully downloaded {shard} ({os.path.getsize(dest_path)/(1024**3):.2f} GB)")
        
print("All Talker/MTP shards downloaded!")
