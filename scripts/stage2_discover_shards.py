import json
import os
from huggingface_hub import hf_hub_download, HfApi

REPO_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
LOCAL_DIR = "models/qwen3-omni-partial"

def discover_shards():
    print(f"Downloading config.json and model.safetensors.index.json from {REPO_ID}...")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json", local_dir=LOCAL_DIR)
    index_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors.index.json", local_dir=LOCAL_DIR)
    
    print(f"Config downloaded to: {config_path}")
    print(f"Index downloaded to: {index_path}")
    
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
        
    weight_map = index_data.get("weight_map", {})
    
    code2wav_tensors = {k: v for k, v in weight_map.items() if k.startswith("code2wav.")}
    talker_tensors = {k: v for k, v in weight_map.items() if k.startswith("talker.")}
    
    code2wav_shards = sorted(list(set(code2wav_tensors.values())))
    talker_shards = sorted(list(set(talker_tensors.values())))
    
    # Query file sizes from Hugging Face API
    api = HfApi()
    try:
        repo_info = api.model_info(repo_id=REPO_ID, files_metadata=True)
        file_sizes = {sibling.rfilename: sibling.size for sibling in repo_info.siblings if sibling.size is not None}
    except Exception as e:
        print(f"Could not fetch remote file sizes: {e}")
        file_sizes = {}
        
    code2wav_shard_info = [
        {"filename": s, "size_bytes": file_sizes.get(s, 0), "size_mb": round(file_sizes.get(s, 0) / (1024**2), 2)}
        for s in code2wav_shards
    ]
    talker_shard_info = [
        {"filename": s, "size_bytes": file_sizes.get(s, 0), "size_mb": round(file_sizes.get(s, 0) / (1024**2), 2)}
        for s in talker_shards
    ]
    
    result = {
        "model_id": REPO_ID,
        "total_tensors": len(weight_map),
        "code2wav": {
            "tensor_count": len(code2wav_tensors),
            "shards": code2wav_shard_info,
            "shard_names": code2wav_shards,
            "tensors": list(code2wav_tensors.keys())
        },
        "talker": {
            "tensor_count": len(talker_tensors),
            "shards": talker_shard_info,
            "shard_names": talker_shards
        }
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/component_shards.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        
    print("\n" + "="*50)
    print(f"Total model tensors in index: {len(weight_map)}")
    print(f"Code2Wav tensors: {len(code2wav_tensors)}")
    print(f"Code2Wav required shards: {code2wav_shards}")
    for s in code2wav_shard_info:
        print(f"  - {s['filename']}: {s['size_mb']} MB ({s['size_bytes'] / (1024**3):.2f} GB)")
        
    print(f"\nTalker tensors: {len(talker_tensors)}")
    print(f"Talker required shards: {len(talker_shards)} shards -> {talker_shards}")
    total_talker_bytes = sum(s['size_bytes'] for s in talker_shard_info)
    print(f"Total Talker shards size: {total_talker_bytes / (1024**3):.2f} GB")
    print("="*50)
    
    print("\nDownloading tonight now:")
    print("Code2Wav only")
    for shard in code2wav_shards:
        print(f"Downloading Code2Wav shard: {shard}...")
        shard_path = hf_hub_download(repo_id=REPO_ID, filename=shard, local_dir=LOCAL_DIR)
        print(f"Downloaded to {shard_path}")
        
    print("\nDiscovery and Code2Wav shard download complete!")

if __name__ == "__main__":
    discover_shards()
