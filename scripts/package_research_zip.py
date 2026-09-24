import os
import sys
import zipfile
import json
import hashlib
import datetime

sys.stdout.reconfigure(encoding="utf-8")

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def create_research_archive():
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    zip_filename = f"romanian_qwen_research_audit_{date_str}.zip"
    zip_filepath = os.path.abspath(zip_filename)
    
    print(f"Creating research audit archive: {zip_filename}...")
    
    # 1. Create a comprehensive checkpoint and model manifest before zipping
    models_manifest = []
    for root, dirs, files in os.walk("models"):
        for file in files:
            full_path = os.path.join(root, file)
            size_bytes = os.path.getsize(full_path)
            rel_path = os.path.relpath(full_path)
            models_manifest.append({
                "path": rel_path.replace("\\", "/"),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024**2), 2),
                "is_large_weight": size_bytes > 50 * 1024 * 1024,
            })
            
    with open("reports/models_and_checkpoints_manifest.json", "w", encoding="utf-8") as f:
        json.dump(models_manifest, f, indent=2)
    print("Generated reports/models_and_checkpoints_manifest.json")
    
    # Directories and files to include
    include_patterns = [
        "reports",
        "scripts",
        "eval",
        "data/manifests",
        "audit_snapshots",
        "outputs",
        "models",
    ]
    
    # Explicit files
    root_files = [
        "ZIP_CONTENTS.md",
        "master_audit_prompt.md",
    ]
    
    # Maximum file size to include (exclude base model safetensors > 30 MB)
    MAX_FILE_SIZE = 30 * 1024 * 1024 # 30 MB limit
    
    total_files_added = 0
    total_uncompressed_bytes = 0
    excluded_files = []
    
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add root documentation files
        for rf in root_files:
            if os.path.exists(rf):
                zipf.write(rf, arcname=rf)
                total_files_added += 1
                total_uncompressed_bytes += os.path.getsize(rf)
                print(f"Added: {rf}")
                
        # Traverse pattern directories
        for pattern_dir in include_patterns:
            if not os.path.exists(pattern_dir):
                continue
                
            for root, dirs, files in os.walk(pattern_dir):
                # Don't recurse into .git or large cache dirs
                if "__pycache__" in root or ".venv" in root:
                    continue
                    
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path)
                    arcname = rel_path.replace("\\", "/")
                    size = os.path.getsize(full_path)
                    
                    # Exclude large model weights
                    if size > MAX_FILE_SIZE:
                        excluded_files.append({
                            "path": arcname,
                            "size_bytes": size,
                            "size_mb": round(size / (1024**2), 2),
                            "reason": "Exceeds 30MB weight threshold (documented in manifest)",
                        })
                        continue
                        
                    zipf.write(full_path, arcname=arcname)
                    total_files_added += 1
                    total_uncompressed_bytes += size
                    
    # Save excluded manifest
    with open("reports/archive_excluded_weights_manifest.json", "w", encoding="utf-8") as f:
        json.dump(excluded_files, f, indent=2)
        
    zip_size_bytes = os.path.getsize(zip_filepath)
    zip_size_mb = zip_size_bytes / (1024**2)
    
    print("\n" + "=" * 70)
    print(f"ARCHIVE SUCCESSFULLY CREATED!")
    print(f"File Path: {zip_filepath}")
    print(f"Archive Size: {zip_size_mb:.2f} MB ({zip_size_bytes:,} bytes)")
    print(f"Total Files Included: {total_files_added}")
    print(f"Uncompressed Data Size: {total_uncompressed_bytes / (1024**2):.2f} MB")
    print(f"Large Base Models Excluded & Manifested: {len(excluded_files)} files")
    print("=" * 70)
    
    return zip_filepath, zip_size_mb

if __name__ == "__main__":
    create_research_archive()
