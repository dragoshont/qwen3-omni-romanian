import os
import shutil

def preserve_checkpoints():
    dest_base = "models/preserved_checkpoints"
    os.makedirs(dest_base, exist_ok=True)
    
    t2_src = "models/T2_talker_only/final"
    t2_dst = os.path.join(dest_base, "T2_talker_only_step1000")
    if os.path.exists(t2_src):
        if os.path.exists(t2_dst):
            shutil.rmtree(t2_dst)
        shutil.copytree(t2_src, t2_dst, ignore=shutil.ignore_patterns("checkpoint_step_*"))
        print(f"Preserved T2 final adapter to {t2_dst}")
        
    t3_src = "models/T3_talker_mtp/final"
    t3_dst = os.path.join(dest_base, "T3_talker_mtp_step1000")
    if os.path.exists(t3_src):
        if os.path.exists(t3_dst):
            shutil.rmtree(t3_dst)
        shutil.copytree(t3_src, t3_dst, ignore=shutil.ignore_patterns("checkpoint_step_*"))
        print(f"Preserved T3 final adapter to {t3_dst}")
        
    manifest = {
        "timestamp": "2026-09-24T13:55:00",
        "T2": {
            "source": t2_src,
            "preserved_path": t2_dst,
            "description": "Talker-Only 4-bit LoRA (r=16, alpha=32) trained for 1000 steps on Romanian speech with explicit codec_eos_token_id supervision."
        },
        "T3": {
            "source": t3_src,
            "preserved_path": t3_dst,
            "description": "Joint Talker LoRA + MTP LoRA (r=16, alpha=32) trained for 1000 steps on Romanian speech."
        }
    }
    import json
    with open(os.path.join(dest_base, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Checkpoints preserved and manifest created.")

if __name__ == "__main__":
    preserve_checkpoints()
