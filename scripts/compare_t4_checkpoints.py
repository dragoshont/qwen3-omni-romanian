import os
import sys
import json
import shutil

sys.stdout.reconfigure(encoding="utf-8")

def compare_t4():
    print("=" * 70)
    print("COMPARING T3 CONTROL VS T4 CHECKPOINTS (500, 1000, 1500, 2000, 2500)")
    print("=" * 70)
    
    t3_quick40_path = "reports/scientific_gate_analysis.json"
    t4_history_path = "reports/t4_checkpoint_eval_history.json"
    
    if not os.path.exists(t3_quick40_path):
        print(f"Error: {t3_quick40_path} not found.")
        return
        
    with open(t3_quick40_path, "r", encoding="utf-8") as f:
        t3_data = json.load(f)["systems"]["B2"]
        
    t4_evals = []
    if os.path.exists(t4_history_path):
        with open(t4_history_path, "r", encoding="utf-8") as f:
            t4_evals = json.load(f)
            
    print(f"\nLoaded T3 Control Metrics and {len(t4_evals)} T4 Checkpoint Evaluations.\n")
    
    md = []
    md.append("# Phase T4 Data-Scaling Checkpoint Comparison")
    md.append("## Direct Comparison: T3 (1-Hour Control) vs T4 Checkpoints (5-Hour Corpus)\n")
    md.append("Evaluated under identical decoding parameters on 40 held-out sentences (`ro_holdout_quick_40.jsonl`).\n")
    
    headers = ["Metric", "T3 Control (Step 1000)"]
    for ev in t4_evals:
        headers.append(f"T4 Step {ev['step']}")
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join([":---"] * len(headers)) + " |")
    
    # Rows
    def row(name, t3_val, t4_key, fmt="{:.2f}%"):
        r = [f"**{name}**", t3_val]
        for ev in t4_evals:
            val = ev.get(t4_key, 0.0)
            if "cer" in t4_key or "wer" in t4_key:
                r.append(fmt.format(val * 100))
            elif "rate" in t4_key:
                r.append(f"{val:.1f}%")
            else:
                r.append(str(val))
        return "| " + " | ".join(r) + " |"
        
    md.append(row("Mean CER", f"{t3_data['mean_cer']*100:.2f}%", "mean_cer"))
    md.append(row("Median CER", f"{t3_data['median_cer']*100:.2f}%", "median_cer"))
    md.append(row("Mean WER", f"{t3_data['mean_wer']*100:.2f}%", "mean_wer"))
    md.append(row("Median WER", f"{t3_data['median_wer']*100:.2f}%", "median_wer"))
    md.append(row("EOS Success Rate", f"{t3_data['eos_success_rate']:.1f}%", "eos_rate"))
    md.append(row("Repetition Rate", f"{t3_data['repetition_loop_rate']:.1f}%", "repetition_rate"))
    md.append(row("Mean Duration (s)", f"{t3_data['mean_generated_duration_s']:.2f}s", "mean_duration_s", "{:.2f}s"))
    
    # Dual-metric evaluation:
    # 1. Best stable-mean checkpoint: lowest mean CER among zero-repetition checkpoints
    zero_rep_ckpts = [ev for ev in t4_evals if ev.get("repetition_rate", 0.0) == 0.0]
    if not zero_rep_ckpts:
        zero_rep_ckpts = t4_evals
        
    best_stable_mean_ckpt = min(zero_rep_ckpts, key=lambda x: x["mean_cer"]) if zero_rep_ckpts else None
    
    # 2. Best median-CER checkpoint: lowest median CER across all checkpoints
    best_median_cer_ckpt = min(t4_evals, key=lambda x: x["median_cer"]) if t4_evals else None
    
    # Check step 2500 domination
    step_2500_ckpt = next((ev for ev in t4_evals if ev["step"] == 2500), None)
    step_2500_dominates = False
    if step_2500_ckpt and best_stable_mean_ckpt and best_median_cer_ckpt:
        if (step_2500_ckpt["mean_cer"] < best_stable_mean_ckpt["mean_cer"] and 
            step_2500_ckpt["median_cer"] < best_median_cer_ckpt["median_cer"] and 
            step_2500_ckpt["repetition_rate"] == 0.0):
            step_2500_dominates = True
            
    checkpoints_to_eval_full200 = []
    if step_2500_dominates:
        checkpoints_to_eval_full200 = [2500]
    else:
        # Retain Step 1000 and Step 2000 as explicitly instructed
        steps_set = set()
        if best_stable_mean_ckpt:
            steps_set.add(best_stable_mean_ckpt["step"])
        if best_median_cer_ckpt:
            steps_set.add(best_median_cer_ckpt["step"])
        # Always ensure 1000 and 2000 are evaluated if present in t4_evals
        for step_val in [1000, 2000]:
            if any(ev["step"] == step_val for ev in t4_evals):
                steps_set.add(step_val)
        checkpoints_to_eval_full200 = sorted(list(steps_set))
        
    md.append(f"\n### Multi-Metric Checkpoint Analysis & Full-200 Evaluation Plan")
    if best_stable_mean_ckpt:
        md.append(f"- **Best Stable-Mean Checkpoint**: **Step {best_stable_mean_ckpt['step']}** (Mean CER: {best_stable_mean_ckpt['mean_cer']*100:.2f}%, Repetition: {best_stable_mean_ckpt['repetition_rate']}%)")
    if best_median_cer_ckpt:
        md.append(f"- **Best Median-CER Checkpoint**: **Step {best_median_cer_ckpt['step']}** (Median CER: {best_median_cer_ckpt['median_cer']*100:.2f}%)")
    md.append(f"- **Step 2500 Dominates Both**: {'YES' if step_2500_dominates else 'NO'}")
    md.append(f"- **Checkpoints Queued for Full-200 Benchmark**: {checkpoints_to_eval_full200}")
    
    table_str = "\n".join(md)
    print(table_str)
    
    with open("reports/t4_checkpoint_comparison.md", "w", encoding="utf-8") as f:
        f.write(table_str)
        
    selection_data = {
        "best_stable_mean_step": best_stable_mean_ckpt["step"] if best_stable_mean_ckpt else 1000,
        "best_median_cer_step": best_median_cer_ckpt["step"] if best_median_cer_ckpt else 2000,
        "step_2500_dominates": step_2500_dominates,
        "checkpoints_to_eval_full200": checkpoints_to_eval_full200,
        "preserve_steps": [1000, 2000, 2500]
    }
    
    with open("reports/t4_checkpoint_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection_data, f, indent=2)
        
    # Auto-preserve checkpoints to models/preserved_checkpoints/
    preserve_base = "models/preserved_checkpoints"
    os.makedirs(preserve_base, exist_ok=True)
    for s_step in selection_data["preserve_steps"]:
        src_dir = f"models/T4_data_scale_5h/final/checkpoint_step_{s_step}"
        if not os.path.exists(src_dir) and s_step == 2500:
            src_dir = "models/T4_data_scale_5h/final"
        if os.path.exists(src_dir):
            dst_dir = os.path.join(preserve_base, f"T4_checkpoint_step_{s_step}")
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print(f"Preserved checkpoint {s_step} to {dst_dir}")
            
    return selection_data

if __name__ == "__main__":
    compare_t4()
