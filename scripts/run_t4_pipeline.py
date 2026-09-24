import os
import sys
import json
import time
import subprocess
import datetime

sys.stdout.reconfigure(encoding="utf-8")

def log(msg):
    t_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)

def run_step(step_name, script_path, extra_args=None):
    log(f"============================================================")
    log(f"STARTING T4 STEP: {step_name}")
    log(f"Script: {script_path} {extra_args if extra_args else ''}")
    log(f"============================================================")
    
    t0 = time.time()
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)
        
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    
    if res.returncode != 0:
        log(f"ERROR: Step '{step_name}' failed with returncode {res.returncode} after {elapsed:.1f}s!")
        return False
    else:
        log(f"SUCCESS: Step '{step_name}' completed cleanly in {elapsed:.1f}s ({elapsed/60:.1f}m).\n")
        return True

def main():
    log("=== PHASE T4 MASTER AUTONOMOUS PIPELINE INITIALIZED ===")
    
    # 1. System Power Manager: Keep awake & Ultimate Performance
    log("Activating Ultimate Performance power scheme & anti-sleep guards...")
    subprocess.run([sys.executable, "scripts/system_power_manager.py"], check=False)
    
    # 2. Dataset Preparation (5 Hours, Phonetic Diversity, Strict Exclusion)
    mimi_codes_5h = "reports/mimi_codes_ro_5h.jsonl"
    if not os.path.exists(mimi_codes_5h):
        success = run_step("Prepare 5-Hour Dataset & RVQ Mimi Encoding", "scripts/prepare_t4_5h_dataset.py")
        if not success:
            log("Dataset preparation failed. Aborting pipeline.")
            return
    else:
        log("5-Hour dataset already prepared and encoded. Proceeding directly to training.")
        
    # 3. Phase T4 Training (Stock Base, Fresh LoRA r=8, alpha=16, 2,500 Steps, Checkpoint Evals)
    t4_summary = "reports/t4_training_summary.json"
    if not os.path.exists(t4_summary):
        success = run_step("Phase T4 Scaling Training (2,500 steps, Checkpoint Quick-40 Evals)", "scripts/train_phase_t4_scaling.py")
        if not success:
            log("T4 Training encountered an error. Aborting pipeline.")
            return
    else:
        log("T4 Training already complete.")
        
    # 4. Compare T3 Control vs T4 Checkpoints and Determine Evaluation Set
    run_step("Compare T3 Control vs T4 Intermediate Checkpoints", "scripts/compare_t4_checkpoints.py")
    
    # 5. Read Selection Manifest (Preserving Step 1000 and Step 2000, Evaluating Both if Different)
    eval_steps = [1000, 2000]
    selection_file = "reports/t4_checkpoint_selection.json"
    if os.path.exists(selection_file):
        with open(selection_file, "r", encoding="utf-8") as f:
            sel_data = json.load(f)
            eval_steps = sel_data.get("checkpoints_to_eval_full200", [1000, 2000])
            
    log(f"Checkpoints queued for Full-200 Held-Out Benchmark: {eval_steps}")
    
    # 6. Run Full 200 Evaluation on each candidate checkpoint
    evaluated_reports = {}
    for s_step in eval_steps:
        adapter_path = f"models/T4_data_scale_5h/final/checkpoint_step_{s_step}"
        if not os.path.exists(adapter_path) and s_step == 2500:
            adapter_path = "models/T4_data_scale_5h/final"
            
        report_file = f"reports/t4_step{s_step}_full200_benchmark.json"
        if not os.path.exists(report_file):
            run_step(
                f"Full 200 Held-Out Evaluation on Step {s_step}",
                "scripts/eval_full_200_champion.py",
                extra_args=[adapter_path, report_file]
            )
        evaluated_reports[s_step] = report_file
        
    # 7. Compile Final Dual Checkpoint Comparison & Declare T4 Champion
    run_step("Compile Final T4 Champion Declaration", "scripts/compile_t4_final_declaration.py")
    
    # 8. Package updated research archive
    run_step("Package Updated Research ZIP Archive", "scripts/package_research_zip.py")
    
    # 9. Restore System Power Settings before 9:30 PM
    log("Restoring Windows system power scheme to Balanced...")
    subprocess.run([sys.executable, "scripts/system_power_manager.py", "revert"], check=False)
    log("=== PHASE T4 MASTER PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
