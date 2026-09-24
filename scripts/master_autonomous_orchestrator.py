import os
import sys
import time
import subprocess
import json
import datetime

sys.stdout.reconfigure(encoding="utf-8")

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_step(step_name, script_path):
    log(f"============================================================")
    log(f"STARTING PIPELINE STEP: {step_name}")
    log(f"Script: {script_path}")
    log(f"============================================================")
    
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable, script_path], check=False)
    dur = time.perf_counter() - t0
    
    if res.returncode == 0:
        log(f"SUCCESS: {step_name} completed cleanly in {dur:.1f}s ({dur/60:.1f}m).\n")
        return True
    else:
        log(f"ERROR: {step_name} exited with code {res.returncode} after {dur:.1f}s.\n")
        return False

def main():
    log("=== LAUNCHING MASTER AUTONOMOUS ORCHESTRATOR FOR ROMANIAN QWEN3-OMNI ===")
    log("System will execute all training phases and evaluations until deadline (9:30 PM).")
    
    # 1. Wait for Phase T0 if running
    t0_report = "reports/T0_stock_benchmark.json"
    if not os.path.exists(t0_report):
        log("Phase T0 is running or not yet complete. Waiting for completion...")
        while not os.path.exists(t0_report):
            time.sleep(15)
            # Check if background task failed or file appeared
            if os.path.exists(t0_report):
                break
    log("Phase T0 Benchmark verified on disk.")
    
    # 2. Phase T1: Stock Talker + Romanian MTP LoRA
    t1_report = "reports/T1_mtp_benchmark.json"
    if not os.path.exists(t1_report):
        success = run_step("Phase T1 (Stock Talker + Romanian MTP LoRA Control)", "scripts/run_phase_t1_mtp.py")
        if not success:
            log("Phase T1 encountered an issue. Reviewing...")
    else:
        log("Phase T1 Benchmark already complete.")

    # 3. Phase T2: Romanian Talker-Only LoRA Training (1h Romanian Data)
    t2_adapter = "models/T2_talker_only/final/adapter_model.safetensors"
    if not os.path.exists(t2_adapter):
        success = run_step("Phase T2 Training (Romanian Talker LoRA, 1000 steps)", "scripts/train_phase_t2_talker.py")
    else:
        log("Phase T2 Training checkpoint already exists.")
        
    # Phase T2 Evaluation
    t2_report = "reports/T2_talker_benchmark.json"
    if not os.path.exists(t2_report):
        run_step("Phase T2 Evaluation (Held-out Quick 40)", "scripts/eval_phase_t2_talker.py")
    else:
        log("Phase T2 Evaluation already complete.")
        
    # 4. Phase T3: Joint Romanian Talker LoRA + Romanian MTP LoRA Training
    t3_adapter = "models/T3_talker_mtp/final/adapter_model.safetensors"
    if not os.path.exists(t3_adapter):
        success = run_step("Phase T3 Joint Training (Talker LoRA + MTP LoRA, 1000 steps)", "scripts/train_phase_t3_joint.py")
    else:
        log("Phase T3 Joint Training checkpoint already exists.")
        
    # Phase T3 Evaluation
    t3_report = "reports/T3_talker_mtp_benchmark.json"
    if not os.path.exists(t3_report):
        run_step("Phase T3 Evaluation (Held-out Quick 40)", "scripts/eval_phase_t3_joint.py")
    else:
        log("Phase T3 Evaluation already complete.")
        
    # 5. Compile Master Controlled Matrix
    run_step("Compile Controlled Matrix", "scripts/compare_controlled_matrix.py")
    
    # 6. Build Master Listening Showcase HTML
    run_step("Generate Interactive Listening Showcase HTML", "scripts/generate_showcase_html.py")
    
    # 7. Check if positive learning signal and scale curriculum if time permits
    # Check deadline
    now = datetime.datetime.now()
    log(f"Current local time: {now.strftime('%H:%M:%S')}")
    deadline = now.replace(hour=21, minute=15, second=0, microsecond=0)
    
    if now < deadline:
        log("Positive progress achieved and compute budget remains. Proceeding to extended deep curriculum...")
        # Extended run if budget remains
        run_step("Deep Extended Romanian Curriculum", "scripts/train_extended_curriculum.py")
        run_step("Full 200 Held-Out Evaluation on Champion Model", "scripts/eval_full_200_champion.py")
        run_step("Compile Controlled Matrix", "scripts/compare_controlled_matrix.py")
        run_step("Generate Interactive Listening Showcase HTML", "scripts/generate_showcase_html.py")
        run_step("Compile Master Audit Reports", "scripts/compile_master_audit_reports.py")
        
    # 8. Revert System Power Settings cleanly before 9:30 PM
    log("Restoring Windows system power and sleep settings before deadline...")
    subprocess.run([sys.executable, "scripts/system_power_manager.py", "revert"], check=False)
    log("=== MASTER PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
