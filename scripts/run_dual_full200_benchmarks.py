import os
import sys
import time
import subprocess
import json

sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("=" * 70, flush=True)
    print("STARTING DUAL FULL-200 BENCHMARKS (STEP 1000 & STEP 2000)", flush=True)
    print("Preserving Step 1000 (stable mean) and Step 2000 (median CER)", flush=True)
    print("=" * 70, flush=True)
    
    python_exe = sys.executable
    
    # 1. Step 1000 Evaluation
    t0 = time.time()
    print("\n>>> [1/2] RUNNING FULL-200 EVALUATION ON STEP 1000...", flush=True)
    cmd1 = [
        python_exe,
        "scripts/eval_full_200_champion.py",
        "models/T4_data_scale_5h/final/checkpoint_step_1000",
        "reports/t4_step1000_full200_benchmark.json",
        "outputs/t4_step1000_full200"
    ]
    res1 = subprocess.run(cmd1)
    if res1.returncode != 0:
        print(f"Error evaluating Step 1000 (code {res1.returncode})", flush=True)
        return
    print(f"Step 1000 Full-200 completed in {(time.time() - t0)/60:.1f} minutes.", flush=True)
    
    # 2. Step 2000 Evaluation
    t1 = time.time()
    print("\n>>> [2/2] RUNNING FULL-200 EVALUATION ON STEP 2000...", flush=True)
    cmd2 = [
        python_exe,
        "scripts/eval_full_200_champion.py",
        "models/T4_data_scale_5h/final/checkpoint_step_2000",
        "reports/t4_step2000_full200_benchmark.json",
        "outputs/t4_step2000_full200"
    ]
    res2 = subprocess.run(cmd2)
    if res2.returncode != 0:
        print(f"Error evaluating Step 2000 (code {res2.returncode})", flush=True)
        return
    print(f"Step 2000 Full-200 completed in {(time.time() - t1)/60:.1f} minutes.", flush=True)
    
    # 3. Compile Final Declaration
    print("\n>>> COMPILING FINAL T4 CHAMPION DECLARATION...", flush=True)
    subprocess.run([python_exe, "scripts/compile_t4_final_declaration.py"])
    
    # 4. Refresh Package Archive
    print("\n>>> REFRESHING RESEARCH AUDIT ZIP ARCHIVE...", flush=True)
    subprocess.run([python_exe, "scripts/package_research_zip.py"])
    
    # 5. Revert System Power Plan
    print("\n>>> REVERTING SYSTEM POWER PLAN TO BALANCED...", flush=True)
    cmd_revert = [python_exe, "scripts/system_power_manager.py", "revert"]
    subprocess.run(cmd_revert)
    
    print("\n" + "=" * 70, flush=True)
    print(f"ALL T4 BENCHMARKS, DECLARATIONS, AND PRESERVATIONS COMPLETE IN {(time.time() - t0)/60:.1f} MINUTES!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
