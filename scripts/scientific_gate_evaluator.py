import os
import sys
import json
import numpy as np
import re

sys.stdout.reconfigure(encoding="utf-8")

def detect_repetition(text):
    """
    Detects pathological repetition loops in speech transcriptions:
    - 3-gram repeated 3+ times
    - 2-gram repeated 4+ times
    - 1-gram (word) repeated 5+ times
    - Character hyphenated loops (e.g., -a-a-a-a-a, -r-o-r-o-r)
    """
    if not text:
        return False
        
    t_clean = text.lower().strip()
    words = re.findall(r'\b\w+\b', t_clean)
    
    # Check word repetitions
    if len(words) >= 4:
        # Check consecutive unigram loops (e.g. pune pune pune pune)
        repeat_count = 1
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                repeat_count += 1
                if repeat_count >= 4:
                    return True
            else:
                repeat_count = 1
                
        # Check bigram loops (e.g. de ce de ce de ce)
        if len(words) >= 6:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            for i in range(2, len(bigrams)):
                if bigrams[i] == bigrams[i-2]:
                    # check if repeated 3+ times
                    if i >= 4 and bigrams[i] == bigrams[i-4]:
                        return True
                        
        # Check trigram loops
        if len(words) >= 9:
            trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
            for i in range(3, len(trigrams)):
                if trigrams[i] == trigrams[i-3]:
                    if i >= 6 and trigrams[i] == trigrams[i-6]:
                        return True
                        
    # Check char hyphen loops like -a-a-a-a-a or -o-r-o-r
    if re.search(r'(-[a-z]){5,}', t_clean):
        return True
        
    # Check high compression ratio failure (e.g. 50 words but only 5 unique)
    if len(words) >= 15:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.35:
            return True
            
    return False

def analyze_system(system_key, data, stock_data=None):
    if not data or "detailed_results" not in data:
        return None
        
    results = data["detailed_results"]
    wers = [item.get("wer", 0.0) for item in results]
    cers = [item.get("cer", 0.0) for item in results]
    durations = [item.get("audio_duration_s", 0.0) for item in results]
    
    mean_wer = np.mean(wers)
    median_wer = np.median(wers)
    mean_cer = np.mean(cers)
    median_cer = np.median(cers)
    
    total_dur = sum(durations)
    mean_dur = np.mean(durations)
    peak_vram = data.get("peak_vram_gb", 0.0)
    
    # Calculate EOS success and max-token hits
    eos_successes = 0
    max_token_hits = 0
    repetition_loops = 0
    
    better_than_stock_cer = 0
    worse_than_stock_cer = 0
    tied_stock_cer = 0
    
    better_than_stock_wer = 0
    worse_than_stock_wer = 0
    tied_stock_wer = 0
    
    stock_results = {item["id"]: item for item in stock_data["detailed_results"]} if stock_data else {}
    
    for item in results:
        s_id = item["id"]
        ref_text = item.get("reference_text", "")
        hyp_text = item.get("whisper_transcription", "")
        dur = item.get("audio_duration_s", 0.0)
        
        # Max tokens policy was: min(350, max(80, len(text.split()) * 20))
        # 1 frame = 0.08s
        ref_words = len(ref_text.split())
        max_tokens = min(350, max(80, ref_words * 20))
        max_dur_approx = max_tokens * 0.08
        
        # If duration is within 0.15s of max possible duration, it hit the max token ceiling
        if dur >= (max_dur_approx - 0.25):
            max_token_hits += 1
        else:
            eos_successes += 1
            
        if detect_repetition(hyp_text):
            repetition_loops += 1
            
        if stock_results and s_id in stock_results:
            stk_cer = stock_results[s_id].get("cer", 0.0)
            stk_wer = stock_results[s_id].get("wer", 0.0)
            cur_cer = item.get("cer", 0.0)
            cur_wer = item.get("wer", 0.0)
            
            if cur_cer < stk_cer - 1e-4:
                better_than_stock_cer += 1
            elif cur_cer > stk_cer + 1e-4:
                worse_than_stock_cer += 1
            else:
                tied_stock_cer += 1
                
            if cur_wer < stk_wer - 1e-4:
                better_than_stock_wer += 1
            elif cur_wer > stk_wer + 1e-4:
                worse_than_stock_wer += 1
            else:
                tied_stock_wer += 1
                
    n = len(results)
    return {
        "system_key": system_key,
        "model_configuration": data.get("model_configuration", system_key),
        "sample_count": n,
        "mean_wer": round(float(mean_wer), 4),
        "median_wer": round(float(median_wer), 4),
        "mean_cer": round(float(mean_cer), 4),
        "median_cer": round(float(median_cer), 4),
        "better_than_stock_cer": better_than_stock_cer,
        "worse_than_stock_cer": worse_than_stock_cer,
        "tied_stock_cer": tied_stock_cer,
        "better_than_stock_wer": better_than_stock_wer,
        "worse_than_stock_wer": worse_than_stock_wer,
        "tied_stock_wer": tied_stock_wer,
        "eos_success_count": eos_successes,
        "eos_success_rate": round(float(eos_successes / n * 100), 2) if n > 0 else 0,
        "max_token_hits": max_token_hits,
        "max_token_hit_rate": round(float(max_token_hits / n * 100), 2) if n > 0 else 0,
        "repetition_loops": repetition_loops,
        "repetition_loop_rate": round(float(repetition_loops / n * 100), 2) if n > 0 else 0,
        "total_audio_duration_s": round(float(total_dur), 2),
        "mean_generated_duration_s": round(float(mean_dur), 2),
        "peak_vram_gb": round(float(peak_vram), 2),
        "detailed_results": results
    }

def run_scientific_gate():
    reports = {
        "A0": "reports/T0_stock_benchmark.json",
        "A1": "reports/T1_mtp_benchmark.json",
        "B1": "reports/T2_talker_benchmark.json",
        "B2": "reports/T3_talker_mtp_benchmark.json"
    }
    
    raw_data = {}
    for k, p in reports.items():
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                raw_data[k] = json.load(f)
        else:
            raw_data[k] = None
            
    if not raw_data.get("A0"):
        print("Error: Stock A0 benchmark report not found!")
        return None
        
    analysis = {}
    analysis["A0"] = analyze_system("A0 (Stock Talker + Stock MTP)", raw_data["A0"], stock_data=raw_data["A0"])
    if raw_data.get("A1"):
        analysis["A1"] = analyze_system("A1 (Stock Talker + Romanian MTP LoRA)", raw_data["A1"], stock_data=raw_data["A0"])
    if raw_data.get("B1"):
        analysis["B1"] = analyze_system("B1 (Romanian Talker LoRA + Stock MTP)", raw_data["B1"], stock_data=raw_data["A0"])
    if raw_data.get("B2"):
        analysis["B2"] = analyze_system("B2 (Romanian Talker LoRA + Romanian MTP LoRA)", raw_data["B2"], stock_data=raw_data["A0"])
        
    gate_output = {
        "timestamp": "2026-09-24T13:58:00",
        "systems": analysis
    }
    
    with open("reports/scientific_gate_analysis.json", "w", encoding="utf-8") as f:
        json.dump(gate_output, f, indent=2, ensure_ascii=False)
        
    return gate_output

if __name__ == "__main__":
    out = run_scientific_gate()
    if out:
        print("Scientific gate analysis completed successfully.")
