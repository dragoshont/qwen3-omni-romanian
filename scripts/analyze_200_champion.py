import os
import sys
import json
import numpy as np
import re

sys.stdout.reconfigure(encoding="utf-8")

def detect_repetition(text):
    if not text:
        return False
    t_clean = text.lower().strip()
    words = re.findall(r'\b\w+\b', t_clean)
    if len(words) >= 4:
        repeat_count = 1
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                repeat_count += 1
                if repeat_count >= 4:
                    return True
            else:
                repeat_count = 1
        if len(words) >= 6:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            for i in range(2, len(bigrams)):
                if bigrams[i] == bigrams[i-2]:
                    if i >= 4 and bigrams[i] == bigrams[i-4]:
                        return True
        if len(words) >= 9:
            trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
            for i in range(3, len(trigrams)):
                if trigrams[i] == trigrams[i-3]:
                    if i >= 6 and trigrams[i] == trigrams[i-6]:
                        return True
    if re.search(r'(-[a-z]){5,}', t_clean):
        return True
    if len(words) >= 15:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.35:
            return True
    return False

def analyze_200():
    report_path = "reports/champion_200_benchmark.json"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    detailed = data["detailed_results"]
    
    # Categories in order
    categories = [
        "conversational",
        "a_a_i_heavy",
        "s_t_heavy",
        "affricates",
        "consonant_clusters",
        "numbers_dates",
        "names_places",
        "technical_english",
        "questions_exclamations",
        "long_sentences"
    ]
    
    # Process each item
    for item in detailed:
        ref_text = item.get("reference_text", "")
        hyp_text = item.get("whisper_transcription", "")
        dur = item.get("audio_duration_s", 0.0)
        
        words = len(ref_text.split())
        max_tokens = min(350, max(80, words * 20))
        max_dur_approx = max_tokens * 0.08
        
        item["hit_max_tokens"] = dur >= (max_dur_approx - 0.25)
        item["eos_success"] = not item["hit_max_tokens"]
        item["has_repetition"] = detect_repetition(hyp_text)

    def compute_stats(items):
        n = len(items)
        if n == 0:
            return None
        cers = [x["cer"] for x in items]
        wers = [x["wer"] for x in items]
        durs = [x["audio_duration_s"] for x in items]
        eos_count = sum(1 for x in items if x["eos_success"])
        max_token_count = sum(1 for x in items if x["hit_max_tokens"])
        rep_count = sum(1 for x in items if x["has_repetition"])
        
        return {
            "count": n,
            "mean_cer": float(np.mean(cers)),
            "median_cer": float(np.median(cers)),
            "p90_cer": float(np.percentile(cers, 90)),
            "mean_wer": float(np.mean(wers)),
            "median_wer": float(np.median(wers)),
            "p90_wer": float(np.percentile(wers, 90)),
            "eos_rate": float(eos_count / n * 100),
            "max_token_hit_rate": float(max_token_count / n * 100),
            "repetition_rate": float(rep_count / n * 100),
            "mean_duration_s": float(np.mean(durs)),
            "total_duration_s": float(sum(durs))
        }

    overall_stats = compute_stats(detailed)
    per_cat_stats = {}
    for cat in categories:
        cat_items = [x for x in detailed if x.get("category") == cat]
        per_cat_stats[cat] = compute_stats(cat_items)

    # 10 Best samples (lowest CER)
    sorted_by_cer = sorted(detailed, key=lambda x: (x["cer"], x["wer"]))
    best_10 = sorted_by_cer[:10]
    
    # 10 Worst samples (highest CER)
    worst_10 = sorted_by_cer[-10:]
    worst_10.reverse() # worst first
    
    # 10 Median-ish samples (closest to overall median CER)
    overall_median_cer = overall_stats["median_cer"]
    sorted_by_median_dist = sorted(detailed, key=lambda x: abs(x["cer"] - overall_median_cer))
    median_10 = sorted_by_median_dist[:10]

    # Evaluate Project Scaling Gate
    gate_checks = {
        "median_cer_le_40": {
            "target": "<= 40.0%",
            "actual": f"{overall_stats['median_cer']*100:.2f}%",
            "pass": overall_stats['median_cer'] <= 0.40
        },
        "mean_cer_le_50": {
            "target": "<= 50.0%",
            "actual": f"{overall_stats['mean_cer']*100:.2f}%",
            "pass": overall_stats['mean_cer'] <= 0.50
        },
        "eos_success_ge_98": {
            "target": ">= 98.0%",
            "actual": f"{overall_stats['eos_rate']:.2f}%",
            "pass": overall_stats['eos_rate'] >= 98.0
        },
        "repetition_loops_le_2": {
            "target": "<= 2.0%",
            "actual": f"{overall_stats['repetition_rate']:.2f}%",
            "pass": overall_stats['repetition_rate'] <= 2.0
        },
        "max_token_hits_le_2": {
            "target": "<= 2.0%",
            "actual": f"{overall_stats['max_token_hit_rate']:.2f}%",
            "pass": overall_stats['max_token_hit_rate'] <= 2.0
        },
        "no_catastrophic_collapse": {
            "target": "No category mean CER > 150%",
            "actual": f"Max category mean CER = {max(s['mean_cer'] for s in per_cat_stats.values())*100:.2f}% ({max(per_cat_stats.keys(), key=lambda k: per_cat_stats[k]['mean_cer'])})",
            "pass": all(s["mean_cer"] <= 1.50 for s in per_cat_stats.values())
        }
    }
    
    gate_passed = all(check["pass"] for check in gate_checks.values())

    output = {
        "overall": overall_stats,
        "per_category": per_cat_stats,
        "best_10_samples": best_10,
        "median_10_samples": median_10,
        "worst_10_samples": worst_10,
        "gate_checks": gate_checks,
        "gate_passed": gate_passed
    }

    with open("reports/champion_200_comprehensive_audit.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "overall": overall_stats,
        "gate_checks": gate_checks,
        "gate_passed": gate_passed
    }, indent=2))

if __name__ == "__main__":
    analyze_200()
