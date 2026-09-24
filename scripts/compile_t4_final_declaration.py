import os
import sys
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("=" * 70)
    print("COMPILING FINAL T4 CHAMPION DECLARATION (FULL-200 MULTI-METRIC MATRIX)")
    print("=" * 70)
    
    t3_report_file = "reports/champion_200_benchmark.json"
    t3_data = None
    if os.path.exists(t3_report_file):
        with open(t3_report_file, "r", encoding="utf-8") as f:
            t3_data = json.load(f)
            
    eval_steps = [1000, 2000, 2500]
    t4_reports = {}
    for s in eval_steps:
        path = f"reports/t4_step{s}_full200_benchmark.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                t4_reports[s] = json.load(f)
                
    md = []
    md.append("# Phase T4 Champion Declaration & Full-200 Comparison")
    md.append("## Rigorous Scientific Evaluation on Full 200 Held-Out Romanian Sentences (`eval/ro_holdout_200.jsonl`)\n")
    md.append("Dual-metric evaluation protocol: evaluating best stable-mean, best median-CER, and step 2500 under identical decoding parameters across all 10 held-out categories.\n")
    
    headers = ["Metric", "T3 Control (1-Hour)"]
    for s in sorted(t4_reports.keys()):
        headers.append(f"T4 Step {s} (5-Hour)")
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join([":---"] * len(headers)) + " |")
    
    def get_stats(data):
        detailed = data["detailed_results"]
        cers = [x["cer"] for x in detailed]
        wers = [x["wer"] for x in detailed]
        durs = [x.get("audio_duration_s", 0) for x in detailed]
        n = len(detailed)
        
        rep_count = sum(1 for x in detailed if x.get("has_repetition", False))
        eos_count = sum(1 for x in detailed if x.get("eos_success", True))
        max_hit_count = sum(1 for x in detailed if x.get("hit_max_tokens", False))
        
        return {
            "mean_cer": float(np.mean(cers)),
            "median_cer": float(np.median(cers)),
            "p90_cer": float(np.percentile(cers, 90)),
            "mean_wer": float(np.mean(wers)),
            "median_wer": float(np.median(wers)),
            "p90_wer": float(np.percentile(wers, 90)),
            "eos_rate": float(data.get("eos_rate", (eos_count / n * 100))),
            "repetition_rate": float(data.get("repetition_rate", (rep_count / n * 100))),
            "max_token_hit_rate": float(data.get("max_token_hit_rate", (max_hit_count / n * 100))),
            "total_dur": sum(durs),
            "mean_dur": float(np.mean(durs)) if durs else 0.0,
            "category_breakdown": data.get("category_breakdown", {})
        }
        
    t3_stats = get_stats(t3_data) if t3_data else {}
    t4_stats = {s: get_stats(data) for s, data in t4_reports.items()}
    
    def row(name, key, fmt="{:.2f}%", is_pct=True):
        if t3_stats and key in t3_stats:
            val = t3_stats[key]
            t3_v = fmt.format(val * 100) if is_pct else fmt.format(val)
        else:
            t3_v = "N/A"
        r = [f"**{name}**", t3_v]
        for s in sorted(t4_reports.keys()):
            val = t4_stats[s][key]
            r.append(fmt.format(val * 100) if is_pct else fmt.format(val))
        return "| " + " | ".join(r) + " |"
        
    md.append(row("Mean CER", "mean_cer"))
    md.append(row("Median CER", "median_cer"))
    md.append(row("P90 CER", "p90_cer"))
    md.append(row("Mean WER", "mean_wer"))
    md.append(row("Median WER", "median_wer"))
    md.append(row("P90 WER", "p90_wer"))
    md.append(row("EOS Success Rate", "eos_rate", "{:.1f}%", False))
    md.append(row("Repetition Rate", "repetition_rate", "{:.1f}%", False))
    md.append(row("Max-Token Hit Rate", "max_token_hit_rate", "{:.1f}%", False))
    md.append(row("Mean Duration (s)", "mean_dur", "{:.2f}s", False))
    md.append(row("Total Duration (s)", "total_dur", "{:.1f}s", False))
    
    # Per-category comparison
    md.append("\n### Per-Category Mean CER Breakdown across 10 Holdout Categories\n")
    all_cats = [
        "conversational", "a_a_i_heavy", "s_t_heavy", "affricates",
        "consonant_clusters", "numbers_dates", "names_places",
        "technical_english", "questions_exclamations", "long_sentences"
    ]
    
    cat_headers = ["Category", "T3 Control"]
    for s in sorted(t4_reports.keys()):
        cat_headers.append(f"T4 Step {s}")
    md.append("| " + " | ".join(cat_headers) + " |")
    md.append("| " + " | ".join([":---"] * len(cat_headers)) + " |")
    
    for cat in all_cats:
        c_row = [f"**{cat}**"]
        t3_c = t3_stats.get("category_breakdown", {}).get(cat, {})
        c_row.append(f"{t3_c.get('mean_cer', 0.0)*100:.2f}%" if t3_c else "N/A")
        for s in sorted(t4_reports.keys()):
            t4_c = t4_stats[s].get("category_breakdown", {}).get(cat, {})
            c_row.append(f"{t4_c.get('mean_cer', 0.0)*100:.2f}%" if t4_c else "N/A")
        md.append("| " + " | ".join(c_row) + " |")
        
    # Multi-metric Champion Decision
    champion_step = None
    champion_reason = ""
    if t4_stats:
        # Check if step 2500 dominates all other candidates on both mean and median CER with zero/low repetition
        candidates = sorted(t4_stats.keys())
        best_mean_step = min(candidates, key=lambda s: t4_stats[s]["mean_cer"])
        best_median_step = min(candidates, key=lambda s: t4_stats[s]["median_cer"])
        
        if best_mean_step == best_median_step:
            champion_step = best_mean_step
            champion_reason = f"Step {champion_step} strictly achieves both the lowest Mean CER ({t4_stats[champion_step]['mean_cer']*100:.2f}%) and lowest Median CER ({t4_stats[champion_step]['median_cer']*100:.2f}%)."
        else:
            # If they differ, check trade-off (e.g. repetition stability, p90, combined rank)
            # Rank sum
            scored = []
            for s in candidates:
                rank_score = t4_stats[s]["mean_cer"] * 0.5 + t4_stats[s]["median_cer"] * 0.5 + (0.1 if t4_stats[s]["repetition_rate"] > 1.0 else 0.0)
                scored.append((rank_score, s))
            scored.sort()
            champion_step = scored[0][1]
            champion_reason = f"Step {champion_step} selected via composite Pareto ranking: Mean CER {t4_stats[champion_step]['mean_cer']*100:.2f}%, Median CER {t4_stats[champion_step]['median_cer']*100:.2f}%, Repetition {t4_stats[champion_step]['repetition_rate']:.1f}%."
            
    md.append(f"\n### Final Multi-Metric Champion Declaration")
    if champion_step is not None:
        champ_c = t4_stats[champion_step]
        md.append(f"- **Declared T4 Champion**: **Step {champion_step}**")
        md.append(f"- **Selection Rationale**: {champion_reason}")
        md.append(f"- **Champion Full-200 Mean CER**: **{champ_c['mean_cer']*100:.2f}%**")
        md.append(f"- **Champion Full-200 Median CER**: **{champ_c['median_cer']*100:.2f}%**")
        md.append(f"- **Champion Full-200 P90 CER**: **{champ_c['p90_cer']*100:.2f}%**")
        md.append(f"- **Champion Full-200 EOS Success**: **{champ_c['eos_rate']:.1f}%**")
        md.append(f"- **Champion Full-200 Repetition Rate**: **{champ_c['repetition_rate']:.1f}%**")
        if t3_stats:
            diff_med = (champ_c["median_cer"] - t3_stats["median_cer"]) / t3_stats["median_cer"] * 100
            diff_mean = (champ_c["mean_cer"] - t3_stats["mean_cer"]) / t3_stats["mean_cer"] * 100
            md.append(f"- **Relative Improvement vs T3 Control (1-Hour)**:")
            md.append(f"  * Median CER: **{-diff_med:.1f}% reduction** ({t3_stats['median_cer']*100:.2f}% -> {champ_c['median_cer']*100:.2f}%)")
            md.append(f"  * Mean CER: **{-diff_mean:.1f}% reduction** ({t3_stats['mean_cer']*100:.2f}% -> {champ_c['mean_cer']*100:.2f}%)")
            
    out_md = "\n".join(md)
    print(out_md)
    with open("reports/t4_final_champion_declaration.md", "w", encoding="utf-8") as f:
        f.write(out_md)
        
    print("\nSaved declaration to reports/t4_final_champion_declaration.md")

if __name__ == "__main__":
    main()
