import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

def generate_matrix():
    print("=" * 70)
    print("COMPILING CONTROLLED EXPERIMENTAL MATRIX (T0 vs T1 vs T2 vs T3)")
    print("=" * 70)
    
    reports = {
        "A0_T0_stock": "reports/T0_stock_benchmark.json",
        "A1_T1_mtp_only": "reports/T1_mtp_benchmark.json",
        "B1_T2_talker_only": "reports/T2_talker_benchmark.json",
        "B2_T3_talker_mtp": "reports/T3_talker_mtp_benchmark.json",
    }
    
    data = {}
    for key, path in reports.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data[key] = json.load(f)
        else:
            data[key] = None
            
    # Markdown comparison table
    md = []
    md.append("# Controlled Romanian Speech AI Experimental Matrix")
    md.append("## Direct Comparison across A0 (Stock) vs A1 (MTP) vs B1 (Talker) vs B2 (Talker+MTP)")
    md.append("\nEvaluated autonomously on **40 held-out Romanian sentences** (`eval/ro_holdout_quick_40.jsonl`) without teacher forcing (Level 3 TTS).\n")
    
    md.append("| Metric | A0: Stock Baseline | A1: MTP-Only LoRA | B1: Talker-Only LoRA | B2: Talker+MTP Joint |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    
    def get_val(key, field, fmt="{:.2f}"):
        if data.get(key) and field in data[key]:
            val = data[key][field]
            if isinstance(val, float):
                return fmt.format(val * 100 if "wer" in field or "cer" in field else val)
            return str(val)
        return "Pending / Running"
        
    md.append(f"| **Model Config** | {get_val('A0_T0_stock', 'model_configuration')} | {get_val('A1_T1_mtp_only', 'model_configuration')} | {get_val('B1_T2_talker_only', 'model_configuration')} | {get_val('B2_T3_talker_mtp', 'model_configuration')} |")
    md.append(f"| **Mean WER (%)** | **{get_val('A0_T0_stock', 'mean_wer')}**% | **{get_val('A1_T1_mtp_only', 'mean_wer')}**% | **{get_val('B1_T2_talker_only', 'mean_wer')}**% | **{get_val('B2_T3_talker_mtp', 'mean_wer')}**% |")
    md.append(f"| **Mean CER (%)** | **{get_val('A0_T0_stock', 'mean_cer')}**% | **{get_val('A1_T1_mtp_only', 'mean_cer')}**% | **{get_val('B1_T2_talker_only', 'mean_cer')}**% | **{get_val('B2_T3_talker_mtp', 'mean_cer')}**% |")
    md.append(f"| **Real-Time Factor (RTF)** | {get_val('A0_T0_stock', 'overall_rtf')} | {get_val('A1_T1_mtp_only', 'overall_rtf')} | {get_val('B1_T2_talker_only', 'overall_rtf')} | {get_val('B2_T3_talker_mtp', 'overall_rtf')} |")
    md.append(f"| **Peak VRAM** | {get_val('A0_T0_stock', 'peak_vram_gb')} GB | {get_val('A1_T1_mtp_only', 'peak_vram_gb')} GB | {get_val('B1_T2_talker_only', 'peak_vram_gb')} GB | {get_val('B2_T3_talker_mtp', 'peak_vram_gb')} GB |")
    md.append(f"| **Provenance Level** | Level 3 (Autonomous) | Level 3 (Autonomous) | Level 3 (Autonomous) | Level 3 (Autonomous) |")
    
    # Category-by-Category breakdown if data available
    categories = [
        "conversational", "a_a_i_heavy", "s_t_heavy", "affricates",
        "consonant_clusters", "numbers_dates", "romanian_names",
        "english_loanwords", "questions_exclamations", "long_sentences"
    ]
    
    md.append("\n### Category-by-Category Character Error Rate (CER %)")
    md.append("| Category | A0: Stock | A1: MTP-Only | B1: Talker-Only | B2: Talker+MTP |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    
    for cat in categories:
        cat_cers = []
        for key in ["A0_T0_stock", "A1_T1_mtp_only", "B1_T2_talker_only", "B2_T3_talker_mtp"]:
            if data.get(key) and "detailed_results" in data[key]:
                cat_items = [item["cer"] for item in data[key]["detailed_results"] if item.get("category") == cat]
                if cat_items:
                    avg_c = sum(cat_items) / len(cat_items)
                    cat_cers.append(f"{avg_c * 100:.1f}%")
                else:
                    cat_cers.append("N/A")
            else:
                cat_cers.append("-")
        md.append(f"| **{cat}** | {cat_cers[0]} | {cat_cers[1]} | {cat_cers[2]} | {cat_cers[3]} |")
        
    out_md = "\n".join(md)
    with open("reports/controlled_matrix_summary.md", "w", encoding="utf-8") as f:
        f.write(out_md)
        
    with open("reports/controlled_matrix.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print("Updated reports/controlled_matrix_summary.md and reports/controlled_matrix.json")
    print(out_md)

if __name__ == "__main__":
    generate_matrix()
