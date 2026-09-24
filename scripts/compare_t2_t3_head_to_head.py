import json

def compare_t2_t3():
    with open("reports/scientific_gate_analysis.json", "r", encoding="utf-8") as f:
        data = json.load(f)["systems"]
        
    b1_results = {x["id"]: x for x in data["B1"]["detailed_results"]}
    b2_results = {x["id"]: x for x in data["B2"]["detailed_results"]}
    
    t3_wins_cer = 0
    t2_wins_cer = 0
    ties_cer = 0
    
    t3_wins_wer = 0
    t2_wins_wer = 0
    ties_wer = 0
    
    for s_id, item_b2 in b2_results.items():
        item_b1 = b1_results[s_id]
        c1, c2 = item_b1["cer"], item_b2["cer"]
        w1, w2 = item_b1["wer"], item_b2["wer"]
        
        if c2 < c1 - 1e-4:
            t3_wins_cer += 1
        elif c2 > c1 + 1e-4:
            t2_wins_cer += 1
        else:
            ties_cer += 1
            
        if w2 < w1 - 1e-4:
            t3_wins_wer += 1
        elif w2 > w1 + 1e-4:
            t2_wins_wer += 1
        else:
            ties_wer += 1
            
    print(f"Direct Head-to-Head Comparison: T2 (Talker-Only) vs T3 (Joint Talker+MTP):")
    print(f"CER Wins: T3 (Joint) wins {t3_wins_cer}/40 ({t3_wins_cer/40*100:.1f}%), T2 wins {t2_wins_cer}/40 ({t2_wins_cer/40*100:.1f}%), Ties: {ties_cer}")
    print(f"WER Wins: T3 (Joint) wins {t3_wins_wer}/40 ({t3_wins_wer/40*100:.1f}%), T2 wins {t2_wins_wer}/40 ({t2_wins_wer/40*100:.1f}%), Ties: {ties_wer}")

if __name__ == "__main__":
    compare_t2_t3()
