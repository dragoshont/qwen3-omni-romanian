import json

def main():
    with open("reports/scientific_gate_analysis.json", "r", encoding="utf-8") as f:
        data = json.load(f)["systems"]
        
    cats = [
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
    
    print(f"{'Category':<24} | {'A0 (Stock)':<12} | {'A1 (MTP-Only)':<14} | {'B1 (Talker-Only)':<16} | {'B2 (Joint Talker+MTP)':<20}")
    print("-" * 95)
    for c in cats:
        c_a0 = [x['cer'] for x in data['A0']['detailed_results'] if x.get('category') == c]
        c_a1 = [x['cer'] for x in data['A1']['detailed_results'] if x.get('category') == c]
        c_b1 = [x['cer'] for x in data['B1']['detailed_results'] if x.get('category') == c]
        c_b2 = [x['cer'] for x in data['B2']['detailed_results'] if x.get('category') == c]
        
        v_a0 = f"{sum(c_a0)/len(c_a0)*100:6.1f}%" if c_a0 else "N/A"
        v_a1 = f"{sum(c_a1)/len(c_a1)*100:6.1f}%" if c_a1 else "N/A"
        v_b1 = f"{sum(c_b1)/len(c_b1)*100:6.1f}%" if c_b1 else "N/A"
        v_b2 = f"{sum(c_b2)/len(c_b2)*100:6.1f}%" if c_b2 else "N/A"
        print(f"{c:<24} | {v_a0:<12} | {v_a1:<14} | {v_b1:<16} | {v_b2:<20}")

if __name__ == "__main__":
    main()
