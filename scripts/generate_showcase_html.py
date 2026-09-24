import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

def generate_showcase():
    print("Generating Interactive Audio Showcase HTML...")
    
    t0_file = "reports/T0_stock_benchmark.json"
    t1_file = "reports/T1_mtp_benchmark.json"
    t2_file = "reports/T2_talker_benchmark.json"
    t3_file = "reports/T3_talker_mtp_benchmark.json"
    
    def load_json(p):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
        
    d0 = load_json(t0_file)
    d1 = load_json(t1_file)
    d2 = load_json(t2_file)
    d3 = load_json(t3_file)
    
    # Load 40 evaluation samples
    samples = []
    with open("eval/ro_holdout_quick_40.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    def get_item(d, sid):
        if d and "detailed_results" in d:
            for item in d["detailed_results"]:
                if item["id"] == sid:
                    return item
        return None

    html = []
    html.append("""<!DOCTYPE html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen3-Omni Romanian TTS — Scientific Controlled Matrix Showcase</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0d14;
            --bg-secondary: #131722;
            --card-bg: rgba(22, 27, 40, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --accent-purple: #9d4edd;
            --accent-green: #10b981;
            --accent-orange: #f59e0b;
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
            --font-main: 'Outfit', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: var(--font-main);
            line-height: 1.6;
            padding: 40px 24px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header {
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(157, 78, 221, 0.1) 100%);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            backdrop-filter: blur(12px);
        }
        h1 {
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }
        .subtitle { color: var(--text-muted); font-size: 1.1rem; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            transition: transform 0.2s, border-color 0.2s;
        }
        .stat-card:hover { transform: translateY(-3px); border-color: var(--accent-cyan); }
        .stat-title { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 8px; }
        .stat-val { font-size: 2.2rem; font-weight: 700; font-family: var(--font-mono); }
        .stat-desc { font-size: 0.85rem; color: var(--text-muted); margin-top: 6px; }
        
        .sample-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            backdrop-filter: blur(8px);
        }
        .sample-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--card-border);
        }
        .sample-id { font-family: var(--font-mono); font-weight: 600; color: var(--accent-cyan); }
        .sample-cat {
            background: rgba(79, 172, 254, 0.15);
            color: var(--accent-blue);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .ref-text {
            font-size: 1.15rem;
            color: #ffffff;
            margin-bottom: 20px;
            font-weight: 400;
        }
        .audio-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }
        .audio-box {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 16px;
        }
        .audio-label {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
        }
        .label-t0 { color: var(--text-muted); }
        .label-t1 { color: var(--accent-orange); }
        .label-t2 { color: var(--accent-blue); }
        .label-t3 { color: var(--accent-green); }
        audio {
            width: 100%;
            height: 38px;
            border-radius: 8px;
            margin-bottom: 10px;
            outline: none;
        }
        .asr-box {
            font-family: var(--font-mono);
            font-size: 0.78rem;
            color: #cbd5e1;
            background: rgba(0, 0, 0, 0.4);
            padding: 8px;
            border-radius: 6px;
            line-height: 1.4;
        }
        .metrics-pill {
            display: inline-block;
            margin-top: 6px;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
        }
        .metrics-pill.good { background: rgba(16, 185, 129, 0.2); color: #34d399; }
        .metrics-pill.mid { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
        .metrics-pill.high { background: rgba(239, 68, 68, 0.2); color: #f87171; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Qwen3-Omni Romanian Speech Synthesis</h1>
            <p class="subtitle">Controlled Scientific Matrix (T0 Stock vs T1 MTP vs T2 Talker vs T3 Joint) on 40 Held-Out Romanian Sentences (Level 3 Autonomous TTS)</p>
        </header>
""")

    # Overview stats
    html.append("""        <div class="stats-grid">""")
    
    phases = [
        ("A0: Stock Baseline", d0, "--text-muted"),
        ("A1: MTP-Only LoRA", d1, "--accent-orange"),
        ("B1: Talker-Only LoRA", d2, "--accent-blue"),
        ("B2: Talker+MTP Joint", d3, "--accent-green"),
    ]
    
    for title, d, col in phases:
        if d:
            wer = f"{d.get('mean_wer', 0)*100:.1f}%"
            cer = f"{d.get('mean_cer', 0)*100:.1f}%"
            rtf = f"{d.get('overall_rtf', 0):.2f}"
            desc = f"CER: {cer} | RTF: {rtf}x | {d.get('model_configuration', '')[:30]}..."
        else:
            wer = "Running..."
            desc = "Experiment in progress"
            
        html.append(f"""
            <div class="stat-card">
                <div class="stat-title">{title}</div>
                <div class="stat-val" style="color: var({col})">{wer}</div>
                <div class="stat-desc">{desc}</div>
            </div>""")
            
    html.append("""        </div>""")
    
    # Samples list
    for idx, sample in enumerate(samples, 1):
        sid = sample["id"]
        cat = sample["category"]
        text = sample["text"]
        
        i0 = get_item(d0, sid)
        i1 = get_item(d1, sid)
        i2 = get_item(d2, sid)
        i3 = get_item(d3, sid)
        
        html.append(f"""
        <div class="sample-card">
            <div class="sample-header">
                <span class="sample-id">{sid}</span>
                <span class="sample-cat">{cat}</span>
            </div>
            <div class="ref-text">„{text}”</div>
            <div class="audio-grid">""")
            
        cards = [
            ("T0: Stock Baseline", i0, "T0_stock", "label-t0"),
            ("T1: MTP-Only LoRA", i1, "T1_mtp_only", "label-t1"),
            ("T2: Talker-Only LoRA", i2, "T2_talker_only", "label-t2"),
            ("T3: Joint Talker+MTP", i3, "T3_talker_mtp", "label-t3"),
        ]
        
        for name, item, folder, label_cls in cards:
            wav_rel = f"{folder}/{sid}.wav"
            if item:
                cer = item.get("cer", 0) * 100
                wer = item.get("wer", 0) * 100
                hyp = item.get("whisper_transcription", "")
                pill_cls = "good" if cer < 25 else ("mid" if cer < 50 else "high")
                pill_html = f'<div class="metrics-pill {pill_cls}">WER {wer:.0f}% | CER {cer:.1f}%</div>'
                hyp_html = f'<div class="asr-box"><strong>Whisper:</strong> {hyp}{pill_html}</div>'
            else:
                hyp_html = '<div class="asr-box"><em>Processing...</em></div>'
                
            html.append(f"""
                <div class="audio-box">
                    <div class="audio-label {label_cls}">
                        <span>{name}</span>
                    </div>
                    <audio controls src="{wav_rel}" preload="none"></audio>
                    {hyp_html}
                </div>""")
                
        html.append("""
            </div>
        </div>""")
        
    html.append("""
    </div>
</body>
</html>""")
    
    out_html_path = "outputs/listening_index.html"
    os.makedirs("outputs", exist_ok=True)
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
        
    print(f"Generated {out_html_path} successfully.")

if __name__ == "__main__":
    generate_showcase()
