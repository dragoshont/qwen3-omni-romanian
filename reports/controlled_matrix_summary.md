# Controlled Romanian Speech AI Experimental Matrix
## Direct Comparison across A0 (Stock) vs A1 (MTP) vs B1 (Talker) vs B2 (Talker+MTP)

Evaluated autonomously on **40 held-out Romanian sentences** (`eval/ro_holdout_quick_40.jsonl`) without teacher forcing (Level 3 TTS).

| Metric | A0: Stock Baseline | A1: MTP-Only LoRA | B1: Talker-Only LoRA | B2: Talker+MTP Joint |
| :--- | :--- | :--- | :--- | :--- |
| **Model Config** | Stock 4-Bit Talker + Stock MTP + Stock Code2Wav | Stock 4-Bit Talker + Romanian MTP LoRA (ro500) + Stock Code2Wav | Romanian 4-Bit Talker LoRA + Stock MTP + Stock Code2Wav | Joint Romanian Talker LoRA + Romanian MTP LoRA + Stock Code2Wav |
| **Mean WER (%)** | **218.65**% | **240.65**% | **287.00**% | **78.97**% |
| **Mean CER (%)** | **209.57**% | **244.99**% | **221.49**% | **30.86**% |
| **Real-Time Factor (RTF)** | 1.59 | 2.13 | 1.89 | 2.40 |
| **Peak VRAM** | 7.65 GB | 7.65 GB | 7.19 GB | 7.23 GB |
| **Provenance Level** | Level 3 (Autonomous) | Level 3 (Autonomous) | Level 3 (Autonomous) | Level 3 (Autonomous) |

### Category-by-Category Character Error Rate (CER %)
| Category | A0: Stock | A1: MTP-Only | B1: Talker-Only | B2: Talker+MTP |
| :--- | :--- | :--- | :--- | :--- |
| **conversational** | 586.1% | 400.7% | 39.7% | 29.4% |
| **a_a_i_heavy** | 296.3% | 63.5% | 268.0% | 35.2% |
| **s_t_heavy** | 187.7% | 585.2% | 204.2% | 33.3% |
| **affricates** | 63.1% | 377.9% | 376.5% | 20.8% |
| **consonant_clusters** | 256.5% | 330.0% | 254.2% | 28.9% |
| **numbers_dates** | 43.8% | 182.8% | 155.0% | 35.7% |
| **romanian_names** | N/A | N/A | N/A | N/A |
| **english_loanwords** | N/A | N/A | N/A | N/A |
| **questions_exclamations** | 300.1% | 46.8% | 43.4% | 26.8% |
| **long_sentences** | 187.9% | 64.8% | 210.1% | 34.9% |