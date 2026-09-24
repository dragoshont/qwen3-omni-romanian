# Phase T4 Data-Scaling Checkpoint Comparison
## Direct Comparison: T3 (1-Hour Control) vs T4 Checkpoints (5-Hour Corpus)

Evaluated under identical decoding parameters on 40 held-out sentences (`ro_holdout_quick_40.jsonl`).

| Metric | T3 Control (Step 1000) | T4 Step 500 | T4 Step 1000 | T4 Step 1500 | T4 Step 2000 | T4 Step 2500 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean CER** | 30.86% | 65.43% | 26.03% | 26.50% | 59.87% | 24.24% |
| **Median CER** | 28.49% | 30.60% | 24.68% | 25.44% | 22.74% | 23.17% |
| **Mean WER** | 78.97% | 165.71% | 70.04% | 71.36% | 106.95% | 68.42% |
| **Median WER** | 78.36% | 80.00% | 69.62% | 72.73% | 66.67% | 66.67% |
| **EOS Success Rate** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **Repetition Rate** | 0.0% | 2.5% | 0.0% | 0.0% | 2.5% | 0.0% |
| **Mean Duration (s)** | 4.58s | 4.56 | 4.54 | 4.5 | 4.52 | 4.52 |

### Multi-Metric Checkpoint Analysis & Full-200 Evaluation Plan
- **Best Stable-Mean Checkpoint**: **Step 2500** (Mean CER: 24.24%, Repetition: 0.0%)
- **Best Median-CER Checkpoint**: **Step 2000** (Median CER: 22.74%)
- **Step 2500 Dominates Both**: NO
- **Checkpoints Queued for Full-200 Benchmark**: [1000, 2000, 2500]