# Phase T4 Champion Declaration & Full-200 Comparison
## Rigorous Scientific Evaluation on Full 200 Held-Out Romanian Sentences (`eval/ro_holdout_200.jsonl`)

Dual-metric evaluation protocol: evaluating best stable-mean, best median-CER, and step 2500 under identical decoding parameters across all 10 held-out categories.

| Metric | T3 Control (1-Hour) | T4 Step 1000 (5-Hour) | T4 Step 2000 (5-Hour) | T4 Step 2500 (5-Hour) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean CER** | 38.53% | 30.09% | 36.00% | 30.73% |
| **Median CER** | 29.30% | 23.91% | 23.46% | 22.85% |
| **P90 CER** | 45.13% | 44.28% | 39.24% | 41.47% |
| **Mean WER** | 87.85% | 76.10% | 77.62% | 75.81% |
| **Median WER** | 75.00% | 70.00% | 65.99% | 63.64% |
| **P90 WER** | 110.82% | 100.00% | 100.00% | 110.00% |
| **EOS Success Rate** | 100.0% | 100.0% | 100.0% | 100.0% |
| **Repetition Rate** | 0.0% | 0.5% | 1.0% | 1.0% |
| **Max-Token Hit Rate** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Mean Duration (s)** | 4.47s | 4.36s | 4.48s | 4.54s |
| **Total Duration (s)** | 894.1s | 871.4s | 895.4s | 908.3s |

### Per-Category Mean CER Breakdown across 10 Holdout Categories

| Category | T3 Control | T4 Step 1000 | T4 Step 2000 | T4 Step 2500 |
| :--- | :--- | :--- | :--- | :--- |
| **conversational** | 22.21% | 19.16% | 18.00% | 70.87% |
| **a_a_i_heavy** | 34.54% | 32.33% | 55.91% | 29.03% |
| **s_t_heavy** | 28.08% | 23.20% | 27.72% | 24.02% |
| **affricates** | 29.79% | 30.63% | 24.62% | 19.32% |
| **consonant_clusters** | 23.26% | 31.59% | 21.06% | 21.45% |
| **numbers_dates** | 36.45% | 36.76% | 31.25% | 33.40% |
| **names_places** | 29.47% | 29.46% | 60.98% | 28.16% |
| **technical_english** | 108.76% | 56.39% | 80.26% | 26.20% |
| **questions_exclamations** | 25.31% | 14.17% | 16.82% | 19.18% |
| **long_sentences** | 47.46% | 27.21% | 23.42% | 35.61% |

### Final Multi-Metric Champion Declaration
- **Declared T4 Champion**: **Step 2500**
- **Selection Rationale**: Step 2500 selected via composite Pareto ranking: Mean CER 30.73%, Median CER 22.85%, Repetition 1.0%.
- **Champion Full-200 Mean CER**: **30.73%**
- **Champion Full-200 Median CER**: **22.85%**
- **Champion Full-200 P90 CER**: **41.47%**
- **Champion Full-200 EOS Success**: **100.0%**
- **Champion Full-200 Repetition Rate**: **1.0%**
- **Relative Improvement vs T3 Control (1-Hour)**:
  * Median CER: **22.0% reduction** (29.30% -> 22.85%)
  * Mean CER: **20.3% reduction** (38.53% -> 30.73%)