# PROJECT GUIDE — Model Building, Fully Ready
**Explainable ML for Predicting Nursing Licensure Examination Failure in Ghana: An XGBoost-SHAP Approach**
Prince Bortey Miller | 22388461 | Supervision Code 289 | Supervisor: Dr. Eric Opoku Osei
Repo: github.com/yhmiller/xgboost-shap-nursing-licensure

---

## 1. Folder structure (set this up on your Mac)

Matches the CAN-DO Lab engineering doc's required layout:

```
xgboost-shap-nursing-licensure/
│
├── data/
│   ├── raw_college_records.xlsx      ← the file the college gives you (NEVER commit/share)
│   └── anonymised_records.csv        ← output of Stage 0 (safe to keep in Drive)
│
├── run.sh                            ← ONE command: set up + run the whole analysis
├── run_all.py                        ← runs Stage 1 + Stage 2 in a single process
├── app.sh                            ← ONE command: launch the educator screening app
│
├── notebooks/                        ← the stages (also runnable cell-by-cell in Colab)
│   ├── 00_data_preparation_and_eda.py    Stage 0: clean → exclude → anonymise → EDA
│   ├── 01_pipeline_and_experiments.py    Stage 1: baselines + XGBoost + SHAP + eval (20 cells)
│   └── 02_model_engineering.py           Stage 2: E-XGBoost vs baseline (10 ENG-cells)
│
├── app/                              ← educator risk-screening web app
│   ├── app.py                            Streamlit UI (upload → at-risk list → SHAP why)
│   ├── inference.py                      loads the model bundle, scores records
│   └── example_students.csv             sample upload to try the app
│
├── results/                          ← everything below is auto-generated
│   ├── eda_descriptives.csv,  eda_missingness.csv
│   ├── eda_failrate_by_*.csv
│   ├── eda_distributions.png, eda_correlation.png
│   ├── shap_beeswarm.png,     shap_waterfall_student*.png
│   ├── calibration_curves.png, roc_pr_curves.png
│   ├── baseline_metrics.json, engineered_metrics.json
│   ├── comparison_table.csv          ← TABLE 1 of your Results chapter
│   ├── inference_bundle.pkl          ← model + preprocessor + transform (feeds the app)
│   └── config.yaml
│
├── requirements.txt
└── README.md
```

---

## How to run

**Prerequisite (once):** Python 3.11 must be installed — the pinned libraries
(`sdv==1.11.0` especially) do not support 3.12+. On macOS:
```bash
brew install python@3.11
```

**Run the whole analysis — one command:**
```bash
./run.sh
```
On first use this builds the `ml_env` virtual environment and installs
dependencies, then runs Stage 1 + Stage 2 end-to-end. Every figure, table and
metric appears in `results/`. Re-running is safe; setup is skipped once it exists.

**Launch the educator screening app:**
```bash
./app.sh
```
Opens in your browser. Upload a spreadsheet of student records (download the
blank template from the app, or try `app/example_students.csv`) to get a ranked
at-risk list and a per-student SHAP explanation. Run `./run.sh` at least once
first so the model bundle exists.

**Manual / Colab path:** each notebook is also organised into numbered cells
(`# CELL N` / `# ENG-CELL N`) you can paste into Jupyter/Colab. Locally, the
equivalent of `./run.sh` is:
```bash
python -m venv ml_env && source ml_env/bin/activate
pip install -r requirements.txt
python run_all.py
```

---

## 2. Run order — the complete chain

| Stage | File | What it does | Run when |
|---|---|---|---|
| **0** | `00_data_preparation_and_eda.py` | Raw Excel → exclusion rules → SHA-256 anonymisation → `anonymised_records.csv` + EDA tables/figures | The day the college hands you data |
| **1** | `01_pipeline_and_experiments.py` | Schema, preprocessing, splits, 3 baselines, XGBoost, SHAP, calibration, DeLong, McNemar, fairness | Now (synthetic) → re-run on real data |
| **2** | `02_model_engineering.py` | Lock baseline → SHAP-guided pruning → **E-XGBoost** → Wilcoxon comparison table | After Stage 1 on real data |

**Today (before real data):** run `./run.sh` — it runs Stage 1 + Stage 2 on
synthetic data and writes everything to `results/`. Everything should complete
green. That proves the machinery works.

**When real data arrives (the only changes you make):**
1. Put the college's file at `data/raw_college_records.xlsx`
2. Open Stage 0, PREP-CELL 2: fix `COLUMN_MAPPING` to match their actual column headers, set `TARGET_RULE`
3. Run Stage 0 → produces `anonymised_records.csv` + your exclusion-flow table
4. Open Stage 1, Cell 6: uncomment `REAL_DATA_PATH = "data/anonymised_records.csv"`, comment out `syn_df`
5. Run `./run.sh` again → all real results in `results/` (and the educator app now serves the real model)
6. Done — every figure and table for the thesis is in `results/`

---

## 3. The three contributions → where the evidence lives

| Contribution (defence-day script) | Evidence file |
|---|---|
| 1. "Original field data — first time in Ghana" | Stage 0 exclusion log + `anonymised_records.csv` provenance |
| 2. "Modern ML model — XGBoost + SHAP" | Stage 1 outputs: SHAP beeswarm, waterfalls, metrics |
| 3. "Engineered model (E-XGBoost) vs baseline" | `comparison_table.csv` with Wilcoxon p-values |

---

## 4. Output → thesis chapter mapping

| Thesis section | Comes from |
|---|---|
| Ch. 3 Methodology — participant flow | Stage 0 PREP-CELL 4 exclusion table |
| Ch. 4 Results — descriptives | `eda_descriptives.csv`, fail-rate-by-group tables |
| Ch. 4 Results — model comparison | Stage 1 metrics table + ROC/PR curves |
| Ch. 4 Results — Table 1 (engineering) | `comparison_table.csv` (baseline vs E-XGBoost, Δ, p) |
| Ch. 4 Results — explainability | SHAP beeswarm, 5 waterfall plots, dependence plots |
| Ch. 4 Results — calibration (H4) | `calibration_curves.png` + Brier scores |
| Ch. 4 Results — fairness | FNR/FPR/EOD tables (Stage 1 Cell 18) |
| Ch. 5 Discussion — H1 | DeLong results (Stage 1 Cell 16) |
| Ch. 5 Discussion — H2 | SHAP global ranking vs demographics |
| Ch. 5 Discussion — H3 | Tutor survey Likert results (separate, Q2 Section C) |

---

## 5. Decision rules already encoded (don't re-decide these)

- **Imbalance:** Stage 0 prints prevalence. 35–65% fail → natural distribution, no SMOTE. Outside that → compare SMOTE / scale_pos_weight / threshold-tuning by validation AUC-PR.
- **Exclusions:** absent / deferred / withheld / missing-outcome / duplicate / missing-core → removed and counted (proposal §4.2 rule).
- **Splits:** stratified 70/15/15 + temporal (train earlier cohorts, test latest) — `cohort_year` is preserved by Stage 0 for this.
- **Engineering verdict:** E-XGBoost better → "significant improvement". Equal with fewer features → "equivalent accuracy, leaner and more interpretable" — both valid (doc accepts parsimony gains).
- **Statistical tests:** DeLong for AUC, McNemar for label disagreement, Wilcoxon for the engineering comparison. p < 0.05.
- **Synthetic data:** pipeline testing ONLY. Never in any reported result.

---

## 6. Status

- [x] Stage 0 built and live-tested on a messy mock college file (528 raw → 488 clean, 0 identifier leaks)
- [x] Stage 1 built, syntax-valid, 20 cells
- [x] Stage 2 built and live-tested end-to-end (Wilcoxon table generates)
- [x] Full chain runs on Mac via `./run.sh` (synthetic) — all 17 `results/` artefacts generate green
- [x] Educator screening app built (`./app.sh`) and verified on the synthetic model
- [ ] Await HuSSREC reference number
- [ ] Real data from college → re-run `./run.sh` → results final → app serves real model