# CardioTriage 3D: Non-Invasive Multi-Vessel Stenosis Localization & Counterfactual Clinical Decision Support System

> **Submission for Multimodal AI Hackathon 2026 (Track A: Cardiovascular Risk Visualization & Prediction)**  
> **Author & Lead Engineer:** Gutta Jagath Chowdary (Aditya University)  
> **Clinical Cohort:** UCI Machine Learning Repository — Extension of Z-Alizadeh Sani Cohort (ID #411, $N=303$, 54 Features)  
> **Live Workstation URL:** [https://cardio-triage3-d.vercel.app](https://cardio-triage3-d.vercel.app) *(or local `http://localhost:8000`)*

---

## 🌟 Executive Summary
Nearly **40% of elective invasive coronary angiographies** reveal non-obstructive coronary arteries, subjecting stable patients to procedural risks, arterial trauma, radiation, and substantial healthcare costs. Conversely, acute single-vessel critical stenoses—particularly in the Left Anterior Descending ("Widowmaker") artery—are frequently under-triaged in busy emergency departments.

**CardioTriage 3D** is an interactive, browser-based clinical cockpit and hemodynamic 3D digital twin. Powered by a monotonicity-constrained multi-target XGBoost ensemble, the system predicts patient-specific stenosis risks across the three principal coronary arteries (**LAD**, **LCX**, **RCA**) alongside overall CAD catheterization requirement (**Cath**), while guaranteeing **Zero Target Leakage**, **real-time TreeSHAP feature attributions**, and **counterfactual "what-if" risk simulations**.

```
       +-----------------------------------------------------------------------+
       |   EHR / Lab / ECG / Echo Data (54 Non-Invasive Clinical Features)     |
       +-----------------------------------+-----------------------------------+
                                           |
                                           v
       +-----------------------------------------------------------------------+
       | Strict Zero Target Leakage Guard (Rule 1d: Assert No Cath/LAD/LCX/RCA)|
       +-----------------------------------+-----------------------------------+
                                           |
                                           v
       +-----------------------------------------------------------------------+
       |      5-Fold Stratified Cross-Validated Multi-Target XGBoost Ensemble   |
       |     + Monotonicity Constraints (Age, BP, LDL >= 0 | EF-TTE, HDL <= 0) |
       +-----------------------------------+-----------------------------------+
                                           |
                     +---------------------+---------------------+
                     |                                           |
                     v                                           v
  +-------------------------------------+     +-------------------------------------+
  |   TreeSHAP Clinical Explainability  |     |   Interactive Three.js 3D Twin      |
  |   - Top Risk Drivers (Increasers)   |     |   - Dynamic Stenosis Color Coded    |
  |   - Protective Factors (Decreasers) |     |   - Pulsing Plaque Constriction     |
  |   - Modifiable Lifestyle Levers     |     |   - Blood-Flow Particle Dynamics    |
  |   - Counterfactual "What-If" Delta  |     |   - Synced Pulse & Ejection Fraction|
  +-------------------------------------+     +-------------------------------------+
```

---

## ⚙️ Key Architectural Highlights

### 1. Zero Target Leakage Guarantee (Requirement 1d)
- **Unit Assertion Verified:** Target columns (`Cath`, `LAD`, `LCX`, `RCA`) are rigorously stripped from the input feature space prior to matrix transformation and training.
- Automated assertion inside training pipeline & runtime API:
  ```python
  assert not any(col.lower() in target_names_lower for col in X_df.columns), "Target leakage detected!"
  ```
- Any inference request attempting to inject ground-truth targets is rejected immediately with an HTTP 400 Bad Request.

### 2. Monotonicity-Constrained Multi-Target XGBoost Ensemble
- Enforces strict biological causality via monotonic gradient boosting constraints:
  - **Positive Monotonicity ($+1$):** `Age`, `BP` (Systolic BP), `LDL`, `TG`, `FBS`, `Current Smoker`, `Region RWMA`, `St Elevation`, `St Depression`, `Tinversion`.
  - **Negative Monotonicity ($-1$):** `EF-TTE` (Ejection Fraction), `HDL` (High-Density Lipoprotein).
- **Clinical Benefit:** Guarantees that lowering blood pressure or LDL in counterfactual simulations *monotonically reduces or preserves risk*, preventing erratic, non-physical AI artifacts.

### 3. Anatomical Vessel Union Probability $P(\text{Any Vessel})$
- Formulates the cumulative probability of multi-vessel obstructive coronary artery disease through complement probability:
  $$P(\text{Any Vessel}) = 1 - \big[(1 - P_{\text{LAD}}) \times (1 - P_{\text{LCX}}) \times (1 - P_{\text{RCA}})\big]$$
- Provides cross-target concordance audits against the systemic catheterization endpoint $P(\text{Cath})$.

### 4. Interactive Three.js 3D Digital Twin Cockpit
- Anatomically accurate Catmull-Rom spline coronary trees with major bifurcation branches ($D_1$, $D_2$, $OM$, $AM$, $PDA$).
- Dynamic color-interpolation shaders: Emerald (`#10b981`, $<25\%$) $\rightarrow$ Amber (`#f59e0b`, $25\text{--}50\%$) $\rightarrow$ Crimson (`#ef4444`, $>50\%$).
- Animated 3D torus plaque constriction rings and velocity-modulated blood-flow particles.
- Real-time myocardial contraction cycle synchronized with patient pulse rate (`PR`, 40–160 BPM) and ventricular amplitude scaled to ejection fraction (`EF-TTE`).

### 5. Dual Sensitivity Triage Decision Logic
- **Standard Clinical Operating Point ($\tau = 0.50$):** High precision for balanced outpatient diagnostic risk stratification.
- **High-Sensitivity Emergency Triage Point ($\tau = 0.35$):** Prioritizes recall (sensitivity), escalating high-risk patients to urgent catheterization/CCTA while minimizing missed stenoses (Cath Recall increases to **93.1%**, LAD Recall to **89.8%**).

---

## 📊 Rigorous Model Evaluation Results (5-Fold Stratified CV)

*Trained and evaluated on the full UCI Extension of Z-Alizadeh Sani cohort ($N=303$, 54 Features) across 5 Stratified Folds with zero target leakage:*

| Target Vessel / Endpoint | Out-of-Fold ROC-AUC | Accuracy ($\tau=0.50$) | Precision ($\tau=0.50$) | Recall ($\tau=0.50$) | F1-Score ($\tau=0.50$) | Triage Recall ($\tau=0.35$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cath (Overall CAD)** | **0.9085** *(Fold Mean: 0.9119)* | **0.8515** | **0.9130** | **0.8750** | **0.8936** | **0.9306 (93.1%)** |
| **LAD (Left Anterior Descending)** | **0.8474** *(Fold Mean: 0.8523)* | **0.7921** | **0.8276** | **0.8136** | **0.8205** | **0.8983 (89.8%)** |
| **LCX (Left Circumflex)** | **0.7269** *(Fold Mean: 0.7276)* | **0.6964** | **0.6031** | **0.6639** | **0.6320** | **0.8151 (81.5%)** |
| **RCA (Right Coronary Artery)** | **0.7165** *(Fold Mean: 0.7182)* | **0.6601** | **0.5470** | **0.5614** | **0.5541** | **0.7544 (75.4%)** |

### Confusion Matrix Breakdown (Out-of-Fold, $\tau=0.50$):
- **Cath:** $TP=189, FP=18, TN=69, FN=27$ (Sensitivity: 87.5%, Specificity: 79.3%, Precision: 91.3%)
- **LAD:** $TP=144, FP=30, TN=96, FN=33$ (Sensitivity: 81.4%, Specificity: 76.2%, Precision: 82.8%)
- **LCX:** $TP=79, FP=52, TN=132, FN=40$ (Sensitivity: 66.4%, Specificity: 71.7%)
- **RCA:** $TP=64, FP=53, TN=136, FN=50$ (Sensitivity: 56.1%, Specificity: 72.0%)

---

## 🗂️ Project Directory Structure

```
CardioTriage3D/
├── vercel.json                 # Vercel serverless deployment routing & file bundling
├── requirements.txt            # Locked Python runtime dependencies
├── README.md                   # Complete architectural documentation & benchmarks
├── backend/
│   ├── server.py               # FastAPI application with REST endpoints & static mount
│   ├── inspect_and_train.py    # Leakage-proof 5-fold training pipeline with monotonic XGBoost
│   ├── test_server.py          # End-to-end integration and leakage verification test suite
│   ├── pipeline.py             # Feature preprocessing & domain classification engine
│   ├── models_bundle.pkl       # Serialized models, TreeSHAP explainers, presets & metadata
│   └── evaluation_report.json  # Comprehensive 5-fold metrics, confusion matrices, SHAP ranks
├── data/
│   └── z_alizadeh_sani_ext.xlsx# UCI Z-Alizadeh Sani Extension Dataset (#411, N=303)
├── docs/
│   └── TECHNICAL_REPORT.md     # 6-Page Clinical & Technical Specification Report
└── frontend/
    ├── index.html              # Cath-Lab Cockpit UI with clinical safety banner
    ├── style.css               # Medical dark-mode design system & responsive layout
    ├── app.js                  # Frontend orchestration, REST client, counterfactual simulator
    ├── heart3d.js              # Three.js 3D myocardial twin, coronary splines, particles & shaders
    ├── reportParser.js         # Client-side / server-side regex clinical note biomarker extractor
    ├── libs/                   # Three.js, OrbitControls, GLTFLoader libraries
    └── assets/                 # Anatomical 3D cardiac reference models
```

---

## 🚀 Local Installation & Quickstart

### Prerequisites
- Python 3.10+ (Tested on Python 3.11 & 3.14)
- Modern WebGL-compatible browser (Chrome, Edge, Firefox, Safari)

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/jagath0511-ui/CardioTriage3D.git
cd CardioTriage3D

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Training Pipeline & Run Test Suite
```bash
# Verify model training and leakage assertions
python backend/inspect_and_train.py

# Run comprehensive integration test suite
python backend/test_server.py
```

### 4. Start the Application Server
```bash
python backend/server.py
```
Open your browser and navigate to:
```
http://localhost:8000
```

---

## 🔬 API Endpoint Specifications

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/init` | `GET` | Fetches 54 feature metadata, valid value domains, 4 clinical patient presets, and full 5-fold CV evaluation report. |
| `/api/predict` | `POST` | Executes leakage-proof inference, anatomical union probability, per-vessel TreeSHAP attributions, counterfactual delta risk, and triage classification. |
| `/api/parse_report` | `POST` | Smart EHR text parser extracting 22+ clinical variables from free-text doctor notes. |
| `/health` | `GET` | Verification endpoint returning loaded model status, feature count, and zero-leakage confirmation. |

---

## 🛡️ Clinical Safety Disclaimer

**CardioTriage 3D** is developed strictly for **clinical decision support, research evaluation, and educational demonstration** within the scope of the Multimodal AI Hackathon 2026. 

Predictions are statistical probability estimates derived from historical cohort data ($N=303$) and **must never replace formal diagnostic coronary angiography, computed tomography coronary angiography (CCTA), or the professional judgment of a board-certified cardiologist**.

