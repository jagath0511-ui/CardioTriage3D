# CardioTriage 3D: Non-Invasive Multi-Vessel Coronary Stenosis Localization & Counterfactual Clinical Decision Support System

**Submission for Multimodal AI Hackathon 2026 — Track A: Cardiovascular Risk Visualization & Prediction**  
**Author:** Gutta Jagath Chowdary  
**Affiliation:** Aditya University  
**Date:** October 2026  
**Clinical Cohort:** UCI Machine Learning Repository Dataset #411 (Z-Alizadeh Sani Extension, $N=303$, 54 Features)  
**System Architecture:** Leakage-Proof Monotonic Multi-Target XGBoost Ensemble + Fast TreeSHAP + Interactive Three.js Hemodynamic Digital Twin  

---

## 1. Executive Summary & Clinical Problem Statement

### 1.1 The Clinical Dilemma of Elective Coronary Angiography
Coronary Artery Disease (CAD) remains the single largest cause of mortality globally. Invasive Coronary Angiography (ICA) represents the historical gold standard for establishing luminal diameter reduction ($\ge 50\%$ stenosis). However, contemporary registry data indicates that **up to 39.2% of elective ICAs yield normal or non-obstructive coronary vessels**, unnecessarily exposing low-to-intermediate risk patients to procedural risks (arterial dissection, retroperitoneal hemorrhage, contrast-induced nephropathy, radiation exposure) at an estimated direct cost exceeding \$10,000 per intervention.

Conversely, outpatient emergency and triage departments face the opposite catastrophe: acute single-vessel critical stenoses—specifically involving the **Left Anterior Descending (LAD) artery ("the widowmaker")**—can manifest with non-specific symptoms, leading to inadvertent discharge and catastrophic major adverse cardiovascular events (MACE).

### 1.2 Proposed Multimodal Solution
**CardioTriage 3D** addresses this dual failure mode by converting 54 non-invasive clinical biomarkers (demographics, symptom scores, physical examination, ECG waveforms, standard laboratory lipids, and transthoracic echocardiography) into:
1. **Multi-Target Stenosis Predictions:** Individualized probabilities for the three primary coronary arteries (**LAD**, **LCX**, **RCA**) and systemic catheterization requirement (**Cath**).
2. **Zero Target Leakage Verification (Rule 1d):** Formal structural exclusion of all diagnostic target labels from the training and inference pipeline, verified by automated runtime unit assertions.
3. **Monotonicity-Constrained Gradient Boosting:** Enforcing physiologically verified monotonic directions (e.g., higher LDL, blood pressure, and wall motion abnormalities never reduce predicted stenosis risk).
4. **Real-Time TreeSHAP Clinical Explainability:** Decomposing every vessel probability into patient-specific risk drivers and protective factors, categorizing them into non-modifiable vs. modifiable levers.
5. **Interactive Three.js 3D Anatomical Digital Twin:** An anatomical heart model that dynamically maps predicted stenosis probabilities onto coronary artery trees (color-coded Emerald $\rightarrow$ Amber $\rightarrow$ Crimson), renders pulsing 3D plaque constriction rings, modulates particle blood-flow velocities, and synchronizes myocardial contraction with patient pulse rate (`PR`) and ejection fraction (`EF-TTE`).
6. **Counterfactual "What-If" Simulation:** Equipping cardiologists and patients with interactive sliders to simulate the impact of lifestyle modifications and pharmacotherapy (e.g., statins, anti-hypertensives, smoking cessation) with real-time risk reduction deltas.

---

## 2. Cohort Characterization & Zero-Leakage Pipeline

### 2.1 Cohort Demographics (UCI Z-Alizadeh Sani Extension, $N=303$)
The dataset comprises $N=303$ consecutive adult patients referred for elective coronary angiography at the Tehran Heart Center. Ground-truth stenoses were established via quantitative coronary angiography (QCA) by board-certified interventional cardiologists:
- **Systemic CAD (`Cath`):** 216 positive (71.3% prevalence), 87 negative (28.7%).
- **LAD Stenosis ($\ge 50\%$):** 177 positive (58.4%), 126 negative (41.6%).
- **LCX Stenosis ($\ge 50\%$):** 119 positive (39.3%), 184 negative (60.7%).
- **RCA Stenosis ($\ge 50\%$):** 114 positive (37.6%), 189 negative (62.4%).

### 2.2 Feature Harmonization Across 5 Clinical Domains
A total of 54 non-invasive features are grouped across five standardized physiological domains:
1. **Demographics & Medical History (17 features):** Age (30–86 yrs), Sex, Weight, BMI, Diabetes Mellitus (`DM`), Hypertension (`HTN`), Current Smoker, Ex-Smoker, Family History (`FH`), Chronic Kidney Disease (`CRF`), Cerebrovascular Accident (`CVA`), Peripheral Vascular Disease (`PVD`), Dyslipidemia (`DL`), Airway Disease, Thyroid Disease, Weak Peripheral Pulses, Lung Rales.
2. **Vitals & Clinical Symptoms (13 features):** Systolic Blood Pressure (`BP`, 90–190 mmHg), Pulse Rate (`PR`, 50–110 bpm), Typical Chest Pain, Atypical Chest Pain, Non-anginal Chest Pain, Exertional Symptoms, Rest CP, Low BP Episode, Dyspnea, Edema, Murmur.
3. **Electrocardiographic Findings (7 features):** Rhythm (Sinus/AF), ST-Segment Elevation, ST-Segment Depression, T-Wave Inversion (`Tinversion`), Poor R-Wave Progression (`PRWP`), Q-Wave Abnormality, BBB.
4. **Laboratory Biomarkers (14 features):** Fasting Blood Sugar (`FBS`), Total Cholesterol, Low-Density Lipoprotein (`LDL`), High-Density Lipoprotein (`HDL`), Triglycerides (`TG`), Blood Urea Nitrogen (`BUN`), Serum Creatinine (`Cr`), Hemoglobin (`Hb`), Platelets, White Blood Cell Count (`WBC`), ESR, K, Na.
5. **Echocardiography & Hemodynamics (3 features):** Left Ventricular Ejection Fraction (`EF-TTE`, 15–65%), Regional Wall Motion Abnormality (`Region RWMA`, 0–4 regions), Valve Disease.

### 2.3 Strict Zero Target Leakage Protocol (Requirement 1d)
A frequent flaw in cardiovascular ML competitions is the inadvertent inclusion of downstream diagnostic targets (e.g., using `LAD` stenosis to predict `Cath`, or vice versa). 

In CardioTriage 3D:
- The target set $\mathcal{Y} = \{\text{Cath}, \text{LAD}, \text{LCX}, \text{RCA}\}$ is strictly isolated from the design matrix:
  $$X = \mathcal{D}_{\text{raw}} \setminus \{\text{Cath}, \text{LAD}, \text{LCX}, \text{RCA}, \text{Exertional CP}\}$$
  *(where `Exertional CP` was eliminated due to zero empirical variance).*
- Automated unit assertions are embedded across the data ingestion pipeline, cross-validation loop, and FastAPI inference endpoint:
  ```python
  assert not any(col.lower() in ["cath", "lad", "lcx", "rca"] for col in X_df.columns), \
      "FATAL VIOLATION: Target leakage detected in feature space!"
  ```
- Any client request attempting to supply `Cath`, `LAD`, `LCX`, or `RCA` inside the input JSON payload is immediately aborted with an `HTTP 400 Bad Request`.

---

## 3. Modeling Methodology & Monotonicity Constraints

### 3.1 Monotonicity-Constrained Gradient Boosted Ensembles
Standard unconstrained tree ensembles frequently produce non-monotonic decision surfaces due to sample noise, leading to dangerous clinical absurdities—such as predicting that a patient's risk *increases* after their LDL cholesterol drops from 160 to 70 mg/dL.

To guarantee clinically compliant behavior, we train separate XGBoost classifiers for each target $k \in \mathcal{Y}$, enforcing exact directional monotonicity vectors $M_k$:
$$\frac{\partial \hat{P}_k(X)}{\partial x_j} \ge 0 \quad \text{for} \quad j \in \mathcal{J}_{\text{risk}}$$
$$\frac{\partial \hat{P}_k(X)}{\partial x_j} \le 0 \quad \text{for} \quad j \in \mathcal{J}_{\text{protective}}$$

| Biomarker | Monotonicity Constraint | Clinical Rationale |
| :--- | :---: | :--- |
| **Systolic BP (`BP`)** | $+1$ (Positive) | Arterial wall shear stress and endothelial injury increase with hypertension. |
| **LDL Cholesterol (`LDL`)** | $+1$ (Positive) | Atherogenic apolipoprotein B deposition drives plaque formation. |
| **Age (`Age`)** | $+1$ (Positive) | Cumulative vascular calcification and intimal thickening increase with age. |
| **Triglycerides (`TG`) & `FBS`** | $+1$ (Positive) | Insulin resistance and atherogenic remnant lipoproteins promote atherogenesis. |
| **Current Smoker** | $+1$ (Positive) | Nicotine and oxidative stress induce endothelial dysfunction and vasoconstriction. |
| **ECG ST Elevation / Depression** | $+1$ (Positive) | Hallmark indicators of acute and subacute transmural/subendocardial ischemia. |
| **Regional Wall Motion (`Region RWMA`)**| $+1$ (Positive) | Direct downstream myocardial hypokinesis indicative of territory-specific ischemia. |
| **Ejection Fraction (`EF-TTE`)** | $-1$ (Negative) | Preserved systolic function indicates intact myocardial perfusion; depression signals severe ischemic insult. |
| **HDL Cholesterol (`HDL`)** | $-1$ (Negative) | Reverse cholesterol transport confers atheroprotective stabilization. |

### 3.2 Cross-Validation Protocol
To prevent data contamination and optimistic bias, hyperparameter tuning and model evaluation were performed strictly via **5-Fold Stratified Cross-Validation**:
- Folds were stratified across the primary label distribution (`Cath`).
- Preprocessing imputations (median for numeric, mode for categorical) and MinMax scalers were fitted exclusively on training folds and applied out-of-fold.
- All reported metrics reflect **out-of-fold (OOF) pooled predictions**, ensuring zero data leakage between splits.

---

## 4. Empirical Evaluation & Quantitative Results

### 4.1 Multi-Target Performance Benchmarks
Table 1 documents the out-of-fold performance across all four targets at the standard classification threshold ($\tau = 0.50$) and the high-sensitivity emergency triage threshold ($\tau = 0.35$).

#### Table 1: Comprehensive 5-Fold Stratified Cross-Validation Results ($N=303$)
| Target Vessel / CAD Endpoint | OOF ROC-AUC | Mean Fold ROC-AUC $\pm$ Std | Accuracy ($\tau=0.50$) | Precision ($\tau=0.50$) | Recall / Sens ($\tau=0.50$) | Specificity ($\tau=0.50$) | F1-Score ($\tau=0.50$) | Triage Recall ($\tau=0.35$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cath (Overall CAD)** | **0.9085** | $0.9119 \pm 0.059$ | **0.8515** | **0.9130** | **0.8750** | **0.7931** | **0.8936** | **0.9306 (93.1%)** |
| **LAD (Anterior Territory)** | **0.8474** | $0.8523 \pm 0.061$ | **0.7921** | **0.8276** | **0.8136** | **0.7619** | **0.8205** | **0.8983 (89.8%)** |
| **LCX (Circumflex Territory)** | **0.7269** | $0.7276 \pm 0.062$ | **0.6964** | **0.6031** | **0.6639** | **0.7174** | **0.6320** | **0.8151 (81.5%)** |
| **RCA (Inferior Territory)** | **0.7165** | $0.7182 \pm 0.060$ | **0.6601** | **0.5470** | **0.5614** | **0.7196** | **0.5541** | **0.7544 (75.4%)** |

### 4.2 Confusion Matrix Out-of-Fold Counts ($\tau = 0.50$)
- **Cath:** $TP=189, FP=18, TN=69, FN=27$ (Sensitivity: 87.5%, Specificity: 79.3%, Positive Predictive Value: 91.3%)
- **LAD:** $TP=144, FP=30, TN=96, FN=33$ (Sensitivity: 81.4%, Specificity: 76.2%, Positive Predictive Value: 82.8%)
- **LCX:** $TP=79, FP=52, TN=132, FN=40$ (Sensitivity: 66.4%, Specificity: 71.7%)
- **RCA:** $TP=64, FP=53, TN=136, FN=50$ (Sensitivity: 56.1%, Specificity: 72.0%)

### 4.3 High-Sensitivity Emergency Triage Threshold ($\tau = 0.35$)
In emergency room chest-pain evaluation, a false negative is clinically unacceptable (discharge of an unrecognized occlusive infarction). By dynamically toggling the triage operating threshold to $\tau = 0.35$:
- **Cath Sensitivity increases from 87.5% to 93.1%** (identifying 201 out of 216 diseased patients, missing only 15).
- **LAD Sensitivity surges from 81.4% to 89.8%** (capturing 159 out of 177 critical LAD lesions).
- **LCX Sensitivity increases to 81.5%** and **RCA Sensitivity to 75.4%**.

### 4.4 Anatomical Union Formulation $P(\text{Any Vessel})$
To evaluate multi-vessel coherence, we compute the cumulative probability of disease across any of the three major arteries assuming conditional anatomical independence:
$$P(\text{Any Vessel}) = 1 - \prod_{k \in \{\text{LAD}, \text{LCX}, \text{RCA}\}} \big(1 - P_k\big)$$
In out-of-fold testing, $P(\text{Any Vessel})$ demonstrates strong concordance with systemic $P(\text{Cath})$, achieving an area under the curve of $0.902$, corroborating that our vessel-specific models are anatomically aligned with overall catheterization findings.

---

## 5. TreeSHAP Explainability & Clinical Validation

### 5.1 Global Feature Importance Ranks
Using exact TreeSHAP algorithms ($O(TLD^2)$ complexity), we calculate mean absolute SHAP values across all 303 patients:

#### Top 10 Drivers for Overall CAD (`Cath`):
1. **Typical Chest Pain ($\text{SHAP} = 0.1661$):** Strongest non-invasive predictor of significant coronary stenosis.
2. **Region RWMA ($\text{SHAP} = 0.0539$):** Regional hypokinesis directly flags territories with diminished coronary reserve.
3. **Hypertension / `HTN` ($\text{SHAP} = 0.0525$):** Long-term vascular wall shear damage.
4. **Ejection Fraction / `EF-TTE` ($\text{SHAP} = 0.0466$):** Protective at high values; strongly risk-inducing when depressed ($<45\%$).
5. **Age ($\text{SHAP} = 0.0410$):** Accelerated calcification beyond age 60.
6. **Weight / Obesity ($\text{SHAP} = 0.0388$):** Modifiable metabolic risk factor.
7. **Non-Anginal Discomfort ($\text{SHAP} = 0.0370$):** Negative driver (protective against obstructive CAD when typical anginal features are absent).
8. **T-Wave Inversion (`Tinversion`) ($\text{SHAP} = 0.0340$):** Subendocardial ischemia marker on ECG.
9. **Diabetes Mellitus / `DM` ($\text{SHAP} = 0.0319$):** Microvascular and macrovascular accelerated atherosclerosis.
10. **Systolic BP / `BP` ($\text{SHAP} = 0.0275$):** Direct hemodynamic burden.

### 5.2 Local Explanation Decomposition
For any patient, the predicted log-odds $f(x)$ decompose into:
$$f(x) = \phi_0 + \sum_{j=1}^{54} \phi_j(x)$$
Where $\phi_0$ is the cohort base value and $\phi_j(x)$ represents the contribution of feature $j$. 

CardioTriage 3D splits these into:
- **Risk Increasers ($\phi_j > 0$):** Highlighted in soft red, identifying why this patient's risk is elevated.
- **Risk Reducers ($\phi_j < 0$):** Highlighted in soft emerald, identifying protective buffers (e.g., normal ECG, preserved EF, high HDL).
- **Modifiable vs. Non-Modifiable Flagging:** Modifiable biomarkers (BP, LDL, TG, FBS, Smoking, Weight) are highlighted with special interactive badges, feeding directly into the counterfactual simulator.

---

## 6. Three.js 3D Hemodynamic Digital Twin Engineering

### 6.1 Anatomical Geometry & Myocardial Shading
Rather than relying on generic, untextured 3D models, CardioTriage 3D renders a procedurally generated, anatomically oriented left and right ventricular muscular mass:
- **Myocardial Shader:** Uses Three.js `MeshPhysicalMaterial` with a rich burgundy tone (`#991b1b`), high surface roughness ($0.40$), and glistening clearcoat ($0.35$) to recreate living cardiac muscle under surgical theater lighting.
- **Great Vessels:** Anatomically placed ascending aorta with the classic aortic arch trifurcation (brachiocephalic trunk, left common carotid, left subclavian artery), pulmonary trunk, and inferior/superior vena cava.

### 6.2 Parametric Coronary Artery Splines
The primary coronary trees are defined via cubic Catmull-Rom space curves hugging the epicardial fat pads:
- **Left Anterior Descending (`LAD`):** Originates from the left main trunk, traverses the anterior interventricular sulcus down to the cardiac apex, giving off prominent diagonal branches ($D_1$, $D_2$).
- **Left Circumflex (`LCX`):** Courses along the left atrioventricular groove toward the obtuse margin, giving off obtuse marginal branches ($OM_1$).
- **Right Coronary Artery (`RCA`):** Courses down the right atrioventricular sulcus, giving off acute marginal branches and concluding as the posterior descending artery ($PDA$) along the inferior diaphragmatic surface.

### 6.3 Dynamic Plaque Rings & Hemodynamic Particle Simulation
- **Stenosis Color Interpolation:** Coronal tubes smoothly transition across three clinical zones:
  $$\text{Emerald } (\le 25\%) \longrightarrow \text{Amber } (25\text{--}50\%) \longrightarrow \text{Crimson } (\ge 50\%)$$
- **3D Plaque Constriction Torus:** For any vessel with $P_k \ge \tau$, a pulsing 3D gold-amber torus geometry (`TorusGeometry(0.12, 0.045, 16, 32)`) clamps onto the proximal segment of the affected artery, pulsating to indicate critical stenosis.
- **Particle Blood-Flow Dynamics:** Hundreds of erythrocyte particles flow along the Catmull-Rom spline curves. As a vessel's predicted stenosis rises, particle transit velocity slows down by up to 70%, visually communicating post-stenotic flow deceleration and ischemic hypoperfusion.
- **Physiological Rate-Responsive Contraction:** Ventricular scale and heartbeat pulsation oscillate at the patient's exact heart rate:
  $$f_{\text{beat}} = \frac{\text{PR}}{60} \text{ Hz}$$
  While myocardial systolic contraction amplitude scales proportionally with ejection fraction (`EF-TTE`):
  $$A_{\text{systole}} \propto \frac{\text{EF-TTE}}{100}$$

---

## 7. Doctor-in-Control Counterfactual Simulation

### 7.1 Counterfactual Formulation
A major critique of clinical AI is passivity: models output risk scores but offer no actionable pathways for risk mitigation. 

In CardioTriage 3D, the physician can modify any patient's modifiable biomarkers $X_{\text{mod}}$ (e.g., lowering blood pressure from 160 to 120 mmHg, titrating LDL from 160 to 70 mg/dL, or toggling smoking status to cessation) while locking non-modifiable history $X_{\text{fixed}}$:
$$\Delta P_k = \hat{P}_k(X_{\text{counterfactual}}) - \hat{P}_k(X_{\text{baseline}})$$

### 7.2 Monotonic Safety in Counterfactuals
Because our models enforce strict monotonicity constraints:
$$\Delta P_k \le 0 \quad \forall \Delta X_{\text{mod}} \in \text{Therapeutic Improvement}$$
Cardiologists are guaranteed that optimizing clinical guidelines will never produce an anomalous, paradoxical increase in predicted stenosis.

### 7.3 Clinical Case Demonstration (Patient #7, Preset 1)
- **Baseline:** 72-year-old male smoker with $BP=130$, $LDL=90$, $DM=1$, $RWMA=4$.
  - Baseline Predicted Risk: $\text{Cath} = 97.0\%$, $\text{LAD} = 91.9\%$, $\text{LCX} = 72.7\%$, $\text{RCA} = 79.3\%$.
- **Simulated Intervention (Smoking Cessation + BP Optimization to 115 mmHg + LDL Reduction to 70 mg/dL):**
  - $\Delta \text{LCX} = -10.03\%$ absolute risk reduction!
  - $\Delta \text{LAD} = -1.31\%$ absolute risk reduction.
  - Overall $P(\text{Any Vessel})$ decreases, visibly opening vessel lumen particles in the 3D twin.

---

## 8. Clinical Governance, Limitations & Ethical Safety

### 8.1 Clinical Limitations
1. **Single-Center Cohort ($N=303$):** While the Z-Alizadeh Sani extension represents an authentic, high-quality angiographic cohort, external validation across diverse multi-ethnic populations (e.g., UK Biobank, Framingham, Asian multi-center registries) is required prior to bedside deployment.
2. **Sub-Branch Resolution:** The current model localizes stenoses to the main arterial trunks (LAD, LCX, RCA) and primary diagonals/marginals. Secondary distal arborizations ($D_3$, posterolateral branches) require higher-resolution CCTA imaging integration.

### 8.2 Mandatory Regulatory Banner (Requirement 5)
In compliance with FDA software as a medical device (SaMD) and clinical decision support (CDS) guidelines, CardioTriage 3D persistently displays a prominent, high-contrast amber alert banner:
> **CLINICAL DECISION SUPPORT & EDUCATIONAL PROTOTYPE ONLY** — Predictions are statistical estimates derived from historical cohort data ($N=303$) and are NOT a substitute for formal diagnostic coronary angiography or professional clinical judgment.

---

## 9. Conclusion
CardioTriage 3D bridges the chasm between raw tabular machine learning and clinical spatial intuition. By uniting zero-leakage monotonicity-constrained gradient boosting, exact TreeSHAP attributions, real-time counterfactual optimization, and an interactive 3D Three.js anatomical digital twin, the system provides a scalable, non-invasive triage cockpit capable of reducing unnecessary invasive angiograms while safeguarding high-risk single-vessel patients.

---

## References
1. Alizadeh-Sani, Z., et al. (2017). "A data mining approach for diagnosis of coronary artery disease." *Computer Methods and Programs in Biomedicine*, 111(1), 52–61.
2. Patel, M. R., et al. (2010). "Low diagnostic yield of elective coronary angiography." *New England Journal of Medicine*, 362(10), 886–895.
3. Lundberg, S. M., & Lee, S.-I. (2017). "A unified approach to interpreting model predictions." *Advances in Neural Information Processing Systems (NeurIPS)*, 4765–4774.
4. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

