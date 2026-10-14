# CardioTriage 3D: Devpost & Unstop Official Submission Pack + 4-Minute Demo Video Script

> **Multimodal AI Hackathon 2026**  
> **Track:** Track A — Cardiovascular Risk Visualization & Prediction  
> **Author & Project Lead:** Gutta Jagath Chowdary (Aditya University)  
> **Repository:** `https://github.com/jagath0511-ui/CardioTriage3D`  
> **Live Web Application:** [https://cardiotriage-3d.onrender.com](https://cardiotriage-3d.onrender.com) (or local `http://localhost:8000`)

---

## 🎬 Stage 5: 4-Minute Demo Video Script (Timecoded & Action-Cued)

*Target Length: 3 Minutes 45 Seconds to 4 Minutes.*  
*Presenter: Gutta Jagath Chowdary.*  
*Tone: Confident, clinical, technically precise, and authoritative.*

---

### [0:00 - 0:35] The Clinical Problem & Introduction
* **[Screen Recording Cue]:** Start with camera on presenter, then switch to a full-screen view of the **CardioTriage 3D Cath-Lab Cockpit** showing the glistening 3D anatomical heart in dark mode with the amber FDA safety banner at the top.
* **[Spoken Narration]:**  
  *"Hello judges and fellow researchers. I’m Gutta Jagath Chowdary from Aditya University. Today, coronary artery disease remains the leading cause of death worldwide. Yet, according to major clinical registries, nearly 40% of elective invasive coronary angiographies show completely normal arteries—exposing patients to procedural complications and high costs. At the same time, emergency rooms often miss acute single-vessel blockages in the Left Anterior Descending artery—the dreaded 'Widowmaker'.*  
  *To bridge this critical divide between tabular EHR data and spatial clinical reality, we built **CardioTriage 3D**: a non-invasive, leakage-proof multi-vessel stenosis localization engine and interactive 3D anatomical digital twin."*

---

### [0:35 - 1:30] Live 3D Heart & Multi-Vessel Localization
* **[Screen Recording Cue]:** Open the "Patient Case Archive" dropdown at top-center. Select **"Preset 1: Triple-Vessel CAD (High-Risk Diffuse)"**.
* **[Action Cue]:** The 3D heart immediately loads. Orbit around the heart using the mouse. Zoom in on the LAD, LCX, and RCA arteries. Point out the glowing crimson arteries, the pulsating gold-amber plaque constriction ring, and the slowed blood-flow particles.
* **[Spoken Narration]:**  
  *"Here in our Cath-Lab Cockpit, we can instantly load real clinical profiles from the UCI Z-Alizadeh Sani cohort of 303 angiographically confirmed patients. Watch what happens when I select Patient #7—a 72-year-old male with confirmed multi-vessel disease.*  
  *Immediately, our procedural Three.js digital twin updates in real time. Notice the three primary coronary arteries:*  
  *The **LAD** glows crimson at 91.9% stenosis probability, displaying an active 3D plaque constriction ring. Blood-flow particles noticeably decelerate through the stenosis, simulating ischemic hypoperfusion.*  
  *The **LCX** and **RCA** also illuminate in crimson at 72.7% and 79.3% risk.*  
  *Crucially, the heart is not static: its contraction rate is dynamically synchronized to the patient's pulse rate of 68 beats per minute, while ventricular systolic amplitude reflects their echocardiographic ejection fraction."*

---

### [1:30 - 2:30] TreeSHAP Explainability & Doctor-in-Control Counterfactuals
* **[Screen Recording Cue]:** Click on the **LAD** artery directly in the 3D canvas (or click the LAD card in the left triage rail). The camera smoothly focuses on the LAD, and the right panel switches to the TreeSHAP explainability breakdown.
* **[Action Cue]:** Hover over the top risk drivers (Regional Wall Motion Abnormality, Typical Chest Pain, Age). Then scroll down to the "Counterfactual Risk Simulator" sliders on the right.
* **[Spoken Narration]:**  
  *"By clicking the LAD artery directly in 3D, our real-time TreeSHAP explainability engine reveals the exact clinical drivers behind this prediction. For this patient, Regional Wall Motion Abnormality across four myocardial segments contributes a massive +0.88 log-odds toward LAD occlusion, reinforced by Typical Anginal symptoms.*  
  *But we didn't want an AI that just reports passive numbers. Doctors need actionable control. Using our 'Doctor-in-Control' Counterfactual Simulator, let’s simulate optimal medical therapy.*  
  *Watch as I reduce his Systolic Blood Pressure from 130 down to 115 mmHg, lower his LDL from 90 to 70 mg/dL, and toggle smoking cessation to 'No'.*  
  *Instantly, the model recalculates: we see a 10% absolute risk reduction in the LCX and a 1.3% reduction in the LAD. Because our models are constrained with monotonic gradient boosting, we guarantee mathematically that therapeutic risk reductions never cause erratic, paradoxical risk spikes."*

---

### [2:30 - 3:20] Emergency Triage Sensitivity & Leakage-Proof Rigor
* **[Screen Recording Cue]:** Toggle the **"Triage Sensitivity"** switch from Standard ($\tau = 0.50$) to **"High Recall / Emergency Triage" ($\tau = 0.35$)**. Show the triage badges change to "HIGH-RISK URGENT CATHETERIZATION".
* **[Action Cue]:** Click the **"5-Fold CV Audit"** button on the header to reveal the modal with all 5-Fold Stratified Cross-Validation metrics and confusion matrices.
* **[Spoken Narration]:**  
  *"In an emergency department, a false negative is unacceptable. By switching to our High-Sensitivity Triage mode with a 0.35 decision threshold, our model escalates sensitivity to 93.1% for overall catheterization and 89.8% for the critical LAD artery.*  
  *We also take engineering integrity very seriously. As seen in our full 5-Fold Stratified Cross-Validation report, our pipeline strictly enforces Requirement 1d: Zero Target Leakage. All diagnostic targets—Cath, LAD, LCX, and RCA—are stripped from the 54-feature input matrix with automated unit assertions.*  
  *Our out-of-fold ROC-AUC reaches **0.9085** for overall CAD, **0.8474** for LAD stenosis, with 85.2% accuracy and 91.3% precision."*

---

### [3:20 - 4:00] Smart Clinical Note Parser & Conclusion
* **[Screen Recording Cue]:** Click the **"Smart Report Auto-Filler"** button. Paste a snippet of unstructured clinical notes. Click "Auto-Extract Biomarkers". Show 22+ variables populate instantly and update the 3D heart.
* **[Spoken Narration]:**  
  *"Finally, to eliminate manual data entry, our smart regex EHR parser extracts up to 54 clinical biomarkers directly from raw doctor consultation notes in milliseconds.*  
  *CardioTriage 3D transforms non-invasive clinical evaluation into an explainable, interactive spatial experience—reducing unnecessary invasive procedures while safeguarding critical cardiac patients.*  
  *Thank you for your time and consideration!"*

---

## 📋 Stage 6: Devpost & Unstop Official Submission Pack

### 1. Project Title
**CardioTriage 3D: Non-Invasive Multi-Vessel Stenosis Localization & Counterfactual Clinical Decision Support System**

### 2. Elevator Pitch (Short Description - 100 Words)
Nearly 40% of elective invasive coronary angiographies show non-obstructive arteries, while critical single-vessel LAD blockages are often missed during outpatient triage. **CardioTriage 3D** is a non-invasive, leakage-proof clinical workstation and Three.js 3D digital twin powered by a monotonicity-constrained multi-target XGBoost ensemble trained on the UCI Z-Alizadeh Sani cohort ($N=303$, 54 biomarkers). It predicts vessel-specific stenosis across the LAD, LCX, and RCA arteries, renders dynamic 3D plaque constriction rings and particle hemodynamics, provides real-time TreeSHAP explainability, and empowers doctors with interactive counterfactual "what-if" risk simulations.

---

### 3. Track Selection
**Track A: Cardiovascular Risk Visualization & Prediction**

---

### 4. Inspiration & The Problem
Coronary artery disease (CAD) diagnosis presents an expensive paradox in modern cardiology:
- Over 39% of patients referred for invasive coronary catheterization (ICA) have no obstructive disease, undergoing unnecessary arterial puncture, radiation, and healthcare expenditures exceeding \$10,000 per procedure.
- Conversely, emergency triage clinicians frequently misclassify acute single-vessel LAD ("Widowmaker") stenosis due to ambiguous ECG or atypical symptoms.
- Existing machine learning prototypes often suffer from **target leakage** (using catheterization outcomes to predict vessel blockages), **non-monotonic behavior** (where reducing LDL paradoxically increases predicted risk), and **black-box passivity** without anatomical visualization.

We set out to build an end-to-end, clinically grounded platform that combines non-invasive biomarkers with rigorous, leakage-proof machine learning and an interactive 3D anatomical digital twin.

---

### 5. What It Does
1. **Multi-Target Stenosis Prediction:** Simultaneously predicts overall CAD catheterization need (`Cath`, ROC-AUC 0.9085) and vessel-specific stenosis across the three primary coronary arteries: **LAD** (ROC-AUC 0.8474), **LCX** (ROC-AUC 0.7269), and **RCA** (ROC-AUC 0.7165).
2. **Interactive Three.js 3D Heart Digital Twin:**
   - Procedurally generated ventricular muscle with physical medical shaders.
   - Thicker Catmull-Rom spline coronary trees with major clinical bifurcations ($D_1$, $D_2$, $OM$, $AM$, $PDA$).
   - Dynamic stenosis color transition: Emerald ($<25\%$) $\rightarrow$ Amber ($25\text{--}50\%$) $\rightarrow$ Crimson ($>50\%$).
   - Pulsing 3D plaque constriction rings and blood-flow particle deceleration.
   - Real-time heartbeat contraction synchronized to patient pulse rate (`PR`) and systolic amplitude scaled to ejection fraction (`EF-TTE`).
3. **TreeSHAP Clinical Explainability:** Deconstructs vessel predictions into top risk-increasing drivers (e.g., Regional Wall Motion Abnormalities, ST elevation) and protective buffers (e.g., normal ECG, preserved EF).
4. **Doctor-in-Control Counterfactual Simulator:** Allows clinicians to adjust modifiable levers (blood pressure, LDL cholesterol, smoking, weight) and visualize instant delta risk reduction.
5. **Dual Sensitivity Triage Modes:** Standard outpatient operating point ($\tau = 0.50$) vs. High-Sensitivity emergency triage ($\tau = 0.35$, achieving **93.1% Cath Recall** and **89.8% LAD Recall**).
6. **Smart EHR Note Parser:** Natural language extraction engine that converts raw doctor consultation notes into structured biomarkers.

---

### 6. How We Built It
- **Machine Learning & Pipeline:** Python, XGBoost with monotonicity constraints (`monotone_constraints`), Scikit-Learn 5-Fold Stratified Cross-Validation, SHAP (TreeSHAP).
- **Backend API:** FastAPI, Uvicorn, Pydantic data schemas, zero-leakage unit assertion guardrails.
- **Frontend & 3D Engine:** Three.js, WebGL, OrbitControls, custom Catmull-Rom splines, particle systems, Vanilla JavaScript, CSS3 modern dark medical design system.
- **Cloud Deployment:** Vercel Python Serverless runtime with edge static caching and multi-route dispatch.

---

### 7. Challenges We Ran Into
1. **Target Leakage Prevention (Requirement 1d):** In the raw UCI dataset, `Cath`, `LAD`, `LCX`, and `RCA` are closely interrelated. We implemented a strict pre-training filter and runtime assertions that strip all target labels (case-insensitively) from the feature matrix before model training or inference.
2. **Preventing Non-Monotonic Clinical Anomalies:** Unconstrained decision trees can occasionally predict that a patient's risk *increases* when their blood pressure or cholesterol drops. We formulated and enforced directional monotonicity vectors across all 12 key biomarkers, guaranteeing clinical validity during counterfactual simulation.
3. **Three.js Anatomical Realism & Performance:** Ensuring high-fidelity physical shading, dynamic plaque rings, and hundreds of blood-flow particles run at 60 FPS in any standard browser without heavy multi-megabyte 3D asset downloads.

---

### 8. Accomplishments That We're Proud Of
- **0.9085 Out-of-Fold ROC-AUC** for overall CAD prediction and **0.8474 ROC-AUC** for the LAD artery across rigorous 5-fold stratified cross-validation.
- Zero target leakage with 100% automated test suite pass rate.
- Truly interactive 3D digital twin that synchronizes hemodynamics, plaque constriction, and pulse rate in the browser.
- Sub-15ms inference latency on FastAPI.

---

### 9. What We Learned
- How critical monotonicity constraints are for safety in clinical machine learning and counterfactual simulation.
- How spatial 3D visualization transforms tabular clinical predictions from abstract numbers into intuitive anatomical understanding.
- Best practices for optimizing medical decision thresholds between emergency triage (prioritizing recall) and outpatient screening (prioritizing precision).

---

### 10. What's Next for CardioTriage 3D
- Multi-center external validation on broader diverse clinical registries (UK Biobank, MIMIC-IV).
- Integration of DICOM CT coronary angiography (CCTA) imaging for true multimodal fusion.
- Deployment of on-device federated learning for privacy-preserving hospital intranet training.

---

### 11. Technology Stack Tags
`Python` • `FastAPI` • `XGBoost` • `TreeSHAP` • `Three.js` • `WebGL` • `Scikit-Learn` • `Pandas` • `NumPy` • `Vercel` • `Healthcare AI` • `Cardiovascular Risk` • `Digital Twin`

