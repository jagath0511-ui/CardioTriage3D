"""
CardioTriage 3D - Stage 1 & Stage 2: Data Verification, Leakage-Proof ML Pipeline & Evaluation
Multimodal AI Hackathon 2026 - Track A: Cardiovascular Risk Visualization & Prediction

Strict Architectural Guarantees:
1. OFFICIAL UCI DATASET #411: Automated download & extraction of the 303-patient Z-Alizadeh Sani extension.
2. ZERO TARGET LEAKAGE GUARD: Targets ('Cath', 'LAD', 'LCX', 'RCA') are strictly excluded from feature matrix X
   with case-insensitive hard assertions.
3. PHYSIOLOGICAL MONOTONE CONSTRAINTS: Directional monotonicity in XGBoost (+1 for Age, BP, LDL, TG, FBS,
   Current Smoker, Region RWMA, ST Elevation, ST Depression, Tinversion; -1 for EF-TTE, HDL).
4. 5-FOLD STRATIFIED CROSS-VALIDATION: Complete out-of-fold evaluation with Accuracy, Precision, Recall,
   F1-Score, ROC-AUC, and Confusion Matrix (TP, FP, TN, FN) at both 0.50 and high-sensitivity 0.35 triage thresholds.
5. REAL CLINICAL PRESETS: 4 authentic patient profiles from the UCI cohort (Triple-Vessel, Isolated LAD,
   RCA + LCX Disease, Healthy Normal).
"""

import os
import io
import json
import pickle
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
DATA_URL = "https://archive.ics.uci.edu/static/public/411/extention+of+z+alizadeh+sani+dataset.zip"
EXCEL_PATH = os.path.join(DATA_DIR, "z_alizadeh_sani_ext.xlsx")
MODELS_BUNDLE_PATH = os.path.join(BACKEND_DIR, "models_bundle.pkl")
EVAL_REPORT_PATH = os.path.join(BACKEND_DIR, "evaluation_report.json")

TARGET_COLS = ["Cath", "LAD", "LCX", "RCA"]

# Feature metadata catalog: maps all 54 features into 5 clinical domains with modifiability flags
FEATURE_CATALOG = {
    # --- Demographics & History (17 features) ---
    "Age": {
        "domain": "Demographics & History",
        "label": "Age",
        "unit": "years",
        "type": "numeric",
        "modifiable": False,
        "description": "Patient age in chronological years"
    },
    "Sex": {
        "domain": "Demographics & History",
        "label": "Biological Sex",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Male"}, {"value": 0, "label": "Female"}],
        "description": "Biological sex (cleaned from raw 'Fmale' typos)"
    },
    "Weight": {
        "domain": "Demographics & History",
        "label": "Weight",
        "unit": "kg",
        "type": "numeric",
        "modifiable": True,
        "description": "Total body weight in kilograms"
    },
    "Length": {
        "domain": "Demographics & History",
        "label": "Height",
        "unit": "cm",
        "type": "numeric",
        "modifiable": False,
        "description": "Patient standing height in centimeters"
    },
    "BMI": {
        "domain": "Demographics & History",
        "label": "Body Mass Index",
        "unit": "kg/m²",
        "type": "numeric",
        "modifiable": True,
        "description": "Body mass index (Weight / Height²)"
    },
    "DM": {
        "domain": "Demographics & History",
        "label": "Diabetes Mellitus",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Clinical diagnosis of diabetes mellitus"
    },
    "HTN": {
        "domain": "Demographics & History",
        "label": "Hypertension",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Documented essential arterial hypertension"
    },
    "Current Smoker": {
        "domain": "Demographics & History",
        "label": "Current Smoker",
        "unit": "binary",
        "type": "binary",
        "modifiable": True,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Active cigarette smoking status"
    },
    "EX-Smoker": {
        "domain": "Demographics & History",
        "label": "Former Smoker",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "History of past tobacco smoking"
    },
    "FH": {
        "domain": "Demographics & History",
        "label": "Family History of CAD",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Premature coronary artery disease in first-degree relatives"
    },
    "Obesity": {
        "domain": "Demographics & History",
        "label": "Clinical Obesity",
        "unit": "binary",
        "type": "binary",
        "modifiable": True,
        "options": [{"value": 1, "label": "Yes (BMI >= 30)"}, {"value": 0, "label": "No"}],
        "description": "Clinical obesity classification"
    },
    "CRF": {
        "domain": "Demographics & History",
        "label": "Chronic Renal Failure",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Chronic kidney disease / impaired renal clearance"
    },
    "CVA": {
        "domain": "Demographics & History",
        "label": "Cerebrovascular Accident",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Prior ischemic stroke or transient ischemic attack"
    },
    "Airway disease": {
        "domain": "Demographics & History",
        "label": "Airway Disease",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Chronic obstructive pulmonary disease or asthma"
    },
    "Thyroid Disease": {
        "domain": "Demographics & History",
        "label": "Thyroid Disorder",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Diagnosed hypothyroidism or hyperthyroidism"
    },
    "CHF": {
        "domain": "Demographics & History",
        "label": "Congestive Heart Failure",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "History of symptomatic congestive heart failure"
    },
    "DLP": {
        "domain": "Demographics & History",
        "label": "Dyslipidemia",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Documented dyslipidemia / hypercholesterolemia"
    },

    # --- Vitals & Symptoms (13 features) ---
    "BP": {
        "domain": "Vitals & Symptoms",
        "label": "Systolic Blood Pressure",
        "unit": "mmHg",
        "type": "numeric",
        "modifiable": True,
        "description": "Resting systolic arterial blood pressure"
    },
    "PR": {
        "domain": "Vitals & Symptoms",
        "label": "Pulse Rate",
        "unit": "bpm",
        "type": "numeric",
        "modifiable": True,
        "description": "Resting heart rate in beats per minute"
    },
    "Edema": {
        "domain": "Vitals & Symptoms",
        "label": "Peripheral Edema",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Present"}, {"value": 0, "label": "Absent"}],
        "description": "Bilateral lower-extremity pitting edema"
    },
    "Weak Peripheral Pulse": {
        "domain": "Vitals & Symptoms",
        "label": "Weak Peripheral Pulse",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Diminished dorsalis pedis or radial arterial pulses"
    },
    "Lung rales": {
        "domain": "Vitals & Symptoms",
        "label": "Pulmonary Rales / Crackles",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Present"}, {"value": 0, "label": "Absent"}],
        "description": "Auscultatory bibasilar pulmonary crackles"
    },
    "Systolic Murmur": {
        "domain": "Vitals & Symptoms",
        "label": "Systolic Murmur",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Present"}, {"value": 0, "label": "Absent"}],
        "description": "Cardiac systolic ejection murmur"
    },
    "Diastolic Murmur": {
        "domain": "Vitals & Symptoms",
        "label": "Diastolic Murmur",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Present"}, {"value": 0, "label": "Absent"}],
        "description": "Cardiac diastolic murmur"
    },
    "Typical Chest Pain": {
        "domain": "Vitals & Symptoms",
        "label": "Typical Anginal Pain",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Substernal chest pressure provoked by exertion and relieved by rest"
    },
    "Dyspnea": {
        "domain": "Vitals & Symptoms",
        "label": "Dyspnea on Exertion",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Shortness of breath with minimal or moderate exertion"
    },
    "Function Class": {
        "domain": "Vitals & Symptoms",
        "label": "NYHA Functional Class",
        "unit": "class (0-3)",
        "type": "categorical",
        "modifiable": False,
        "options": [{"value": 0, "label": "Class 0 (None)"}, {"value": 1, "label": "Class I"}, {"value": 2, "label": "Class II"}, {"value": 3, "label": "Class III"}],
        "description": "New York Heart Association symptomatic functional classification"
    },
    "Atypical": {
        "domain": "Vitals & Symptoms",
        "label": "Atypical Chest Pain",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Chest pain lacking one or more classical anginal criteria"
    },
    "Nonanginal": {
        "domain": "Vitals & Symptoms",
        "label": "Non-Anginal Chest Discomfort",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Pleuritic or sharp non-cardiac chest discomfort"
    },
    "LowTH Ang": {
        "domain": "Vitals & Symptoms",
        "label": "Low-Threshold Angina",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Angina triggered by minimal physical or emotional stress"
    },

    # --- ECG Findings (7 features) ---
    "Q Wave": {
        "domain": "ECG Findings",
        "label": "Pathological Q Wave",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Present"}, {"value": 0, "label": "Absent"}],
        "description": "Pathologic Q waves on 12-lead ECG suggesting prior myocardial infarction"
    },
    "St Elevation": {
        "domain": "ECG Findings",
        "label": "ST-Segment Elevation",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Present"}, {"value": 0, "label": "Absent"}],
        "description": "Acute or persistent ST-segment elevation"
    },
    "St Depression": {
        "domain": "ECG Findings",
        "label": "ST-Segment Depression",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Present"}, {"value": 0, "label": "Absent"}],
        "description": "Horizontal or down-sloping ST depression >= 1mm"
    },
    "Tinversion": {
        "domain": "ECG Findings",
        "label": "T-Wave Inversion",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Present"}, {"value": 0, "label": "Absent"}],
        "description": "Symmetric negative T-waves indicating myocardial ischemia"
    },
    "LVH": {
        "domain": "ECG Findings",
        "label": "Left Ventricular Hypertrophy",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Sokolow-Lyon or Cornell voltage criteria for LVH"
    },
    "Poor R Progression": {
        "domain": "ECG Findings",
        "label": "Poor R-Wave Progression",
        "unit": "binary",
        "type": "binary",
        "modifiable": False,
        "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
        "description": "Loss of normal anterior R-wave amplitude progression across V1-V3"
    },
    "BBB": {
        "domain": "ECG Findings",
        "label": "Bundle Branch Block",
        "unit": "categorical",
        "type": "categorical",
        "modifiable": False,
        "options": [{"value": 0, "label": "Normal / None"}, {"value": 1, "label": "LBBB"}, {"value": 2, "label": "RBBB"}],
        "description": "Intraventricular conduction defect (0: None, 1: LBBB, 2: RBBB)"
    },

    # --- Lab Biomarkers (14 features) ---
    "FBS": {
        "domain": "Lab Biomarkers",
        "label": "Fasting Blood Sugar",
        "unit": "mg/dL",
        "type": "numeric",
        "modifiable": True,
        "description": "Serum glucose level following an overnight 8-hour fast"
    },
    "CR": {
        "domain": "Lab Biomarkers",
        "label": "Serum Creatinine",
        "unit": "mg/dL",
        "type": "numeric",
        "modifiable": False,
        "description": "Serum creatinine biomarker of kidney filtration function"
    },
    "TG": {
        "domain": "Lab Biomarkers",
        "label": "Triglycerides",
        "unit": "mg/dL",
        "type": "numeric",
        "modifiable": True,
        "description": "Circulating serum triglyceride concentration"
    },
    "LDL": {
        "domain": "Lab Biomarkers",
        "label": "Low-Density Lipoprotein (LDL-C)",
        "unit": "mg/dL",
        "type": "numeric",
        "modifiable": True,
        "description": "Atherogenic low-density lipoprotein cholesterol"
    },
    "HDL": {
        "domain": "Lab Biomarkers",
        "label": "High-Density Lipoprotein (HDL-C)",
        "unit": "mg/dL",
        "type": "numeric",
        "modifiable": True,
        "description": "Cardioprotective high-density lipoprotein cholesterol"
    },
    "BUN": {
        "domain": "Lab Biomarkers",
        "label": "Blood Urea Nitrogen",
        "unit": "mg/dL",
        "type": "numeric",
        "modifiable": False,
        "description": "Blood urea nitrogen waste concentration"
    },
    "ESR": {
        "domain": "Lab Biomarkers",
        "label": "Erythrocyte Sedimentation Rate",
        "unit": "mm/hr",
        "type": "numeric",
        "modifiable": False,
        "description": "Non-specific systemic inflammatory erythrocyte sedimentation rate"
    },
    "HB": {
        "domain": "Lab Biomarkers",
        "label": "Hemoglobin",
        "unit": "g/dL",
        "type": "numeric",
        "modifiable": False,
        "description": "Total circulating blood hemoglobin concentration"
    },
    "K": {
        "domain": "Lab Biomarkers",
        "label": "Serum Potassium",
        "unit": "mEq/L",
        "type": "numeric",
        "modifiable": False,
        "description": "Serum potassium electrolyte concentration"
    },
    "Na": {
        "domain": "Lab Biomarkers",
        "label": "Serum Sodium",
        "unit": "mEq/L",
        "type": "numeric",
        "modifiable": False,
        "description": "Serum sodium electrolyte concentration"
    },
    "WBC": {
        "domain": "Lab Biomarkers",
        "label": "White Blood Cell Count",
        "unit": "cells/µL",
        "type": "numeric",
        "modifiable": False,
        "description": "Total leukocyte cell count"
    },
    "Lymph": {
        "domain": "Lab Biomarkers",
        "label": "Lymphocyte Percentage",
        "unit": "%",
        "type": "numeric",
        "modifiable": False,
        "description": "Percentage of lymphocytes in differential leukocyte count"
    },
    "Neut": {
        "domain": "Lab Biomarkers",
        "label": "Neutrophil Percentage",
        "unit": "%",
        "type": "numeric",
        "modifiable": False,
        "description": "Percentage of neutrophils in differential leukocyte count"
    },
    "PLT": {
        "domain": "Lab Biomarkers",
        "label": "Platelet Count",
        "unit": "x10³/µL",
        "type": "numeric",
        "modifiable": False,
        "description": "Total circulating thrombocyte platelet count"
    },

    # --- Echo & Wall Motion (3 features) ---
    "EF-TTE": {
        "domain": "Echo & Wall Motion",
        "label": "Ejection Fraction (TTE)",
        "unit": "%",
        "type": "numeric",
        "modifiable": False,
        "description": "Transthoracic echocardiographic left ventricular ejection fraction"
    },
    "Region RWMA": {
        "domain": "Echo & Wall Motion",
        "label": "Regional Wall Motion Abnormality",
        "unit": "regions (0-4)",
        "type": "categorical",
        "modifiable": False,
        "options": [{"value": 0, "label": "0 (Normal)"}, {"value": 1, "label": "1 Region"}, {"value": 2, "label": "2 Regions"}, {"value": 3, "label": "3 Regions"}, {"value": 4, "label": "4 Regions"}],
        "description": "Number of akinetic or hypokinetic myocardial wall motion segments"
    },
    "VHD": {
        "domain": "Echo & Wall Motion",
        "label": "Valvular Heart Disease",
        "unit": "severity",
        "type": "categorical",
        "modifiable": False,
        "options": [{"value": 0, "label": "Normal / None"}, {"value": 1, "label": "Mild"}, {"value": 2, "label": "Moderate"}, {"value": 3, "label": "Severe"}],
        "description": "Severity of valvular stenosis or regurgitant heart disease"
    }
}


def download_dataset() -> str:
    """Step 1: Download official UCI Dataset #411 zip and extract .xlsx."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(EXCEL_PATH) and os.path.getsize(EXCEL_PATH) > 1000:
        print(f"[Step 1] Verified existing UCI Dataset #411 at: {EXCEL_PATH} ({os.path.getsize(EXCEL_PATH)} bytes)")
        return EXCEL_PATH

    print(f"[Step 1] Downloading official UCI Dataset #411 from: {DATA_URL}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(DATA_URL, headers=headers)
    with urllib.request.urlopen(req) as response:
        zip_bytes = response.read()

    print(f"[Step 1] Extracting excel file ({len(zip_bytes)} bytes downloaded)...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for file_info in z.infolist():
            if file_info.filename.endswith(".xlsx"):
                with z.open(file_info) as source, open(EXCEL_PATH, "wb") as target:
                    target.write(source.read())
                print(f"[Step 1] Successfully extracted and saved: {EXCEL_PATH}")
                return EXCEL_PATH

    raise FileNotFoundError("Could not locate .xlsx file within the UCI Dataset #411 archive.")


def inspect_and_preprocess():
    """
    Step 2, 3, 4:
    - Inspect columns and target distributions
    - Enforce Strict Zero Target Leakage Guard with case-insensitive assert
    - Handle string typos and zero-variance columns
    - Map features to clinical domains and modifiable flags
    """
    excel_path = download_dataset()
    raw_df = pd.read_excel(excel_path)
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    print("\n" + "="*80)
    print("STEP 2: DATA INSPECTION & TARGET DISTRIBUTIONS")
    print("="*80)
    print(f"Dataset Raw Shape: {raw_df.shape[0]} patients, {raw_df.shape[1]} raw columns")
    print(f"Raw Column Names ({len(raw_df.columns)}):")
    print(", ".join(raw_df.columns.tolist()))

    print("\nTarget Value Counts:")
    for target in TARGET_COLS:
        if target in raw_df.columns:
            vc = raw_df[target].value_counts(dropna=False).to_dict()
            print(f"  - {target:5s}: {vc}")
        else:
            raise KeyError(f"Expected target '{target}' missing from raw dataset!")

    # Check zero-variance / constant columns
    constant_columns = [c for c in raw_df.columns if raw_df[c].nunique(dropna=False) <= 1]
    print(f"\nZero-Variance / Constant Columns Detected: {constant_columns}")

    # Standardize binary targets
    targets = {
        "Cath": (raw_df["Cath"].astype(str).str.strip().str.upper() == "CAD").astype(int),
        "LAD": (raw_df["LAD"].astype(str).str.strip().str.upper() == "STENOTIC").astype(int),
        "LCX": (raw_df["LCX"].astype(str).str.strip().str.upper() == "STENOTIC").astype(int),
        "RCA": (raw_df["RCA"].astype(str).str.strip().str.upper() == "STENOTIC").astype(int),
    }

    # =========================================================================
    # STEP 3: STRICT TARGET LEAKAGE GUARD (Requirement 1d)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: STRICT TARGET LEAKAGE GUARD (REQUIREMENT 1d)")
    print("="*80)
    X_df = raw_df.drop(columns=TARGET_COLS).copy()

    # Case-insensitive verification
    target_names_lower = {t.lower() for t in TARGET_COLS}
    feature_names_lower = {c.lower() for c in X_df.columns}
    leakage_intersection = target_names_lower.intersection(feature_names_lower)

    assert len(leakage_intersection) == 0, (
        f"CRITICAL TARGET LEAKAGE DETECTED! Found targets in features: {leakage_intersection}"
    )
    assert not any(col.lower() in target_names_lower for col in X_df.columns), (
        "Hard assert failed: target column found in feature matrix X!"
    )
    print("[LEAKAGE GUARD PASSED] Verified 100% disjoint feature space:")
    print(f"  Excluded Target Set: {TARGET_COLS}")
    print(f"  Case-insensitive check: {target_names_lower} disjoint from feature matrix columns.")

    # =========================================================================
    # STEP 4: PREPROCESS FEATURES & DOMAIN CLASSIFICATION
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: PREPROCESSING & CLINICAL DOMAIN CLASSIFICATION")
    print("="*80)

    # Drop constant columns (e.g., 'Exertional CP')
    dropped_cols = []
    for c in list(X_df.columns):
        if X_df[c].nunique(dropna=False) <= 1:
            dropped_cols.append(c)
            X_df = X_df.drop(columns=[c])
    print(f"Dropped zero-variance columns: {dropped_cols}")

    # Handle string typos and category conversions
    # Typos handled: 'Fmale' -> Female, 'Y'/'N' -> 1/0, VHD multi-level, BBB conduction
    for c in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[c]):
            s = X_df[c].astype(str).str.strip()
            unique_vals = set(s.unique())
            if unique_vals.issubset({"Y", "N"}):
                X_df[c] = (s == "Y").astype(int)
            elif unique_vals.issubset({"Male", "Fmale"}):
                # Correct "Fmale" typo -> 0 (Female), Male -> 1
                X_df[c] = (s == "Male").astype(int)
            elif c == "VHD":
                vhd_map = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}
                X_df[c] = s.map(vhd_map).fillna(0).astype(int)
            elif c == "BBB":
                bbb_map = {"N": 0, "LBBB": 1, "RBBB": 2}
                X_df[c] = s.map(bbb_map).fillna(0).astype(int)
            else:
                X_df[c] = pd.factorize(s)[0]

    X_df = X_df.astype(float)
    assert all(pd.api.types.is_numeric_dtype(X_df[c]) for c in X_df.columns), "All features must be numeric float64!"
    print(f"Final clean feature matrix X shape: {X_df.shape[0]} rows, {X_df.shape[1]} features.")

    # Verify every feature is in catalog and classify into domains
    domain_counts = {}
    modifiable_count = 0
    feature_metadata = {}

    for col in X_df.columns:
        cat = FEATURE_CATALOG.get(col, {
            "domain": "Vitals & Symptoms",
            "label": col,
            "unit": "",
            "type": "numeric",
            "modifiable": False,
            "description": col
        })
        dom = cat["domain"]
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        if cat.get("modifiable", False):
            modifiable_count += 1

        meta = {
            "name": col,
            "label": cat["label"],
            "domain": dom,
            "unit": cat["unit"],
            "type": cat["type"],
            "modifiable": cat.get("modifiable", False),
            "description": cat.get("description", ""),
            "min": float(X_df[col].min()),
            "max": float(X_df[col].max()),
            "median": float(X_df[col].median()),
            "mean": float(round(X_df[col].mean(), 2)),
            "std": float(round(X_df[col].std(), 2)),
            "default": float(X_df[col].median())
        }
        if "options" in cat:
            meta["options"] = cat["options"]
        feature_metadata[col] = meta

    print("\nClinical Domains Breakdown (5 Domains):")
    for dom, cnt in domain_counts.items():
        print(f"  - {dom:25s}: {cnt:2d} features")
    print(f"Clinically Modifiable Features for Sensitivity Analysis: {modifiable_count} / {len(X_df.columns)}")

    return X_df, targets, feature_metadata, dropped_cols


def run_pipeline():
    """
    Step 5, 6, 7:
    - Apply physiologically sound monotone constraints
    - Run 5-Fold Stratified CV with standard (0.50) and high-sensitivity (0.35) thresholds
    - Fit final production models and TreeExplainer
    - Extract 4 real patient presets
    - Save models_bundle.pkl and evaluation_report.json
    """
    X_df, targets, feature_metadata, dropped_cols = inspect_and_preprocess()
    feature_names = list(X_df.columns)

    # =========================================================================
    # STEP 5: PHYSIOLOGICALLY SOUND MONOTONE CONSTRAINTS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: PHYSIOLOGICALLY SOUND MONOTONE CONSTRAINTS")
    print("="*80)
    constraint_dict = {
        # Established clinical positive associations (higher value -> higher risk)
        "Age": 1,
        "BP": 1,
        "LDL": 1,
        "TG": 1,
        "FBS": 1,
        "Current Smoker": 1,
        "Region RWMA": 1,
        "St Elevation": 1,
        "St Depression": 1,
        "Tinversion": 1,
        # Established clinical protective / inverse associations (higher value -> lower risk)
        "EF-TTE": -1,
        "HDL": -1
    }
    constraints = tuple(constraint_dict.get(c, 0) for c in feature_names)
    print("Applied Directional Monotone Constraints in XGBoost:")
    for feat, direct in constraint_dict.items():
        if feat in feature_names:
            arrow = "(+1) Risk Increasing" if direct == 1 else "(-1) Risk Reducing / Protective"
            print(f"  - {feat:15s}: {arrow}")

    # =========================================================================
    # STEP 6: 5-FOLD STRATIFIED CROSS-VALIDATION
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: 5-FOLD STRATIFIED CROSS-VALIDATION (STANDARD 0.50 & TRIAGE 0.35)")
    print("="*80)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = {}
    production_models = {}
    production_explainers = {}
    top_importances = {}

    for tgt_name, y_series in targets.items():
        y = y_series.values
        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        scale_pos_weight = neg_count / max(1, pos_count)

        oof_probs = np.zeros(len(y))
        fold_records = []

        xgb_params = {
            "n_estimators": 75,
            "max_depth": 3,
            "learning_rate": 0.06,
            "scale_pos_weight": scale_pos_weight,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "monotone_constraints": constraints,
            "random_state": 42,
            "eval_metric": "logloss"
        }

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_df, y), 1):
            clf = XGBClassifier(**xgb_params)
            clf.fit(X_df.iloc[train_idx], y[train_idx])
            probs_fold = clf.predict_proba(X_df.iloc[test_idx])[:, 1]
            oof_probs[test_idx] = probs_fold

            # Per fold metrics at 0.50
            preds_fold_50 = (probs_fold >= 0.50).astype(int)
            fold_records.append({
                "fold": fold_idx,
                "accuracy": float(accuracy_score(y[test_idx], preds_fold_50)),
                "precision": float(precision_score(y[test_idx], preds_fold_50, zero_division=0)),
                "recall": float(recall_score(y[test_idx], preds_fold_50, zero_division=0)),
                "f1": float(f1_score(y[test_idx], preds_fold_50, zero_division=0)),
                "roc_auc": float(roc_auc_score(y[test_idx], probs_fold))
            })

        # Overall Out-Of-Fold Evaluation at Standard Threshold (0.50)
        p50 = (oof_probs >= 0.50).astype(int)
        tn50, fp50, fn50, tp50 = confusion_matrix(y, p50).ravel()
        acc50 = accuracy_score(y, p50)
        prec50 = precision_score(y, p50, zero_division=0)
        rec50 = recall_score(y, p50, zero_division=0)
        f1_50 = f1_score(y, p50, zero_division=0)
        auc = roc_auc_score(y, oof_probs)

        # Overall Out-Of-Fold Evaluation at High-Sensitivity Triage Threshold (0.35)
        p35 = (oof_probs >= 0.35).astype(int)
        tn35, fp35, fn35, tp35 = confusion_matrix(y, p35).ravel()
        acc35 = accuracy_score(y, p35)
        prec35 = precision_score(y, p35, zero_division=0)
        rec35 = recall_score(y, p35, zero_division=0)
        f1_35 = f1_score(y, p35, zero_division=0)

        cv_summary = {
            "target": tgt_name,
            "total_samples": len(y),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_prevalence_pct": round(pos_count / len(y) * 100, 1),
            "roc_auc": float(round(auc, 4)),
            "threshold_0_50": {
                "threshold": 0.50,
                "accuracy": float(round(acc50, 4)),
                "precision": float(round(prec50, 4)),
                "recall": float(round(rec50, 4)),
                "f1_score": float(round(f1_50, 4)),
                "confusion_matrix": {"tp": int(tp50), "fp": int(fp50), "tn": int(tn50), "fn": int(fn50)}
            },
            "threshold_0_35_triage": {
                "threshold": 0.35,
                "accuracy": float(round(acc35, 4)),
                "precision": float(round(prec35, 4)),
                "recall": float(round(rec35, 4)),
                "f1_score": float(round(f1_35, 4)),
                "confusion_matrix": {"tp": int(tp35), "fp": int(fp35), "tn": int(tn35), "fn": int(fn35)}
            },
            "per_fold_mean": {
                "accuracy": float(round(np.mean([f["accuracy"] for f in fold_records]), 4)),
                "accuracy_std": float(round(np.std([f["accuracy"] for f in fold_records]), 4)),
                "precision": float(round(np.mean([f["precision"] for f in fold_records]), 4)),
                "recall": float(round(np.mean([f["recall"] for f in fold_records]), 4)),
                "f1": float(round(np.mean([f["f1"] for f in fold_records]), 4)),
                "roc_auc": float(round(np.mean([f["roc_auc"] for f in fold_records]), 4))
            },
            "folds": fold_records
        }
        cv_results[tgt_name] = cv_summary

        print(f"Target: {tgt_name:5s} | Preval: {cv_summary['positive_prevalence_pct']:4.1f}% | OOF ROC-AUC: {auc:.4f}")
        print(f"  [Standard T=0.50] Acc: {acc50:.4f} | Prec: {prec50:.4f} | Rec: {rec50:.4f} | F1: {f1_50:.4f} | "
              f"TP:{tp50:3d} FP:{fp50:2d} TN:{tn50:3d} FN:{fn50:2d}")
        print(f"  [Triage   T=0.35] Acc: {acc35:.4f} | Prec: {prec35:.4f} | Rec: {rec35:.4f} | F1: {f1_35:.4f} | "
              f"TP:{tp35:3d} FP:{fp35:2d} TN:{tn35:3d} FN:{fn35:2d}")

        # Train final production model on full 303 cohort
        final_model = XGBClassifier(**xgb_params)
        final_model.fit(X_df, y)
        production_models[tgt_name] = final_model

        # Train SHAP TreeExplainer
        explainer = shap.TreeExplainer(final_model)
        production_explainers[tgt_name] = explainer

        # Top feature importances
        imps = final_model.feature_importances_
        sorted_idx = np.argsort(imps)[::-1]
        top_importances[tgt_name] = [
            {
                "feature": feature_names[i],
                "label": feature_metadata[feature_names[i]]["label"],
                "domain": feature_metadata[feature_names[i]]["domain"],
                "modifiable": feature_metadata[feature_names[i]]["modifiable"],
                "importance": float(round(imps[i], 4))
            }
            for i in sorted_idx[:15]
        ]

    # =========================================================================
    # STEP 7: EXTRACT 4 REAL PATIENT PRESETS & SAVE BUNDLE
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 7: REAL CLINICAL PRESET EXTRACTION & BUNDLE SERIALIZATION")
    print("="*80)

    # 4 Authentic Presets matching requirements:
    # 1. Triple-Vessel CAD (Cath=1, LAD=1, LCX=1, RCA=1)
    # 2. Isolated LAD Stenosis (Cath=1, LAD=1, LCX=0, RCA=0)
    # 3. RCA + LCX Disease (Cath=1, LAD=0, LCX=1, RCA=1)
    # 4. Healthy Normal Baseline (Cath=0, LAD=0, LCX=0, RCA=0)
    cath_arr = targets["Cath"].values
    lad_arr = targets["LAD"].values
    lcx_arr = targets["LCX"].values
    rca_arr = targets["RCA"].values

    idx_tvd = int(np.where((cath_arr == 1) & (lad_arr == 1) & (lcx_arr == 1) & (rca_arr == 1))[0][1]) # index 7
    idx_lad = int(np.where((cath_arr == 1) & (lad_arr == 1) & (lcx_arr == 0) & (rca_arr == 0))[0][0]) # index 2
    idx_rca_lcx = int(np.where((cath_arr == 1) & (lad_arr == 0) & (lcx_arr == 1) & (rca_arr == 1))[0][0]) # index 9
    idx_normal = int(np.where((cath_arr == 0) & (lad_arr == 0) & (lcx_arr == 0) & (rca_arr == 0))[0][4]) # index 17

    presets = [
        {
            "id": "preset_triple_vessel",
            "name": "Triple-Vessel CAD (High-Risk Diffuse)",
            "subtitle": "Severe multivessel coronary obstruction",
            "description": "72yo Diabetic Male smoker with severe diffuse CAD, extensive anterior & inferior wall motion abnormality (RWMA 4), and typical anginal pain.",
            "patient_index": idx_tvd,
            "ground_truth": {"Cath": "CAD", "LAD": "Stenotic", "LCX": "Stenotic", "RCA": "Stenotic"},
            "features": {col: float(X_df.iloc[idx_tvd][col]) for col in feature_names}
        },
        {
            "id": "preset_isolated_lad",
            "name": "Isolated LAD Stenosis ('Widowmaker' Profile)",
            "subtitle": "Critical anterior wall ischemia with depressed ejection fraction",
            "description": "54yo Male smoker with RWMA anterior wall motion hypokinesis (score 2), reduced EF (40%), and isolated critical LAD stenosis.",
            "patient_index": idx_lad,
            "ground_truth": {"Cath": "CAD", "LAD": "Stenotic", "LCX": "Normal", "RCA": "Normal"},
            "features": {col: float(X_df.iloc[idx_lad][col]) for col in feature_names}
        },
        {
            "id": "preset_rca_lcx",
            "name": "RCA + LCX Disease (Inferolateral Multivessel)",
            "subtitle": "Circumflex and right coronary dual territory stenosis",
            "description": "60yo Diabetic Male with elevated blood pressure (130 mmHg), RWMA score 2, inferolateral dual-vessel disease (RCA + LCX), and patent LAD.",
            "patient_index": idx_rca_lcx,
            "ground_truth": {"Cath": "CAD", "LAD": "Normal", "LCX": "Stenotic", "RCA": "Stenotic"},
            "features": {col: float(X_df.iloc[idx_rca_lcx][col]) for col in feature_names}
        },
        {
            "id": "preset_healthy_normal",
            "name": "Healthy Normal Baseline (Low Risk)",
            "subtitle": "Non-obstructive coronary angiogram & normal hemodynamic profile",
            "description": "68yo Female non-smoker with normotensive BP (115 mmHg), preserved ejection fraction (60%), no RWMA, and clear coronary angiogram.",
            "patient_index": idx_normal,
            "ground_truth": {"Cath": "Normal", "LAD": "Normal", "LCX": "Normal", "RCA": "Normal"},
            "features": {col: float(X_df.iloc[idx_normal][col]) for col in feature_names}
        }
    ]

    print("Extracted 4 Real Patient Presets:")
    for p in presets:
        print(f"  - [{p['id']}] {p['name']} (Patient #{p['patient_index']}): {p['ground_truth']}")

    # Save models bundle
    bundle = {
        "models": production_models,
        "explainers": production_explainers,
        "feature_names": feature_names,
        "feature_metadata": feature_metadata,
        "presets": presets,
        "monotone_constraints": constraint_dict,
        "dropped_columns": dropped_cols,
        "zero_target_leakage_verified": True
    }
    with open(MODELS_BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[Artifact Saved] Models Bundle: {MODELS_BUNDLE_PATH} ({os.path.getsize(MODELS_BUNDLE_PATH)} bytes)")

    # Compute domain counts from feature_metadata
    domain_counts = {}
    for f_meta in feature_metadata.values():
        dom = f_meta["domain"]
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    # Save comprehensive evaluation report
    report = {
        "project": "CardioTriage 3D",
        "competition": "Multimodal AI Hackathon 2026 - Track A",
        "dataset": "UCI Extension of Z-Alizadeh Sani (#411)",
        "sample_size": len(X_df),
        "feature_count": len(feature_names),
        "zero_target_leakage_guard": {
            "status": "PASS",
            "assertion_checked": "assert not any(col.lower() in target_names_lower for col in X_df.columns)",
            "excluded_targets": TARGET_COLS,
            "dropped_zero_variance_columns": dropped_cols
        },
        "monotone_constraints_applied": constraint_dict,
        "clinical_domains": domain_counts,
        "cross_validation": {
            "method": "5-Fold Stratified K-Fold (shuffle=True, random_state=42)",
            "evaluation_thresholds": [0.50, 0.35],
            "metrics": cv_results
        },
        "top_feature_importances": top_importances,
        "presets_catalog": [
            {"id": p["id"], "name": p["name"], "patient_index": p["patient_index"], "ground_truth": p["ground_truth"]}
            for p in presets
        ]
    }
    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Artifact Saved] Evaluation Report: {EVAL_REPORT_PATH} ({os.path.getsize(EVAL_REPORT_PATH)} bytes)")
    print("\nStage 1 & Stage 2 ML Pipeline Execution Successfully Completed!\n")
    return report


if __name__ == "__main__":
    run_pipeline()
