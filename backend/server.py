"""
CardioTriage 3D - FastAPI Clinical Cockpit Backend Server
Multimodal AI Hackathon 2026 - Track A: Cardiovascular Risk Visualization & Prediction

Key Capabilities:
1. /api/init (GET): Feature metadata (ranges, domains, modifiables), 4 authentic presets, 5-Fold CV evaluation report.
2. /api/predict (POST): Leakage-proof inference, anatomical union probability P(Any Vessel),
   exact per-target Tree-SHAP attributions with modifiable flags, counterfactual delta risk comparison,
   and dynamic triage thresholds (0.50 vs 0.35).
3. Static files mounted at / to serve the frontend on http://localhost:8000.
"""

import os
import time
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODELS_BUNDLE_PATH = os.path.join(BACKEND_DIR, "models_bundle.pkl")
EVAL_REPORT_PATH = os.path.join(BACKEND_DIR, "evaluation_report.json")

TARGET_COLS = ["Cath", "LAD", "LCX", "RCA"]
TARGET_COLS_LOWER = {t.lower() for t in TARGET_COLS}

# Initialize FastAPI app
app = FastAPI(
    title="CardioTriage 3D API",
    description="Multimodal Cardiovascular Risk Visualization & Prediction Engine (UCI Dataset #411)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.endswith((".js", ".css", ".html")) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Startup validation & model loading
if not os.path.exists(MODELS_BUNDLE_PATH):
    raise FileNotFoundError(
        f"Models bundle not found at {MODELS_BUNDLE_PATH}. Run 'python backend/inspect_and_train.py' first!"
    )

with open(MODELS_BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)

models = bundle["models"]
explainers = bundle["explainers"]
feature_names = bundle["feature_names"]
feature_metadata = bundle["feature_metadata"]
presets = bundle["presets"]
monotone_constraints = bundle.get("monotone_constraints", {})

eval_report = {}
if os.path.exists(EVAL_REPORT_PATH):
    with open(EVAL_REPORT_PATH, "r") as f:
        eval_report = json.load(f)


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Patient clinical features matching the 54 input variables."
    )
    baseline_features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional baseline features for counterfactual delta risk analysis."
    )
    triage_threshold: Optional[float] = Field(
        default=0.50,
        ge=0.01,
        le=0.99,
        description="Decision threshold for positive risk triage classification (default 0.50, clinical triage 0.35)."
    )


def classify_risk_tier(prob: float, threshold: float = 0.50) -> Dict[str, Any]:
    """Classifies risk probability into clinical risk tiers and triage alert status."""
    pct = round(prob * 100, 1)
    is_triaged_positive = bool(prob >= threshold)

    if pct < 25.0:
        return {
            "tier": "Low Risk",
            "tier_code": "low",
            "color": "#10b981", # Emerald
            "is_triaged_positive": is_triaged_positive,
            "stenosis_label": "Patent / Non-Obstructive (<25%)",
            "clinical_recommendation": "Lifestyle risk-reduction & primary cardiovascular prevention."
        }
    elif pct < 50.0:
        return {
            "tier": "Moderate Risk",
            "tier_code": "moderate",
            "color": "#f59e0b", # Amber
            "is_triaged_positive": is_triaged_positive,
            "stenosis_label": "Borderline / Intermediate Stenosis (25-49%)",
            "clinical_recommendation": "Non-invasive functional stress imaging (CTCA / Stress Echo) recommended."
        }
    else:
        return {
            "tier": "High Risk",
            "tier_code": "high",
            "color": "#ef4444", # Crimson
            "is_triaged_positive": is_triaged_positive,
            "stenosis_label": "Significant Obstructive Stenosis (>=50%)",
            "clinical_recommendation": "Urgent invasive coronary angiography referral recommended."
        }


def build_feature_dataframe(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Sanitizes, enforces leakage prevention, and aligns incoming features with training matrix schema.
    Missing features are imputed with the population median from UCI Dataset #411.
    """
    # STRICT RUNTIME LEAKAGE GUARD: Reject any payload containing targets
    detected_leaks = [k for k in input_dict.keys() if k.lower() in TARGET_COLS_LOWER]
    if detected_leaks:
        raise HTTPException(
            status_code=400,
            detail=f"STRICT TARGET LEAKAGE GUARD VIOLATION: Target columns {detected_leaks} detected in feature input!"
        )

    row = {}
    for col in feature_names:
        meta = feature_metadata[col]
        val = input_dict.get(col)
        if val is not None and str(val).strip() != "":
            try:
                row[col] = float(val)
            except (ValueError, TypeError):
                row[col] = meta["median"]
        else:
            row[col] = meta["median"]

    df = pd.DataFrame([row], columns=feature_names)
    assert not any(col.lower() in TARGET_COLS_LOWER for col in df.columns), "Target leakage in aligned matrix!"
    return df


@app.get("/api/init")
def get_initialization_data():
    """
    Returns feature metadata (ranges, domains, modifiable flags), the 4 real patient presets,
    and the complete 5-Fold CV evaluation_report.json.
    """
    domains = [
        "Demographics & History",
        "Vitals & Symptoms",
        "ECG Findings",
        "Lab Biomarkers",
        "Echo & Wall Motion"
    ]

    features_catalog_list = [feature_metadata[col] for col in feature_names]

    return {
        "status": "success",
        "project": "CardioTriage 3D",
        "version": "2.0.0",
        "dataset_source": "UCI Dataset #411 (Extension of Z-Alizadeh Sani, N=303)",
        "zero_target_leakage": {
            "status": "ENFORCED",
            "excluded_targets": TARGET_COLS,
            "rule": "Mandatory Requirement 1d: Features matrix strictly disjoint from targets"
        },
        "domains": domains,
        "feature_names": feature_names,
        "features": features_catalog_list,
        "feature_metadata": feature_metadata,
        "presets": presets,
        "evaluation_report": eval_report,
        "monotone_constraints": monotone_constraints
    }


@app.post("/api/predict")
def predict_cardiac_risk(payload: PredictRequest):
    """
    Inference Endpoint:
    - Enforces Strict Zero Target Leakage on features and baseline_features
    - Computes calibrated probabilities for Cath, LAD, LCX, RCA
    - Computes Anatomical Union Probability P(Any Vessel) = 1 - (1 - P_LAD)*(1 - P_LCX)*(1 - P_RCA)
    - Computes exact per-target Tree-SHAP attributions with modifiable flags
    - Computes counterfactual risk delta if baseline_features is provided
    """
    start_time = time.time()
    t_thresh = payload.triage_threshold

    # 1. Process primary features
    X_sample = build_feature_dataframe(payload.features)

    # 2. Compute predictions and SHAP for each target
    predictions = {}
    shap_explanations = {}

    for target in TARGET_COLS:
        model = models[target]
        explainer = explainers[target]

        # Probability
        prob = float(model.predict_proba(X_sample)[0, 1])
        tier_info = classify_risk_tier(prob, threshold=t_thresh)

        predictions[target] = {
            "target": target,
            "probability": round(prob, 4),
            "percentage": round(prob * 100, 1),
            "tier": tier_info["tier"],
            "tier_code": tier_info["tier_code"],
            "color": tier_info["color"],
            "is_triaged_positive": tier_info["is_triaged_positive"],
            "stenosis_label": tier_info["stenosis_label"],
            "clinical_recommendation": tier_info["clinical_recommendation"]
        }

        # Tree-SHAP local explanations
        shap_vals = explainer(X_sample)
        sample_shap = shap_vals.values[0]
        base_val = float(shap_vals.base_values[0]) if hasattr(shap_vals.base_values, "__len__") else float(shap_vals.base_values)

        attributions = []
        for i, col in enumerate(feature_names):
            s_val = float(sample_shap[i])
            meta = feature_metadata[col]
            actual_val = float(X_sample.iloc[0][col])

            # Clinical display formatting
            if meta["type"] == "binary":
                disp_val = "Yes" if actual_val >= 0.5 else "No"
            elif "options" in meta:
                match_opt = next((opt["label"] for opt in meta["options"] if opt["value"] == int(actual_val)), str(actual_val))
                disp_val = match_opt
            else:
                disp_val = f"{actual_val:g} {meta['unit']}".strip()

            attributions.append({
                "feature": col,
                "label": meta["label"],
                "domain": meta["domain"],
                "modifiable": meta.get("modifiable", False),
                "actual_value": actual_val,
                "display_value": disp_val,
                "shap_value": round(s_val, 4),
                "impact": "increases_risk" if s_val > 0 else "reduces_risk"
            })

        # Separate into risk-increasing and risk-reducing (protective)
        increasing = sorted([a for a in attributions if a["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)
        reducing = sorted([a for a in attributions if a["shap_value"] < 0], key=lambda x: x["shap_value"])
        all_sorted = sorted(attributions, key=lambda x: abs(x["shap_value"]), reverse=True)

        shap_explanations[target] = {
            "base_value": round(base_val, 4),
            "top_risk_increasing": increasing[:10],
            "top_risk_reducing": reducing[:10],
            "all_attributions": all_sorted[:15]
        }

    # 3. Anatomical Union Probability Calculation
    # P(Any Vessel) = 1 - (1 - P_LAD) * (1 - P_LCX) * (1 - P_RCA)
    p_lad = predictions["LAD"]["probability"]
    p_lcx = predictions["LCX"]["probability"]
    p_rca = predictions["RCA"]["probability"]
    p_cath = predictions["Cath"]["probability"]

    p_union = 1.0 - (1.0 - p_lad) * (1.0 - p_lcx) * (1.0 - p_rca)
    p_union = float(round(p_union, 4))
    union_tier = classify_risk_tier(p_union, threshold=t_thresh)

    # Coherence check between P(Any Vessel) and Cath
    coherence_delta = round(abs(p_cath - p_union), 4)
    if p_cath >= 0.50 and p_union >= 0.50:
        coherence_status = "High Concordance: Multivessel anatomical probability corroborates overall CAD catheterization risk."
        is_coherent = True
    elif p_cath < 0.35 and p_union < 0.35:
        coherence_status = "Low Risk Concordance: Overall CAD and individual branch vessels all indicate patent coronary circulation."
        is_coherent = True
    elif p_cath >= 0.50 and p_union < 0.40:
        coherence_status = "Clinical Notice: High overall CAD with lower epicardial vessel burden suggests distal or microvascular disease."
        is_coherent = True
    elif p_cath < 0.40 and p_union >= 0.50:
        coherence_status = "Triage Alert: High anatomical vessel probability with lower overall CAD suggests focal critical lesion."
        is_coherent = False
    else:
        coherence_status = f"Moderate Concordance: Coherence delta |P_Cath - P_Union| = {coherence_delta:.2f}."
        is_coherent = True

    anatomical_union = {
        "probability": p_union,
        "percentage": round(p_union * 100, 1),
        "formula": "1 - (1 - P_LAD) * (1 - P_LCX) * (1 - P_RCA)",
        "tier": union_tier["tier"],
        "tier_code": union_tier["tier_code"],
        "color": union_tier["color"],
        "is_triaged_positive": union_tier["is_triaged_positive"],
        "cath_probability": p_cath,
        "coherence_delta": coherence_delta,
        "is_coherent": is_coherent,
        "coherence_assessment": coherence_status
    }

    # 4. Counterfactual Baseline Comparison (if baseline_features provided)
    counterfactual = None
    if payload.baseline_features:
        X_baseline = build_feature_dataframe(payload.baseline_features)
        base_preds = {}
        risk_deltas = {}

        for target in TARGET_COLS:
            m = models[target]
            base_prob = float(m.predict_proba(X_baseline)[0, 1])
            curr_prob = predictions[target]["probability"]
            delta = curr_prob - base_prob
            base_preds[target] = round(base_prob, 4)
            risk_deltas[target] = {
                "current_probability": round(curr_prob, 4),
                "baseline_probability": round(base_prob, 4),
                "risk_delta": round(delta, 4),
                "risk_delta_pct": round(delta * 100, 2),
                "direction": "increased_risk" if delta > 0.001 else ("reduced_risk" if delta < -0.001 else "unchanged")
            }

        base_union = 1.0 - (1.0 - base_preds["LAD"]) * (1.0 - base_preds["LCX"]) * (1.0 - base_preds["RCA"])
        delta_union = p_union - base_union
        risk_deltas["Any_Vessel_Union"] = {
            "current_probability": p_union,
            "baseline_probability": round(base_union, 4),
            "risk_delta": round(delta_union, 4),
            "risk_delta_pct": round(delta_union * 100, 2),
            "direction": "increased_risk" if delta_union > 0.001 else ("reduced_risk" if delta_union < -0.001 else "unchanged")
        }

        # Identify modifiable features that were altered
        altered_modifiables = []
        for col in feature_names:
            if feature_metadata[col].get("modifiable", False):
                curr_v = float(X_sample.iloc[0][col])
                base_v = float(X_baseline.iloc[0][col])
                if abs(curr_v - base_v) > 1e-4:
                    altered_modifiables.append({
                        "feature": col,
                        "label": feature_metadata[col]["label"],
                        "baseline_value": base_v,
                        "current_value": curr_v,
                        "delta_value": round(curr_v - base_v, 2),
                        "unit": feature_metadata[col]["unit"]
                    })

        counterfactual = {
            "baseline_predictions": base_preds,
            "risk_deltas": risk_deltas,
            "altered_modifiable_features": altered_modifiables
        }

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "status": "success",
        "inference_latency_ms": elapsed_ms,
        "triage_threshold": t_thresh,
        "predictions": predictions,
        "anatomical_union": anatomical_union,
        "shap_attributions": shap_explanations,
        "counterfactual_comparison": counterfactual
    }


class ParseReportRequest(BaseModel):
    text: str = Field(..., description="Raw EHR clinical note or laboratory report text.")


def extract_biomarkers_from_text(raw_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Fast regex and keyword extraction engine for clinical summaries and lab reports."""
    import re
    text = raw_text.strip()
    extracted: Dict[str, Any] = {}
    details: List[Dict[str, Any]] = []

    def is_negated(pattern: str, window: int = 40) -> bool:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if not matches:
            return False
        for m in matches:
            start = max(0, m.start() - window)
            prefix = text[start:m.start()].lower()
            if any(neg in prefix for neg in ["no ", "not ", "denies ", "negative ", "without ", "non-", "never "]):
                return True
        return False

    # 1. Age
    m_age = re.search(r'(?:age\s*[:=]?\s*(\d{2})|(\d{2})[\s-]*(?:year[\s-]*old|yo|y\.o\.|years))', text, re.I)
    if m_age:
        val = int(m_age.group(1) or m_age.group(2))
        if 20 <= val <= 100:
            extracted["Age"] = val
            details.append({"feature": "Age", "value": val, "label": "Age"})

    # 2. Sex
    if re.search(r'\b(?:female|woman|lady)\b', text, re.I) and not re.search(r'\bnon-female\b', text, re.I):
        extracted["Sex"] = "Female"
        details.append({"feature": "Sex", "value": "Female", "label": "Biological Sex"})
    elif re.search(r'\b(?:male|man|gentleman)\b', text, re.I) and not re.search(r'\bfemale\b', text, re.I):
        extracted["Sex"] = "Male"
        details.append({"feature": "Sex", "value": "Male", "label": "Biological Sex"})

    # 3. Blood Pressure (Systolic BP)
    m_bp = re.search(r'(?:BP|blood\s+pressure|systolic(?:\s+BP)?)\s*[:=]?\s*(\d{2,3})(?:/\d{2,3})?', text, re.I)
    if m_bp:
        val = int(m_bp.group(1))
        if 70 <= val <= 240:
            extracted["BP"] = val
            details.append({"feature": "BP", "value": val, "label": "Blood Pressure"})

    # 4. Pulse Rate (PR)
    m_pr = re.search(r'(?:PR|pulse(?:\s+rate)?|heart\s+rate|HR)\s*[:=]?\s*(\d{2,3})\s*(?:bpm)?', text, re.I)
    if not m_pr:
        m_pr = re.search(r'(?:sinus\s+(?:rhythm|tachycardia|bradycardia)\s+(?:at\s+)?(\d{2,3}))', text, re.I)
    if m_pr:
        val = int(m_pr.group(1))
        if 40 <= val <= 200:
            extracted["PR"] = val
            details.append({"feature": "PR", "value": val, "label": "Pulse Rate"})

    # 5. Weight & Height & BMI
    m_wt = re.search(r'(?:weight|wt)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|kilos)?', text, re.I)
    m_ht = re.search(r'(?:height|ht)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:cm)?', text, re.I)
    if m_wt:
        val = float(m_wt.group(1))
        if 40 <= val <= 180:
            extracted["Weight"] = val
            details.append({"feature": "Weight", "value": val, "label": "Weight"})
    if m_ht:
        val = float(m_ht.group(1))
        if 130 <= val <= 215:
            extracted["Height"] = val
            details.append({"feature": "Height", "value": val, "label": "Height"})

    m_bmi = re.search(r'(?:BMI|body\s+mass\s+index)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)', text, re.I)
    if m_bmi:
        val = float(m_bmi.group(1))
        if 14 <= val <= 60:
            extracted["BMI"] = val
            extracted["Obesity"] = 1 if val >= 30 else 0
            details.append({"feature": "BMI", "value": val, "label": "Body Mass Index"})
    elif "Weight" in extracted and "Height" in extracted:
        calc_bmi = round(extracted["Weight"] / ((extracted["Height"] / 100.0) ** 2), 1)
        extracted["BMI"] = calc_bmi
        extracted["Obesity"] = 1 if calc_bmi >= 30 else 0
        details.append({"feature": "BMI", "value": calc_bmi, "label": "Body Mass Index (Calculated)"})

    # 6. Current Smoker & EX-Smoker
    if re.search(r'\b(?:non-smoker|never\s+smoked|denies\s+smoking|no\s+smoking|smoker\s*:\s*(?:no|none|negative|0))\b', text, re.I):
        extracted["Current Smoker"] = 0
        extracted["EX-Smoker"] = 0
        details.append({"feature": "Current Smoker", "value": 0, "label": "Current Smoker (No)"})
    elif re.search(r'\b(?:ex-smoker|former\s+smoker|quit\s+smoking|past\s+smoker)\b', text, re.I):
        extracted["Current Smoker"] = 0
        extracted["EX-Smoker"] = 1
        details.append({"feature": "Current Smoker", "value": 0, "label": "Current Smoker (No)"})
        details.append({"feature": "EX-Smoker", "value": 1, "label": "Former Smoker (Yes)"})
    elif re.search(r'\b(?:current\s+smoker|smokes?|tobacco\s+smoker|active\s+smoker|pack-years?|1\s+pack/day|smoker\s*:\s*(?:yes|positive|1))\b', text, re.I):
        extracted["Current Smoker"] = 1
        details.append({"feature": "Current Smoker", "value": 1, "label": "Current Smoker (Yes)"})

    # 7. Diabetes Mellitus (DM)
    if is_negated(r'\b(?:diabetic|diabetes|T2DM|DM2|DM)\b') or re.search(r'\b(?:non-diabetic|no\s+diabetes)\b', text, re.I):
        extracted["DM"] = 0
        details.append({"feature": "DM", "value": 0, "label": "Diabetes Mellitus (No)"})
    elif re.search(r'\b(?:diabetic|diabetes|T2DM|DM2|DM|type\s+2\s+diabetes)\b', text, re.I):
        extracted["DM"] = 1
        details.append({"feature": "DM", "value": 1, "label": "Diabetes Mellitus (Yes)"})

    # 8. Hypertension (HTN)
    if is_negated(r'\b(?:hypertension|HTN|essential\s+hypertension|hypertensive)\b') or re.search(r'\b(?:normotensive|no\s+hypertension|no\s+HTN)\b', text, re.I):
        extracted["HTN"] = 0
        details.append({"feature": "HTN", "value": 0, "label": "Hypertension (No)"})
    elif re.search(r'\b(?:hypertension|HTN|essential\s+hypertension|hypertensive)\b', text, re.I):
        extracted["HTN"] = 1
        details.append({"feature": "HTN", "value": 1, "label": "Hypertension (Yes)"})

    # 9. Family History (FH)
    if is_negated(r'\b(?:family\s+history|FH)\b') or re.search(r'\bno\s+family\s+history\b', text, re.I):
        extracted["FH"] = 0
        details.append({"feature": "FH", "value": 0, "label": "Family History (No)"})
    elif re.search(r'\b(?:family\s+history(?:\s+of\s+CAD)?|FH(?:\s*:\s*(?:pos|yes|\+|1))|premature\s+CAD\s+in\s+family)\b', text, re.I):
        extracted["FH"] = 1
        details.append({"feature": "FH", "value": 1, "label": "Family History (Yes)"})

    # 10. Symptoms: Chest Pain, Dyspnea, Edema, Rales
    if re.search(r'\b(?:typical\s+(?:angina|chest\s+pain|substernal\s+pain|chest\s+pressure))\b', text, re.I):
        extracted["Typical Chest Pain"] = 1
        extracted["Atypical"] = 0
        extracted["Nonanginal"] = 0
        details.append({"feature": "Typical Chest Pain", "value": 1, "label": "Typical Chest Pain (Yes)"})
    elif re.search(r'\b(?:atypical\s+(?:angina|chest\s+pain|chest\s+heaviness))\b', text, re.I):
        extracted["Atypical"] = 1
        extracted["Typical Chest Pain"] = 0
        details.append({"feature": "Atypical", "value": 1, "label": "Atypical Chest Pain (Yes)"})
    elif re.search(r'\b(?:nonanginal|non-anginal|non-cardiac\s+chest\s+pain)\b', text, re.I):
        extracted["Nonanginal"] = 1
        extracted["Typical Chest Pain"] = 0
        details.append({"feature": "Nonanginal", "value": 1, "label": "Nonanginal Pain (Yes)"})
    elif re.search(r'\b(?:asymptomatic|no\s+chest\s+pain)\b', text, re.I):
        extracted["Typical Chest Pain"] = 0
        extracted["Atypical"] = 0
        extracted["Nonanginal"] = 0
        details.append({"feature": "Typical Chest Pain", "value": 0, "label": "Chest Pain (None)"})

    if is_negated(r'\b(?:dyspnea|shortness\s+of\s+breath|SOB)\b') or re.search(r'\bno\s+dyspnea\b', text, re.I):
        extracted["Dyspnea"] = 0
        details.append({"feature": "Dyspnea", "value": 0, "label": "Dyspnea (No)"})
    elif re.search(r'\b(?:dyspnea|shortness\s+of\s+breath|SOB)\b', text, re.I):
        extracted["Dyspnea"] = 1
        details.append({"feature": "Dyspnea", "value": 1, "label": "Dyspnea (Yes)"})

    if is_negated(r'\b(?:edema|pedal\s+edema|peripheral\s+edema)\b') or re.search(r'\bno\s+edema\b', text, re.I):
        extracted["Edema"] = 0
        details.append({"feature": "Edema", "value": 0, "label": "Edema (No)"})
    elif re.search(r'\b(?:edema|pedal\s+edema|peripheral\s+edema)\b', text, re.I):
        extracted["Edema"] = 1
        details.append({"feature": "Edema", "value": 1, "label": "Edema (Yes)"})

    if is_negated(r'\b(?:rales|crackles|lung\s+rales)\b') or re.search(r'\bno\s+rales\b', text, re.I):
        extracted["Lung rales"] = 0
        details.append({"feature": "Lung rales", "value": 0, "label": "Lung Rales (No)"})
    elif re.search(r'\b(?:rales|crackles|lung\s+rales)\b', text, re.I):
        extracted["Lung rales"] = 1
        details.append({"feature": "Lung rales", "value": 1, "label": "Lung Rales (Yes)"})

    # 11. Laboratory Biomarkers
    lab_regexes = [
        ("FBS", r'(?:FBS|fasting\s+blood\s+sugar|fasting\s+glucose)\s*[:=]?\s*(\d{2,3})', 50, 450, "Fasting Blood Sugar"),
        ("TG", r'(?:TG|triglycerides?)\s*[:=]?\s*(\d{2,4})', 40, 1000, "Triglycerides"),
        ("LDL", r'(?:LDL(?:-C)?|low\s+density\s+lipoprotein)\s*[:=]?\s*(\d{2,3})', 30, 350, "LDL Cholesterol"),
        ("HDL", r'(?:HDL(?:-C)?|high\s+density\s+lipoprotein)\s*[:=]?\s*(\d{2,3})', 15, 120, "HDL Cholesterol"),
        ("CR", r'(?:serum\s+creatinine|creatinine|cr)\s*[:=]?\s*(\d+(?:\.\d+)?)', 0.4, 12.0, "Serum Creatinine"),
        ("BUN", r'(?:BUN|blood\s+urea\s+nitrogen)\s*[:=]?\s*(\d{1,3})', 4, 150, "BUN"),
        ("K", r'(?:potassium|K\+?)\s*[:=]?\s*(\d+(?:\.\d+)?)', 2.0, 8.0, "Potassium"),
        ("Na", r'(?:sodium|Na\+?)\s*[:=]?\s*(\d{2,3})', 110, 165, "Sodium"),
        ("HB", r'(?:hemoglobin|hb|hgb)\s*[:=]?\s*(\d+(?:\.\d+)?)', 6.0, 22.0, "Hemoglobin"),
        ("WBC", r'(?:WBC|white\s+blood\s+cells?)\s*[:=]?\s*(\d{3,5})', 1500, 30000, "WBC Count"),
        ("PLT", r'(?:platelets?|plt)\s*[:=]?\s*(\d{2,4})', 40, 700, "Platelets"),
        ("ESR", r'(?:ESR|erythrocyte\s+sedimentation\s+rate)\s*[:=]?\s*(\d{1,3})', 1, 140, "ESR")
    ]
    for key, pattern, min_v, max_v, lbl in lab_regexes:
        m = re.search(pattern, text, re.I)
        if m:
            val = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
            if min_v <= val <= max_v:
                extracted[key] = val
                details.append({"feature": key, "value": val, "label": lbl})

    # 12. Echocardiography (EF-TTE & RWMA)
    m_ef = re.search(r'(?:LVEF|EF|ejection\s+fraction|EF-TTE)\s*[:=]?\s*(\d{2})%?', text, re.I)
    if m_ef:
        val = int(m_ef.group(1))
        if 15 <= val <= 85:
            extracted["EF-TTE"] = val
            details.append({"feature": "EF-TTE", "value": val, "label": "Ejection Fraction (EF)"})

    if is_negated(r'\b(?:RWMA|wall\s+motion\s+abnormalit(?:y|ies))\b') or re.search(r'\b(?:normal\s+wall\s+motion|RWMA\s*:\s*0|no\s+RWMA)\b', text, re.I):
        extracted["Region RWMA"] = 0
        details.append({"feature": "Region RWMA", "value": 0, "label": "Regional Wall Motion (None)"})
    else:
        m_rwma = re.search(r'(?:RWMA|wall\s+motion\s+abnormalit(?:y|ies))\s*(?:count|present\s+with|segments?|zones?)?[:=]?\s*([0-4])', text, re.I)
        if m_rwma:
            val = int(m_rwma.group(1))
            extracted["Region RWMA"] = val
            details.append({"feature": "Region RWMA", "value": val, "label": f"Regional Wall Motion ({val} zones)"})
        elif re.search(r'\b(?:RWMA\s+(?:noted|present|positive)|hypokinesia|hypokinetic|akinesia)\b', text, re.I):
            extracted["Region RWMA"] = 3
            details.append({"feature": "Region RWMA", "value": 3, "label": "Regional Wall Motion (3 zones)"})

    # 13. 12-Lead ECG Findings
    if is_negated(r'\b(?:ST[\s-]*(?:segment\s+)?elevation|STE)\b') or re.search(r'\bno\s+ST[\s-]*elevation\b', text, re.I):
        extracted["St Elevation"] = 0
        details.append({"feature": "St Elevation", "value": 0, "label": "ST Elevation (No)"})
    elif re.search(r'\b(?:ST[\s-]*(?:segment\s+)?elevation|STE)\b', text, re.I):
        extracted["St Elevation"] = 1
        details.append({"feature": "St Elevation", "value": 1, "label": "ST Elevation (Yes)"})

    if is_negated(r'\b(?:ST[\s-]*(?:segment\s+)?depression|STD)\b') or re.search(r'\bno\s+ST[\s-]*depression\b', text, re.I):
        extracted["St Depression"] = 0
        details.append({"feature": "St Depression", "value": 0, "label": "ST Depression (No)"})
    elif re.search(r'\b(?:ST[\s-]*(?:segment\s+)?depression|STD)\b', text, re.I):
        extracted["St Depression"] = 1
        details.append({"feature": "St Depression", "value": 1, "label": "ST Depression (Yes)"})

    if is_negated(r'\b(?:T[\s-]*wave\s+inversion|T[\s-]*inversion)\b') or re.search(r'\bno\s+T[\s-]*inversion\b', text, re.I):
        extracted["Tinversion"] = 0
        details.append({"feature": "Tinversion", "value": 0, "label": "T Inversion (No)"})
    elif re.search(r'\b(?:T[\s-]*wave\s+inversion|T[\s-]*inversion|inverted\s+T[\s-]*waves?)\b', text, re.I):
        extracted["Tinversion"] = 1
        details.append({"feature": "Tinversion", "value": 1, "label": "T Inversion (Yes)"})

    if is_negated(r'\b(?:pathologic(?:al)?\s+Q[\s-]*waves?|Q[\s-]*waves?)\b') or re.search(r'\bno\s+Q[\s-]*waves?\b', text, re.I):
        extracted["Q Wave"] = 0
        details.append({"feature": "Q Wave", "value": 0, "label": "Q Wave (No)"})
    elif re.search(r'\b(?:pathologic(?:al)?\s+Q[\s-]*waves?|Q[\s-]*waves?)\b', text, re.I):
        extracted["Q Wave"] = 1
        details.append({"feature": "Q Wave", "value": 1, "label": "Q Wave (Yes)"})

    if is_negated(r'\b(?:left\s+ventricular\s+hypertrophy|LVH)\b') or re.search(r'\bno\s+LVH\b', text, re.I):
        extracted["LVH"] = 0
        details.append({"feature": "LVH", "value": 0, "label": "LV Hypertrophy (No)"})
    elif re.search(r'\b(?:left\s+ventricular\s+hypertrophy|LVH)\b', text, re.I):
        extracted["LVH"] = 1
        details.append({"feature": "LVH", "value": 1, "label": "LV Hypertrophy (Yes)"})

    return {
        "status": "success",
        "extracted_features": extracted,
        "extracted_count": len(extracted),
        "details": details
    }


@app.post("/api/parse_report")
def parse_clinical_report_endpoint(req: ParseReportRequest):
    """Parses raw EHR clinical notes or laboratory reports into 54 structured biomarkers."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Report text cannot be empty.")
    return extract_biomarkers_from_text(req.text, feature_metadata)


@app.get("/health")
def health_check():
    """Health and model verification endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "features_count": len(feature_names),
        "presets_count": len(presets),
        "zero_target_leakage": True
    }


# Mount static frontend directory at /
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\nStarting CardioTriage 3D FastAPI server on http://localhost:8000...")
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
