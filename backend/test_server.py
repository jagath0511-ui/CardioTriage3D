"""
Unit and integration verification for CardioTriage 3D FastAPI Endpoints.
Verifies GET /api/init, POST /api/predict, Anatomical Union P(Any Vessel),
SHAP explanations with modifiable flags, counterfactual delta risk, and Rule 1d Target Leakage guard.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.server import app, TARGET_COLS

client = TestClient(app)

def test_api():
    print("="*80)
    print("RUNNING CARDIOTRIAGE 3D BACKEND INTEGRATION TEST SUITE")
    print("="*80)

    # 1. Test GET /api/init
    print("\n[Test 1] Testing GET /api/init...")
    resp = client.get("/api/init")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    init_data = resp.json()
    assert init_data["status"] == "success"
    assert len(init_data["features"]) == 54, f"Expected 54 features, got {len(init_data['features'])}"
    assert len(init_data["presets"]) == 4, f"Expected 4 presets, got {len(init_data['presets'])}"
    assert len(init_data["domains"]) == 5, f"Expected 5 domains, got {len(init_data['domains'])}"
    assert "evaluation_report" in init_data
    cv_metrics = init_data["evaluation_report"]["cross_validation"]["metrics"]
    assert "Cath" in cv_metrics and "LAD" in cv_metrics and "LCX" in cv_metrics and "RCA" in cv_metrics
    print(f"  -> Features: {len(init_data['features'])} across 5 domains {init_data['domains']}")
    print(f"  -> Presets: {[p['name'] for p in init_data['presets']]}")
    print(f"  -> Cath 5-Fold OOF ROC-AUC: {cv_metrics['Cath']['roc_auc']:.4f}")
    print(f"  -> LAD  5-Fold OOF ROC-AUC: {cv_metrics['LAD']['roc_auc']:.4f}")

    # 2. Test POST /api/predict with Preset 1 (Triple-Vessel)
    print("\n[Test 2] Testing POST /api/predict (Triple-Vessel Preset)...")
    preset_1 = init_data["presets"][0]
    payload = {
        "features": preset_1["features"],
        "triage_threshold": 0.50
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    pred_data = resp.json()
    assert pred_data["status"] == "success"

    for target in TARGET_COLS:
        assert target in pred_data["predictions"], f"Missing {target} in predictions"
        assert target in pred_data["shap_attributions"], f"Missing {target} in shap"
        prob = pred_data["predictions"][target]["probability"]
        color = pred_data["predictions"][target]["color"]
        top_driver = pred_data["shap_attributions"][target]["top_risk_increasing"][0]
        print(f"  -> {target:5s} Prob: {prob*100:5.1f}% | Color: {color} | Top Driver: {top_driver['label']} ({top_driver['display_value']}) SHAP={top_driver['shap_value']:+.4f} (Modifiable: {top_driver['modifiable']})")

    # 3. Test Anatomical Union Probability
    print("\n[Test 3] Testing Anatomical Union Probability P(Any Vessel)...")
    union = pred_data["anatomical_union"]
    assert "probability" in union
    assert "formula" in union
    p_lad = pred_data["predictions"]["LAD"]["probability"]
    p_lcx = pred_data["predictions"]["LCX"]["probability"]
    p_rca = pred_data["predictions"]["RCA"]["probability"]
    expected_union = round(1.0 - (1.0 - p_lad) * (1.0 - p_lcx) * (1.0 - p_rca), 4)
    assert abs(union["probability"] - expected_union) < 1e-4, f"Mismatch in union: {union['probability']} vs {expected_union}"
    print(f"  -> P(Any Vessel): {union['probability']*100:.1f}% (Formula: {union['formula']})")
    print(f"  -> Coherence with Cath ({pred_data['predictions']['Cath']['probability']*100:.1f}%): {union['coherence_assessment']}")

    # 4. Test Counterfactual Risk Delta Analysis
    print("\n[Test 4] Testing Counterfactual Risk Delta Comparison...")
    # Modify BP, LDL, and Current Smoker in baseline to simulate lifestyle / pharmacological intervention
    baseline_features = dict(preset_1["features"])
    baseline_features["BP"] = 115.0 # reduced from 130
    baseline_features["LDL"] = 70.0 # reduced
    baseline_features["Current Smoker"] = 0 # quit smoking

    cf_payload = {
        "features": preset_1["features"],
        "baseline_features": baseline_features,
        "triage_threshold": 0.35
    }
    resp_cf = client.post("/api/predict", json=cf_payload)
    assert resp_cf.status_code == 200
    cf_data = resp_cf.json()
    assert cf_data["counterfactual_comparison"] is not None
    cf_comp = cf_data["counterfactual_comparison"]
    print(f"  -> Altered modifiable features: {len(cf_comp['altered_modifiable_features'])}")
    for mod in cf_comp["altered_modifiable_features"]:
        print(f"     - {mod['label']}: Baseline {mod['baseline_value']} -> Current {mod['current_value']} {mod['unit']}")
    for target in TARGET_COLS + ["Any_Vessel_Union"]:
        delta_info = cf_comp["risk_deltas"][target]
        print(f"     - {target:16s}: Base {delta_info['baseline_probability']*100:4.1f}% -> Curr {delta_info['current_probability']*100:4.1f}% (Delta: {delta_info['risk_delta_pct']:+5.2f}%)")

    # 5. Test STRICT TARGET LEAKAGE GUARD (Requirement 1d)
    print("\n[Test 5] Testing STRICT TARGET LEAKAGE GUARD (Requirement 1d)...")
    for bad_col in ["Cath", "cath", "LAD", "lcx", "RCA"]:
        leaky_payload = {"features": {**preset_1["features"], bad_col: 1}}
        resp_leaky = client.post("/api/predict", json=leaky_payload)
        assert resp_leaky.status_code == 400, f"Expected 400 for '{bad_col}', got {resp_leaky.status_code}"
    print("  -> Successfully verified: all target columns (case-insensitive) are strictly rejected with 400!")

    # 6. Test POST /api/parse_report (Smart Clinical Report Auto-Filler)
    print("\n[Test 6] Testing POST /api/parse_report (Smart Clinical Report Auto-Filler)...")
    sample_er_note = """
    EMERGENCY DEPARTMENT ADMISSION:
    67-year-old male with acute typical chest pain.
    BP: 162/96 mmHg, Pulse: 96 bpm. Active smoker. Positive family history of premature CAD.
    History of HTN and DM2. No prior CVA or CRF.
    Labs: FBS 168 mg/dL, TG 245 mg/dL, LDL 182 mg/dL, HDL 31 mg/dL, Cr 1.2 mg/dL, BUN 22 mg/dL.
    Echo: LVEF 34%, RWMA: 4 segments. ECG: acute ST elevation, T inversion, pathologic Q waves.
    """
    resp_parse = client.post("/api/parse_report", json={"text": sample_er_note})
    assert resp_parse.status_code == 200, f"Expected 200, got {resp_parse.status_code}: {resp_parse.text}"
    parse_data = resp_parse.json()
    assert parse_data["status"] == "success"
    ext_features = parse_data["extracted_features"]
    assert ext_features.get("Age") == 67
    assert ext_features.get("Sex") == "Male"
    assert ext_features.get("BP") == 162
    assert ext_features.get("PR") == 96
    assert ext_features.get("Current Smoker") == 1
    assert ext_features.get("DM") == 1
    assert ext_features.get("HTN") == 1
    assert ext_features.get("FH") == 1
    assert ext_features.get("Typical Chest Pain") == 1
    assert ext_features.get("FBS") == 168
    assert ext_features.get("LDL") == 182
    assert ext_features.get("TG") == 245
    assert ext_features.get("HDL") == 31
    assert ext_features.get("EF-TTE") == 34
    assert ext_features.get("Region RWMA") == 4
    assert ext_features.get("St Elevation") == 1
    assert ext_features.get("Tinversion") == 1
    assert ext_features.get("Q Wave") == 1
    print(f"  -> Successfully parsed {parse_data['extracted_count']} clinical variables from EHR text:")
    for d in parse_data["details"][:8]:
        print(f"     - {d['label']}: {d['value']}")
    print("     ... and more!")

    print("\n" + "="*80)
    print("ALL INTEGRATION TESTS PASSED WITH ZERO LEAKAGE & FULL CONCORDANCE!")
    print("="*80)

if __name__ == "__main__":
    test_api()
