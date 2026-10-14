/**
 * CardioTriage 3D - Smart Clinical Report Auto-Filler & Natural Language Extractor
 * Multimodal AI Hackathon 2026 - Track A
 * 
 * Provides:
 * 1. 1-Click authentic sample reports (ER Acute STEMI, Routine Annual Screening, Diabetic Dyslipidemia).
 * 2. Instant client-side Regex/NLP extractor with medical negation detection and range clamping.
 */

const SAMPLE_CLINICAL_REPORTS = {
  er: `EMERGENCY DEPARTMENT ADMISSION & TRIAGE RECORD:
Patient: 67-year-old male presenting via EMS with acute severe retrosternal typical chest pain radiating to left jaw and shoulder, onset 2 hours ago.
Vital Signs on Arrival: BP 162/96 mmHg, Pulse Rate 96 bpm, O2 Sat 94% on room air.
Social History: Active heavy cigarette smoker (30 pack-years, currently smokes 1 pack/day). Positive family history of premature CAD (father had fatal MI at age 52).
Medical History: Essential hypertension (HTN), Type 2 diabetes mellitus (DM2), dyslipidemia. No history of prior stroke (CVA) or chronic renal failure (CRF).
Clinical Exam: Anxious, diaphoretic. Mild bilateral lung crackles/rales. No systolic murmur, no pedal edema.
12-Lead ECG Findings:
- Sinus tachycardia at 96 bpm.
- Acute ST-segment elevation (2.5 mm) in leads V1 through V4.
- Reciprocal ST-segment depression in inferior leads.
- Pathologic Q waves and marked T-wave inversion in anterior leads.
Laboratory Biomarkers:
- Fasting Blood Sugar (FBS): 168 mg/dL
- Total Triglycerides (TG): 245 mg/dL
- LDL Cholesterol: 182 mg/dL
- HDL Cholesterol: 31 mg/dL
- Serum Creatinine (Cr): 1.2 mg/dL
- BUN: 22 mg/dL
- Potassium (K): 4.1 mEq/L, Sodium (Na): 138 mEq/L
- Hemoglobin (Hb): 14.8 g/dL, WBC: 11,200 /mcL, ESR: 24 mm/hr
Emergency Bedside Echocardiogram (TTE):
- Left Ventricular Ejection Fraction (LVEF): 34% (severely depressed)
- Regional Wall Motion Abnormality (RWMA) noted with 4 hypokinetic/akinetic anterior-septal and apical wall zones.
Assessment: Acute anterior STEMI, critical multivessel ischemia. Urgent cardiac catheterization activated.`,

  routine: `ANNUAL PREVENTIVE HEALTH & EXECUTIVE CARDIAC SCREENING:
Patient: 54-year-old female presenting for routine periodic health surveillance. Asymptomatic, no chest pain, no dyspnea on exertion.
Physical Examination & Vitals:
- Blood Pressure: 116/74 mmHg (normotensive)
- Pulse Rate: 64 bpm (regular sinus rhythm)
- Weight: 62 kg, Height: 165 cm, BMI: 22.8 kg/m2
Medical & Social History:
- Non-smoker (never smoked tobacco).
- No personal or family history of coronary artery disease.
- No diabetes (non-diabetic), no hypertension, no history of renal or airway disease.
Comprehensive Metabolic & Lipid Panel:
- Fasting Blood Sugar (FBS): 86 mg/dL
- Total Triglycerides (TG): 98 mg/dL
- LDL Cholesterol: 84 mg/dL (optimal)
- HDL Cholesterol: 62 mg/dL (protective)
- Serum Creatinine: 0.8 mg/dL, BUN: 14 mg/dL
- Electrolytes: Potassium 4.4 mEq/L, Sodium 141 mEq/L
- CBC: Hemoglobin 13.6 g/dL, WBC 5,800 /mcL, Platelets 230, ESR 8 mm/hr
Cardiovascular Diagnostic Testing:
- Resting 12-Lead ECG: Normal sinus rhythm. No ST elevation, no ST depression, no T-wave inversion, no pathologic Q waves.
- Screening Transthoracic Echo (TTE): Normal LV systolic function, LVEF 64%. No regional wall motion abnormalities (RWMA: 0). Valves structurally normal.
Impression: Healthy cardiovascular profile. Low 10-year atherosclerotic risk.`,

  diabetic: `ENDOCRINOLOGY & CARDIOLOGY FOLLOW-UP CONSULTATION:
Patient: 61-year-old male with long-standing poorly controlled Type 2 Diabetes Mellitus and combined dyslipidemia.
Complains of intermittent atypical chest heaviness and exertional dyspnea (NYHA Class 2).
Vitals: BP 144/88 mmHg, Heart Rate: 78 bpm. Weight: 86 kg, Height: 172 cm (BMI 29.1 kg/m2, overweight).
History: Former tobacco smoker (quit 3 years ago). Documented hypertension and chronic dyslipidemia.
Recent Laboratory Biomarkers:
- Fasting Blood Sugar (FBS): 215 mg/dL (marked hyperglycemia)
- Total Triglycerides (TG): 310 mg/dL (severe hypertriglyceridemia)
- LDL Cholesterol: 156 mg/dL
- HDL Cholesterol: 36 mg/dL
- Serum Creatinine: 1.3 mg/dL, BUN: 26 mg/dL
- Potassium: 4.6 mEq/L, Sodium: 139 mEq/L
- Hemoglobin: 14.1 g/dL, WBC: 7,900 /mcL, ESR: 19 mm/hr
Cardiac Evaluation:
- 12-Lead ECG: Sinus rhythm. Subtle ST-segment depression (0.5 mm) in inferolateral leads. Flat T waves, no acute ST elevation. Left ventricular hypertrophy (LVH) by voltage criteria.
- Transthoracic Echocardiogram (TTE): LVEF 48% (mildly reduced). Regional wall motion abnormality (RWMA) present with 2 hypokinetic segments along the inferior wall.
Plan: Intensify lipid and glycemic control, evaluate for elective invasive or CT coronary angiography.`
};

/**
 * Intelligent regex parser that extracts clinical biomarkers from raw medical prose or lab tables.
 * @param {string} rawText 
 * @param {object} featureMetadata 
 * @returns {object} { extractedFeatures, extractedCount, details }
 */
function parseClinicalReportClient(rawText, featureMetadata = {}) {
  if (!rawText || typeof rawText !== "string") {
    return { extractedFeatures: {}, extractedCount: 0, details: [] };
  }

  const text = rawText.trim();
  const extracted = {};
  const details = [];

  function isNegated(regex) {
    const matches = Array.from(text.matchAll(new RegExp(regex, "gi")));
    if (matches.length === 0) return false;
    for (const m of matches) {
      const start = Math.max(0, m.index - 45);
      const prefix = text.substring(start, m.index).toLowerCase();
      if (/(?:no\s+|not\s+|denies\s+|negative\s+|without\s+|non-|never\s+|zero\s+)/.test(prefix)) {
        return true;
      }
    }
    return false;
  }

  // 1. Age
  const mAge = text.match(/(?:age\s*[:=]?\s*(\d{2})|(\d{2})[\s-]*(?:year[\s-]*old|yo|y\.o\.|years))/i);
  if (mAge) {
    const val = parseInt(mAge[1] || mAge[2], 10);
    if (val >= 20 && val <= 100) {
      extracted["Age"] = val;
      details.push({ feature: "Age", value: `${val} yrs`, label: "Age", modifiable: false });
    }
  }

  // 2. Sex
  if (/\b(?:female|woman|lady)\b/i.test(text) && !/\bnon-female\b/i.test(text)) {
    extracted["Sex"] = "Female";
    details.push({ feature: "Sex", value: "Female", label: "Biological Sex", modifiable: false });
  } else if (/\b(?:male|man|gentleman)\b/i.test(text) && !/\bfemale\b/i.test(text)) {
    extracted["Sex"] = "Male";
    details.push({ feature: "Sex", value: "Male", label: "Biological Sex", modifiable: false });
  }

  // 3. Blood Pressure (Systolic BP)
  const mBp = text.match(/(?:BP|blood\s+pressure|systolic(?:\s+BP)?)\s*[:=]?\s*(\d{2,3})(?:\/\d{2,3})?/i);
  if (mBp) {
    const val = parseInt(mBp[1], 10);
    if (val >= 70 && val <= 240) {
      extracted["BP"] = val;
      details.push({ feature: "BP", value: `${val} mmHg`, label: "Blood Pressure", modifiable: true });
    }
  }

  // 4. Pulse Rate (PR)
  let mPr = text.match(/(?:PR|pulse(?:\s+rate)?|heart\s+rate|HR)\s*[:=]?\s*(\d{2,3})\s*(?:bpm)?/i);
  if (!mPr) {
    mPr = text.match(/(?:sinus\s+(?:rhythm|tachycardia|bradycardia)\s+(?:at\s+)?(\d{2,3}))/i);
  }
  if (mPr) {
    const val = parseInt(mPr[1], 10);
    if (val >= 40 && val <= 200) {
      extracted["PR"] = val;
      details.push({ feature: "PR", value: `${val} bpm`, label: "Pulse Rate", modifiable: true });
    }
  }

  // 5. Weight & Height & BMI
  const mWt = text.match(/(?:weight|wt)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|kilos)?/i);
  const mHt = text.match(/(?:height|ht)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:cm)?/i);
  if (mWt) {
    const val = parseFloat(mWt[1]);
    if (val >= 40 && val <= 180) {
      extracted["Weight"] = val;
      details.push({ feature: "Weight", value: `${val} kg`, label: "Weight", modifiable: true });
    }
  }
  if (mHt) {
    const val = parseFloat(mHt[1]);
    if (val >= 130 && val <= 215) {
      extracted["Height"] = val;
      details.push({ feature: "Height", value: `${val} cm`, label: "Height", modifiable: false });
    }
  }

  const mBmi = text.match(/(?:BMI|body\s+mass\s+index)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)/i);
  if (mBmi) {
    const val = parseFloat(mBmi[1]);
    if (val >= 14 && val <= 60) {
      extracted["BMI"] = val;
      extracted["Obesity"] = val >= 30 ? 1 : 0;
      details.push({ feature: "BMI", value: `${val} kg/m²`, label: "BMI", modifiable: true });
    }
  } else if (extracted["Weight"] && extracted["Height"]) {
    const calcBmi = Math.round((extracted["Weight"] / Math.pow(extracted["Height"] / 100.0, 2)) * 10) / 10;
    extracted["BMI"] = calcBmi;
    extracted["Obesity"] = calcBmi >= 30 ? 1 : 0;
    details.push({ feature: "BMI", value: `${calcBmi} kg/m² (calc)`, label: "BMI", modifiable: true });
  }

  // 6. Current Smoker & EX-Smoker
  if (/\b(?:non-smoker|never\s+smoked|denies\s+smoking|no\s+smoking|smoker\s*:\s*(?:no|none|negative|0))\b/i.test(text)) {
    extracted["Current Smoker"] = 0;
    extracted["EX-Smoker"] = 0;
    details.push({ feature: "Current Smoker", value: "No", label: "Smoker", modifiable: true });
  } else if (/\b(?:ex-smoker|former\s+smoker|quit\s+smoking|past\s+smoker)\b/i.test(text)) {
    extracted["Current Smoker"] = 0;
    extracted["EX-Smoker"] = 1;
    details.push({ feature: "Current Smoker", value: "No (Ex-Smoker)", label: "Smoker", modifiable: true });
  } else if (/\b(?:current\s+smoker|smokes?|tobacco\s+smoker|active\s+smoker|pack-years?|1\s+pack\/day|smoker\s*:\s*(?:yes|positive|1))\b/i.test(text)) {
    extracted["Current Smoker"] = 1;
    details.push({ feature: "Current Smoker", value: "Yes", label: "Smoker", modifiable: true });
  }

  // 7. Diabetes Mellitus (DM)
  if (isNegated(/\b(?:diabetic|diabetes|T2DM|DM2|DM)\b/) || /\b(?:non-diabetic|no\s+diabetes)\b/i.test(text)) {
    extracted["DM"] = 0;
    details.push({ feature: "DM", value: "No", label: "Diabetes Mellitus", modifiable: false });
  } else if (/\b(?:diabetic|diabetes|T2DM|DM2|DM|type\s+2\s+diabetes)\b/i.test(text)) {
    extracted["DM"] = 1;
    details.push({ feature: "DM", value: "Yes", label: "Diabetes Mellitus", modifiable: false });
  }

  // 8. Hypertension (HTN)
  if (isNegated(/\b(?:hypertension|HTN|essential\s+hypertension|hypertensive)\b/) || /\b(?:normotensive|no\s+hypertension|no\s+HTN)\b/i.test(text)) {
    extracted["HTN"] = 0;
    details.push({ feature: "HTN", value: "No", label: "Hypertension", modifiable: false });
  } else if (/\b(?:hypertension|HTN|essential\s+hypertension|hypertensive)\b/i.test(text)) {
    extracted["HTN"] = 1;
    details.push({ feature: "HTN", value: "Yes", label: "Hypertension", modifiable: false });
  }

  // 9. Family History (FH)
  if (isNegated(/\b(?:family\s+history|FH)\b/) || /\bno\s+family\s+history\b/i.test(text)) {
    extracted["FH"] = 0;
    details.push({ feature: "FH", value: "No", label: "Family History", modifiable: false });
  } else if (/\b(?:family\s+history(?:\s+of\s+CAD)?|FH(?:\s*:\s*(?:pos|yes|\+|1))|premature\s+CAD\s+in\s+family)\b/i.test(text)) {
    extracted["FH"] = 1;
    details.push({ feature: "FH", value: "Yes", label: "Family History", modifiable: false });
  }

  // 10. Chest Pain Symptoms
  if (/\b(?:typical\s+(?:angina|chest\s+pain|substernal\s+pain|chest\s+pressure))\b/i.test(text)) {
    extracted["Typical Chest Pain"] = 1;
    extracted["Atypical"] = 0;
    extracted["Nonanginal"] = 0;
    details.push({ feature: "Typical Chest Pain", value: "Typical", label: "Chest Pain", modifiable: false });
  } else if (/\b(?:atypical\s+(?:angina|chest\s+pain|chest\s+heaviness))\b/i.test(text)) {
    extracted["Atypical"] = 1;
    extracted["Typical Chest Pain"] = 0;
    details.push({ feature: "Atypical", value: "Atypical", label: "Chest Pain", modifiable: false });
  } else if (/\b(?:nonanginal|non-anginal|non-cardiac\s+chest\s+pain)\b/i.test(text)) {
    extracted["Nonanginal"] = 1;
    extracted["Typical Chest Pain"] = 0;
    details.push({ feature: "Nonanginal", value: "Nonanginal", label: "Chest Pain", modifiable: false });
  } else if (/\b(?:asymptomatic|no\s+chest\s+pain)\b/i.test(text)) {
    extracted["Typical Chest Pain"] = 0;
    extracted["Atypical"] = 0;
    extracted["Nonanginal"] = 0;
    details.push({ feature: "Typical Chest Pain", value: "None", label: "Chest Pain", modifiable: false });
  }

  // Dyspnea, Edema, Rales
  if (isNegated(/\b(?:dyspnea|shortness\s+of\s+breath|SOB)\b/) || /\bno\s+dyspnea\b/i.test(text)) {
    extracted["Dyspnea"] = 0;
  } else if (/\b(?:dyspnea|shortness\s+of\s+breath|SOB)\b/i.test(text)) {
    extracted["Dyspnea"] = 1;
    details.push({ feature: "Dyspnea", value: "Yes", label: "Dyspnea", modifiable: false });
  }

  if (isNegated(/\b(?:edema|pedal\s+edema|peripheral\s+edema)\b/) || /\bno\s+edema\b/i.test(text)) {
    extracted["Edema"] = 0;
  } else if (/\b(?:edema|pedal\s+edema|peripheral\s+edema)\b/i.test(text)) {
    extracted["Edema"] = 1;
    details.push({ feature: "Edema", value: "Yes", label: "Peripheral Edema", modifiable: false });
  }

  if (isNegated(/\b(?:rales|crackles|lung\s+rales)\b/) || /\bno\s+rales\b/i.test(text)) {
    extracted["Lung rales"] = 0;
  } else if (/\b(?:rales|crackles|lung\s+rales)\b/i.test(text)) {
    extracted["Lung rales"] = 1;
    details.push({ feature: "Lung rales", value: "Yes", label: "Lung Rales", modifiable: false });
  }

  // 11. Laboratory Biomarkers
  const labPatterns = [
    { key: "FBS", regex: /(?:FBS|fasting\s+blood\s+sugar|fasting\s+glucose)\s*[:=]?\s*(\d{2,3})/i, min: 50, max: 450, unit: "mg/dL", label: "Fasting Blood Sugar", modifiable: true },
    { key: "TG", regex: /(?:TG|triglycerides?)\s*[:=]?\s*(\d{2,4})/i, min: 40, max: 1000, unit: "mg/dL", label: "Triglycerides", modifiable: true },
    { key: "LDL", regex: /(?:LDL(?:-C)?|low\s+density\s+lipoprotein)\s*[:=]?\s*(\d{2,3})/i, min: 30, max: 350, unit: "mg/dL", label: "LDL Cholesterol", modifiable: true },
    { key: "HDL", regex: /(?:HDL(?:-C)?|high\s+density\s+lipoprotein)\s*[:=]?\s*(\d{2,3})/i, min: 15, max: 120, unit: "mg/dL", label: "HDL Cholesterol", modifiable: true },
    { key: "CR", regex: /(?:serum\s+creatinine|creatinine|cr)\s*[:=]?\s*(\d+(?:\.\d+)?)/i, min: 0.4, max: 12.0, unit: "mg/dL", label: "Creatinine", modifiable: false },
    { key: "BUN", regex: /(?:BUN|blood\s+urea\s+nitrogen)\s*[:=]?\s*(\d{1,3})/i, min: 4, max: 150, unit: "mg/dL", label: "BUN", modifiable: false },
    { key: "K", regex: /(?:potassium|K\+?)\s*[:=]?\s*(\d+(?:\.\d+)?)/i, min: 2.0, max: 8.0, unit: "mEq/L", label: "Potassium (K)", modifiable: false },
    { key: "Na", regex: /(?:sodium|Na\+?)\s*[:=]?\s*(\d{2,3})/i, min: 110, max: 165, unit: "mEq/L", label: "Sodium (Na)", modifiable: false },
    { key: "HB", regex: /(?:hemoglobin|hb|hgb)\s*[:=]?\s*(\d+(?:\.\d+)?)/i, min: 6.0, max: 22.0, unit: "g/dL", label: "Hemoglobin", modifiable: false },
    { key: "WBC", regex: /(?:WBC|white\s+blood\s+cells?)\s*[:=]?\s*(\d{3,5})/i, min: 1500, max: 30000, unit: "/mcL", label: "WBC Count", modifiable: false },
    { key: "PLT", regex: /(?:platelets?|plt)\s*[:=]?\s*(\d{2,4})/i, min: 40, max: 700, unit: "10³/mcL", label: "Platelets", modifiable: false },
    { key: "ESR", regex: /(?:ESR|erythrocyte\s+sedimentation\s+rate)\s*[:=]?\s*(\d{1,3})/i, min: 1, max: 140, unit: "mm/hr", label: "ESR", modifiable: false }
  ];

  labPatterns.forEach(lp => {
    const m = text.match(lp.regex);
    if (m) {
      const val = m[1].includes(".") ? parseFloat(m[1]) : parseInt(m[1], 10);
      if (val >= lp.min && val <= lp.max) {
        extracted[lp.key] = val;
        details.push({ feature: lp.key, value: `${val} ${lp.unit}`, label: lp.label, modifiable: lp.modifiable });
      }
    }
  });

  // 12. Echocardiography (EF-TTE & RWMA)
  const mEf = text.match(/(?:LVEF|EF|ejection\s+fraction|EF-TTE)\s*[:=]?\s*(\d{2})%?/i);
  if (mEf) {
    const val = parseInt(mEf[1], 10);
    if (val >= 15 && val <= 85) {
      extracted["EF-TTE"] = val;
      details.push({ feature: "EF-TTE", value: `${val}%`, label: "Ejection Fraction", modifiable: false });
    }
  }

  if (isNegated(/\b(?:RWMA|wall\s+motion\s+abnormalit(?:y|ies))\b/) || /\b(?:normal\s+wall\s+motion|RWMA\s*:\s*0|no\s+RWMA)\b/i.test(text)) {
    extracted["Region RWMA"] = 0;
    details.push({ feature: "Region RWMA", value: "0 zones", label: "Wall Motion (RWMA)", modifiable: false });
  } else {
    const mRwma = text.match(/(?:RWMA|wall\s+motion\s+abnormalit(?:y|ies))\s*(?:count|present\s+with|segments?|zones?)?[:=]?\s*([0-4])/i);
    if (mRwma) {
      const val = parseInt(mRwma[1], 10);
      extracted["Region RWMA"] = val;
      details.push({ feature: "Region RWMA", value: `${val} zones`, label: "Wall Motion (RWMA)", modifiable: false });
    } else if (/\b(?:RWMA\s+(?:noted|present|positive)|hypokinesia|hypokinetic|akinesia)\b/i.test(text)) {
      extracted["Region RWMA"] = 3;
      details.push({ feature: "Region RWMA", value: "3 zones", label: "Wall Motion (RWMA)", modifiable: false });
    }
  }

  // 13. 12-Lead ECG Findings
  if (isNegated(/\b(?:ST[\s-]*(?:segment\s+)?elevation|STE)\b/) || /\bno\s+ST[\s-]*elevation\b/i.test(text)) {
    extracted["St Elevation"] = 0;
    details.push({ feature: "St Elevation", value: "No", label: "ST Elevation", modifiable: false });
  } else if (/\b(?:ST[\s-]*(?:segment\s+)?elevation|STE)\b/i.test(text)) {
    extracted["St Elevation"] = 1;
    details.push({ feature: "St Elevation", value: "Yes", label: "ST Elevation", modifiable: false });
  }

  if (isNegated(/\b(?:ST[\s-]*(?:segment\s+)?depression|STD)\b/) || /\bno\s+ST[\s-]*depression\b/i.test(text)) {
    extracted["St Depression"] = 0;
    details.push({ feature: "St Depression", value: "No", label: "ST Depression", modifiable: false });
  } else if (/\b(?:ST[\s-]*(?:segment\s+)?depression|STD)\b/i.test(text)) {
    extracted["St Depression"] = 1;
    details.push({ feature: "St Depression", value: "Yes", label: "ST Depression", modifiable: false });
  }

  if (isNegated(/\b(?:T[\s-]*wave\s+inversion|T[\s-]*inversion)\b/) || /\bno\s+T[\s-]*inversion\b/i.test(text)) {
    extracted["Tinversion"] = 0;
    details.push({ feature: "Tinversion", value: "No", label: "T Inversion", modifiable: false });
  } else if (/\b(?:T[\s-]*wave\s+inversion|T[\s-]*inversion|inverted\s+T[\s-]*waves?)\b/i.test(text)) {
    extracted["Tinversion"] = 1;
    details.push({ feature: "Tinversion", value: "Yes", label: "T Inversion", modifiable: false });
  }

  if (isNegated(/\b(?:pathologic(?:al)?\s+Q[\s-]*waves?|Q[\s-]*waves?)\b/) || /\bno\s+Q[\s-]*waves?\b/i.test(text)) {
    extracted["Q Wave"] = 0;
  } else if (/\b(?:pathologic(?:al)?\s+Q[\s-]*waves?|Q[\s-]*waves?)\b/i.test(text)) {
    extracted["Q Wave"] = 1;
    details.push({ feature: "Q Wave", value: "Yes", label: "Q Wave", modifiable: false });
  }

  if (isNegated(/\b(?:left\s+ventricular\s+hypertrophy|LVH)\b/) || /\bno\s+LVH\b/i.test(text)) {
    extracted["LVH"] = 0;
  } else if (/\b(?:left\s+ventricular\s+hypertrophy|LVH)\b/i.test(text)) {
    extracted["LVH"] = 1;
    details.push({ feature: "LVH", value: "Yes", label: "LV Hypertrophy", modifiable: false });
  }

  return {
    extractedFeatures: extracted,
    extractedCount: Object.keys(extracted).length,
    details: details
  };
}
