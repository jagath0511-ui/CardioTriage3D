/**
 * CardioTriage 3D — Cath-Lab Workstation Controller & UI Logic
 * Multimodal AI Hackathon 2026 - Track A: Cardiovascular Risk Visualization & Prediction
 */

(function () {
  "use strict";

  // Application State
  const state = {
    featureNames: [],
    featureMetadata: {},
    domainMapping: {},
    presets: [],
    evaluationReport: {},
    currentFeatures: {},
    baselineFeatures: null,
    triageThreshold: 0.50,
    selectedTarget: "Cath", // 'Cath' | 'LAD' | 'LCX' | 'RCA'
    viewMode: "clinician", // 'clinician' | 'bedside'
    activeDomain: "Demographics & History",
    lastPredictionResponse: null,
    isPredicting: false,
    parsedReportFeatures: null
  };

  // 3D Viewer Instance
  let viewer3D = null;
  let debounceTimer = null;

  // DOM Elements
  const dom = {
    presetSelector: document.getElementById("preset-selector"),
    triageModeToggle: document.getElementById("triage-mode-toggle"),
    btnTriageStd: document.getElementById("btn-triage-std"),
    btnTriageEr: document.getElementById("btn-triage-er"),
    viewModeToggle: document.getElementById("view-mode-toggle"),
    btnViewClinician: document.getElementById("btn-view-clinician"),
    btnViewBedside: document.getElementById("btn-view-bedside"),
    btnOpenModal: document.getElementById("btn-open-metrics-modal"),
    btnCloseModal: document.getElementById("btn-close-modal"),
    btnDismissModal: document.getElementById("btn-modal-dismiss"),
    metricsModal: document.getElementById("metrics-modal"),
    metricsTableBody: document.getElementById("metrics-table-body"),

    // Smart Report Import Modal & Toast
    btnOpenImportModal: document.getElementById("btn-open-import-modal"),
    importReportModal: document.getElementById("import-report-modal"),
    btnCloseImportModal: document.getElementById("btn-close-import-modal"),
    btnCancelImportModal: document.getElementById("btn-cancel-import-modal"),
    reportRawText: document.getElementById("report-raw-text"),
    reportCharCount: document.getElementById("report-char-count"),
    btnClearReportText: document.getElementById("btn-clear-report-text"),
    btnApplyReport: document.getElementById("btn-apply-report"),
    detectedBiomarkersCount: document.getElementById("detected-biomarkers-count"),
    detectedBiomarkersChips: document.getElementById("detected-biomarkers-chips"),
    toastContainer: document.getElementById("toast-container"),

    // Left Console
    domainTabsNav: document.getElementById("domain-tabs-nav"),
    featuresContainer: document.getElementById("features-form-container"),
    btnLockBaseline: document.getElementById("btn-lock-baseline"),
    btnSimulateLifestyle: document.getElementById("btn-simulate-lifestyle"),
    baselineStatusBar: document.getElementById("baseline-status-bar"),
    baselineStatusText: document.getElementById("baseline-status-text"),
    btnClearBaseline: document.getElementById("btn-clear-baseline"),
    lockBaselineLabel: document.getElementById("lock-baseline-label"),
    modifiableBadge: document.getElementById("modifiable-summary-badge"),

    // HUD Vitals
    hudPr: document.getElementById("hud-pr-val"),
    hudEf: document.getElementById("hud-ef-val"),
    hudRwma: document.getElementById("hud-rwma-val"),

    // Camera buttons
    btnCamAp: document.getElementById("btn-cam-ap"),
    btnCamLao: document.getElementById("btn-cam-lao"),
    btnCamRao: document.getElementById("btn-cam-rao"),
    btnCamReset: document.getElementById("btn-cam-reset"),

    // 3D Toggles
    toggleChambers: document.getElementById("toggle-chambers"),
    togglePerfusion: document.getElementById("toggle-perfusion"),
    toggleFlow: document.getElementById("toggle-flow"),
    toggleBeat: document.getElementById("toggle-beat"),

    // HUD Vessel Pills
    pillCath: document.getElementById("pill-target-cath"),
    pillLad: document.getElementById("pill-target-lad"),
    pillLcx: document.getElementById("pill-target-lcx"),
    pillRca: document.getElementById("pill-target-rca"),

    // Triage Card
    triageBadge: document.getElementById("triage-badge"),
    triageThresholdLabel: document.getElementById("triage-threshold-label"),
    triageActionTitle: document.getElementById("triage-action-title"),
    triageActionDesc: document.getElementById("triage-action-desc"),
    cohCathVal: document.getElementById("coh-cath-val"),
    cohUnionVal: document.getElementById("coh-union-val"),
    cohChip: document.getElementById("coherence-chip"),
    cohExplanation: document.getElementById("coh-explanation-text"),

    // Risk Cards
    cardCath: document.getElementById("card-risk-cath"),
    cardLad: document.getElementById("card-risk-lad"),
    cardLcx: document.getElementById("card-risk-lcx"),
    cardRca: document.getElementById("card-risk-rca"),
    probCath: document.getElementById("prob-val-cath"),
    probLad: document.getElementById("prob-val-lad"),
    probLcx: document.getElementById("prob-val-lcx"),
    probRca: document.getElementById("prob-val-rca"),
    barCath: document.getElementById("bar-cath"),
    barLad: document.getElementById("bar-lad"),
    barLcx: document.getElementById("bar-lcx"),
    barRca: document.getElementById("bar-rca"),
    badgeCath: document.getElementById("badge-cath"),
    badgeLad: document.getElementById("badge-lad"),
    badgeLcx: document.getElementById("badge-lcx"),
    badgeRca: document.getElementById("badge-rca"),
    deltaCath: document.getElementById("delta-cath"),
    deltaLad: document.getElementById("delta-lad"),
    deltaLcx: document.getElementById("delta-lcx"),
    deltaRca: document.getElementById("delta-rca"),

    // SHAP elements
    shapTargetTitle: document.getElementById("shap-target-title"),
    shapBarsList: document.getElementById("shap-bars-list"),
    clinicianShapView: document.getElementById("clinician-shap-view"),
    bedsideShapView: document.getElementById("bedside-shap-view"),
    bedsideBody: document.getElementById("bedside-body-content")
  };

  // =========================================================================
  // INITIALIZATION
  // =========================================================================
  async function initApp() {
    try {
      // 1. Initialize 3D Viewer
      viewer3D = new Heart3DViewer();
      viewer3D.init("heart-canvas-container");
      viewer3D.onVesselClicked = (target) => selectTargetArtery(target);

      // 2. Fetch /api/init schema & presets
      const resp = await fetch("/api/init");
      if (!resp.ok) throw new Error(`HTTP Error: ${resp.status}`);
      const data = await resp.json();

      state.featureNames = data.feature_names || [];
      state.featureMetadata = data.feature_metadata || {};
      state.presets = data.presets || [];
      state.evaluationReport = data.evaluation_report || {};

      // Build domain partitioning
      state.domainMapping = {};
      state.featureNames.forEach((fname) => {
        const meta = state.featureMetadata[fname];
        const dom = meta.domain || "Vitals & Symptoms";
        if (!state.domainMapping[dom]) state.domainMapping[dom] = [];
        state.domainMapping[dom].push(fname);
      });

      // Populate presets dropdown
      populatePresets(state.presets);

      // Populate 5-Fold CV metrics modal
      populateMetricsModal(state.evaluationReport);

      // Render feature input console for active domain
      renderFeatureForm(state.activeDomain);

      // Attach UI event listeners
      setupEventListeners();

      // Load Preset 1 (Triple-Vessel CAD) as default
      if (state.presets.length > 0) {
        loadPreset(state.presets[0].id);
      }
    } catch (err) {
      console.error("[Init Error]", err);
      dom.featuresContainer.innerHTML = `
        <div style="color:#ef4444; padding:20px; font-size:0.8rem;">
          <strong>Failed to load /api/init:</strong> ${err.message}<br><br>
          Ensure the FastAPI server is running on http://localhost:8000.
        </div>`;
    }
  }

  // =========================================================================
  // PRESET & FORM SETUP
  // =========================================================================
  function populatePresets(presets) {
    dom.presetSelector.innerHTML = presets.map((p, idx) => `
      <option value="${p.id}">${idx + 1}. ${p.name} (Patient #${p.patient_index})</option>
    `).join("");
  }

  function loadPreset(presetId) {
    const p = state.presets.find(item => item.id === presetId);
    if (!p) return;

    dom.presetSelector.value = presetId;
    state.currentFeatures = JSON.parse(JSON.stringify(p.features));
    state.baselineFeatures = null;
    updateBaselineStatusUI();

    // Re-render inputs to reflect new preset
    renderFeatureForm(state.activeDomain);

    // Trigger prediction
    triggerPrediction();
  }

  function populateMetricsModal(evalReport) {
    const metrics = evalReport.cross_validation?.metrics || {};
    const tbody = dom.metricsTableBody;
    tbody.innerHTML = "";

    const targets = ["Cath", "LAD", "LCX", "RCA"];
    targets.forEach(tgt => {
      const m = metrics[tgt];
      if (!m) return;

      const t50 = m.threshold_0_50 || {};
      const t35 = m.threshold_0_35_triage || {};
      const cm35 = t35.confusion_matrix || {};

      const row = document.createElement("tr");
      row.innerHTML = `
        <td><strong style="color:#ffffff;">${tgt}</strong></td>
        <td>${m.positive_prevalence_pct}% (${m.positive_count}/${m.total_samples})</td>
        <td><span style="color:#38bdf8; font-weight:700;">${m.roc_auc.toFixed(4)}</span></td>
        <td>${(t50.accuracy * 100).toFixed(1)}% / ${(t50.precision * 100).toFixed(1)}% / ${(t50.recall * 100).toFixed(1)}% / ${(t50.f1_score * 100).toFixed(1)}%</td>
        <td><strong style="color:#34d399;">${(t35.accuracy * 100).toFixed(1)}% / ${(t35.precision * 100).toFixed(1)}% / ${(t35.recall * 100).toFixed(1)}% / ${(t35.f1_score * 100).toFixed(1)}%</strong></td>
        <td>TP:${cm35.tp} FP:${cm35.fp} TN:${cm35.tn} <span style="color:#f87171;">FN:${cm35.fn}</span></td>
      `;
      tbody.appendChild(row);
    });
  }

  // =========================================================================
  // FEATURE FORM RENDERING
  // =========================================================================
  function renderFeatureForm(domainName) {
    state.activeDomain = domainName;
    const fnames = state.domainMapping[domainName] || [];
    dom.featuresContainer.innerHTML = "";

    fnames.forEach(fname => {
      const meta = state.featureMetadata[fname];
      if (!meta) return;

      const currentVal = state.currentFeatures[fname] !== undefined ? state.currentFeatures[fname] : meta.median;
      const isModifiable = meta.modifiable === true;

      const row = document.createElement("div");
      row.className = `feature-row ${isModifiable ? "is-modifiable" : ""}`;
      row.dataset.feature = fname;

      // Header row
      const labelRow = document.createElement("div");
      labelRow.className = "feature-label-row";
      labelRow.innerHTML = `
        <div class="feature-title-group">
          <span class="feature-name">${meta.label || fname}</span>
          ${isModifiable ? '<span class="mod-pill">Modifiable</span>' : ''}
        </div>
        <span class="feature-value-badge" id="val-badge-${fname}">${formatDisplayVal(fname, currentVal, meta)}</span>
      `;
      row.appendChild(labelRow);

      // Control row
      if (meta.type === "binary") {
        const binGroup = document.createElement("div");
        binGroup.className = "binary-toggle-group";
        const isYes = Number(currentVal) === 1;
        const isSex = fname.toLowerCase() === "sex" || (meta.label && meta.label.toLowerCase().includes("sex"));
        const posLabel = isSex ? "Male" : "Yes";
        const negLabel = isSex ? "Female" : "No";

        binGroup.innerHTML = `
          <button type="button" class="binary-btn ${isYes ? 'active' : ''}" data-val="1">${posLabel}</button>
          <button type="button" class="binary-btn ${!isYes ? 'active' : ''}" data-val="0">${negLabel}</button>
        `;
        binGroup.querySelectorAll(".binary-btn").forEach(btn => {
          btn.addEventListener("click", () => {
            const val = Number(btn.dataset.val);
            state.currentFeatures[fname] = val;
            binGroup.querySelectorAll(".binary-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(`val-badge-${fname}`).textContent = val === 1 ? posLabel : negLabel;
            debouncePredict();
          });
        });
        row.appendChild(binGroup);
      } else if (meta.options && meta.options.length > 0) {
        const select = document.createElement("select");
        select.className = "cat-select";
        select.dataset.feature = fname;
        select.id = `select-${fname}`;
        select.name = fname;
        select.innerHTML = meta.options.map(opt => `
          <option value="${opt.value}" ${Number(currentVal) === opt.value ? 'selected' : ''}>${opt.label}</option>
        `).join("");
        select.addEventListener("change", (e) => {
          const val = Number(e.target.value);
          state.currentFeatures[fname] = val;
          document.getElementById(`val-badge-${fname}`).textContent = formatDisplayVal(fname, val, meta);
          debouncePredict();
        });
        row.appendChild(select);
      } else {
        // Numeric slider
        const sliderRow = document.createElement("div");
        sliderRow.className = "feature-slider-row";
        const minVal = Number.isInteger(meta.min) ? meta.min : Number(meta.min).toFixed(1);
        const maxVal = Number.isInteger(meta.max) ? meta.max : Number(meta.max).toFixed(1);
        const step = (meta.max - meta.min) <= 10 ? "0.1" : "1";

        sliderRow.innerHTML = `
          <span class="feature-meta-minmax">${minVal}</span>
          <input type="range" class="feature-slider" data-feature="${fname}" id="slider-${fname}" name="${fname}" min="${meta.min}" max="${meta.max}" step="${step}" value="${currentVal}">
          <span class="feature-meta-minmax">${maxVal}</span>
        `;
        const slider = sliderRow.querySelector(".feature-slider");
        slider.addEventListener("input", (e) => {
          const val = Number(e.target.value);
          state.currentFeatures[fname] = val;
          document.getElementById(`val-badge-${fname}`).textContent = formatDisplayVal(fname, val, meta);
          debouncePredict();
        });
        row.appendChild(sliderRow);
      }

      dom.featuresContainer.appendChild(row);
    });
  }

  function formatDisplayVal(fname, val, meta) {
    const isSex = fname.toLowerCase() === "sex" || (meta && meta.label && meta.label.toLowerCase().includes("sex"));
    if (isSex) {
      return Number(val) === 1 ? "Male" : "Female";
    }
    if (meta && meta.type === "binary") {
      return Number(val) === 1 ? "Yes" : "No";
    }
    if (meta && meta.options && meta.options.length > 0) {
      const match = meta.options.find(o => o.value === Number(val));
      if (match) return match.label;
    }
    const num = Number(val);
    if (!isNaN(num)) {
      const displayNum = Number.isInteger(num) ? num : Number(num.toFixed(1));
      let unit = (meta && meta.unit) || "";
      if (fname === "BMI" || unit.startsWith("kg/m") || unit.includes("\ufffd")) {
        unit = "kg/m²";
      }
      return `${displayNum} ${unit}`.trim();
    }
    return `${val} ${(meta && meta.unit) || ''}`.trim();
  }

  function debouncePredict() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      triggerPrediction();
    }, 180);
  }

  // =========================================================================
  // API PREDICTION CALL & RESPONSE HANDLING
  // =========================================================================
  async function triggerPrediction() {
    if (state.isPredicting) return;
    state.isPredicting = true;

    // Update HUD vitals instantly
    const prVal = state.currentFeatures["PR"] || 75;
    const efVal = state.currentFeatures["EF-TTE"] || 50;
    const rwmaVal = state.currentFeatures["Region RWMA"] || 0;
    dom.hudPr.textContent = prVal;
    dom.hudEf.textContent = efVal;
    dom.hudRwma.textContent = rwmaVal;

    if (viewer3D) {
      viewer3D.updatePhysiology(prVal, efVal, rwmaVal);
    }

    try {
      const payload = {
        features: state.currentFeatures,
        baseline_features: state.baselineFeatures,
        triage_threshold: state.triageThreshold
      };

      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const errJson = await resp.json();
        throw new Error(errJson.detail || `Prediction failed: ${resp.status}`);
      }

      const result = await resp.json();
      state.lastPredictionResponse = result;

      // Update 3D Canvas
      if (viewer3D) {
        viewer3D.updateRisks(result.predictions, state.triageThreshold);
      }

      // Update UI components
      updateRiskCards(result.predictions, result.counterfactual_comparison);
      updateTriageRecommendation(result);
      renderSHAPAttributions(state.selectedTarget, result);
    } catch (err) {
      console.error("[Prediction Error]", err);
    } finally {
      state.isPredicting = false;
    }
  }

  // =========================================================================
  // UI UPDATE ROUTINES
  // =========================================================================
  function updateRiskCards(predictions, counterfactual) {
    const targets = ["Cath", "LAD", "LCX", "RCA"];
    targets.forEach(tgt => {
      const p = predictions[tgt];
      if (!p) return;

      const key = tgt.toLowerCase();
      const probEl = document.getElementById(`prob-val-${key}`);
      const barEl = document.getElementById(`bar-${key}`);
      const badgeEl = document.getElementById(`badge-${key}`);
      const deltaEl = document.getElementById(`delta-${key}`);

      if (probEl) probEl.textContent = p.percentage.toFixed(1);
      if (barEl) {
        barEl.style.width = `${Math.min(100, p.percentage)}%`;
        barEl.style.backgroundColor = p.color;
      }
      if (badgeEl) {
        badgeEl.textContent = p.tier;
        badgeEl.className = `risk-tier-badge ${p.tier_code}`;
      }

      // Baseline delta chip
      if (deltaEl) {
        if (counterfactual && counterfactual.risk_deltas && counterfactual.risk_deltas[tgt]) {
          const d = counterfactual.risk_deltas[tgt];
          const deltaPct = d.risk_delta_pct;
          deltaEl.classList.remove("hidden");
          deltaEl.textContent = `${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(1)}%`;
          deltaEl.className = `delta-chip ${deltaPct < 0 ? 'delta-reduced' : 'delta-increased'}`;
        } else {
          deltaEl.classList.add("hidden");
        }
      }
    });
  }

  function updateTriageRecommendation(result) {
    const cath = result.predictions["Cath"];
    const union = result.anatomical_union;
    const thresh = state.triageThreshold;

    dom.triageThresholdLabel.textContent = `Cutoff: ${thresh.toFixed(2)}`;
    dom.cohCathVal.textContent = `${cath.percentage.toFixed(1)}%`;
    dom.cohUnionVal.textContent = `${union.percentage.toFixed(1)}%`;
    dom.cohExplanation.textContent = union.coherence_assessment;

    dom.cohChip.textContent = union.is_coherent ? "CONCORDANT" : "DISCORDANT";
    dom.cohChip.style.color = union.is_coherent ? "#34d399" : "#f87171";

    const isHigh = cath.probability >= thresh;
    if (isHigh) {
      dom.triageBadge.className = "triage-badge high";
      dom.triageBadge.textContent = "POSITIVE CAD TRIAGE";
      dom.triageActionTitle.textContent = "Urgent Invasive Coronary Angiography (Cath-Lab Referral)";
      dom.triageActionDesc.textContent = "Predicted risk of significant obstructive coronary artery disease exceeds the triage threshold. Urgent interventional cardiology consultation warranted.";
    } else if (cath.probability >= 0.25) {
      dom.triageBadge.className = "triage-badge moderate";
      dom.triageBadge.textContent = "INTERMEDIATE RISK";
      dom.triageActionTitle.textContent = "Non-Invasive Diagnostic Imaging (CCTA / Stress Echo)";
      dom.triageActionDesc.textContent = "Intermediate pre-test probability. Functional stress echocardiography or coronary computed tomography angiography (CCTA) recommended to rule out ischemia.";
    } else {
      dom.triageBadge.className = "triage-badge low";
      dom.triageBadge.textContent = "LOW RISK / NEGATIVE TRIAGE";
      dom.triageActionTitle.textContent = "Outpatient Preventive Cardiology & Risk Optimization";
      dom.triageActionDesc.textContent = "Probability of critical obstructive lesion is below threshold. Continued aggressive lifestyle modification and guideline lipid management advised.";
    }
  }

  function renderSHAPAttributions(targetKey, result) {
    if (!result || !result.shap_attributions) return;
    const shapData = result.shap_attributions[targetKey];
    if (!shapData) return;

    dom.shapTargetTitle.textContent = targetKey === "Cath" ? "Overall CAD (Cath)" : `${targetKey} Territory`;

    // 1. Clinician View
    const listEl = dom.shapBarsList;
    listEl.innerHTML = "";

    // Combine top drivers
    const topDrivers = shapData.all_attributions || [];
    const maxMag = Math.max(0.1, ...topDrivers.map(d => Math.abs(d.shap_value)));

    topDrivers.slice(0, 10).forEach(d => {
      const isPos = d.shap_value > 0;
      const pctWidth = Math.min(100, (Math.abs(d.shap_value) / maxMag) * 100);

      const row = document.createElement("div");
      row.className = "shap-row";
      row.innerHTML = `
        <div class="shap-row-header">
          <div class="shap-feature-label-group">
            <span class="shap-feature-name" title="${d.label}">${d.label}</span>
            <span class="shap-patient-val">${d.display_value}</span>
            ${d.modifiable ? '<span class="mod-pill">Modifiable</span>' : ''}
          </div>
          <span class="shap-val-num ${isPos ? 'positive' : 'negative'}">
            ${isPos ? '+' : ''}${d.shap_value.toFixed(4)}
          </span>
        </div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill ${isPos ? 'positive' : 'negative'}" style="width: ${pctWidth}%;"></div>
        </div>
      `;
      listEl.appendChild(row);
    });

    // 2. Patient Bedside View
    renderBedsideExplainer(targetKey, topDrivers);
  }

  function renderBedsideExplainer(targetKey, drivers) {
    const vesselDescriptions = {
      Cath: "the entire coronary artery network supplying oxygen-rich blood to your heart muscle",
      LAD: "the Left Anterior Descending artery ('Widowmaker'), which feeds the main front wall of your heart",
      LCX: "the Circumflex artery, which wraps around to nourish the left side and back wall of your heart",
      RCA: "the Right Coronary Artery, which powers the lower bottom wall and rhythm center of your heart"
    };

    const topRiskInc = drivers.filter(d => d.shap_value > 0).slice(0, 3);
    const topProtective = drivers.filter(d => d.shap_value < 0).slice(0, 2);

    let html = `
      <div class="bedside-point">
        <strong>Vessel Territory:</strong> This view examines <em>${vesselDescriptions[targetKey] || targetKey}</em>.
      </div>
    `;

    if (topRiskInc.length > 0) {
      html += `
        <div class="bedside-point">
          <strong>Key Numbers Pushing Your Risk Up:</strong><br>
          ${topRiskInc.map(d => `• <strong>${d.label}</strong> (Current Value: <em>${d.display_value}</em>) is adding to blood vessel strain.`).join("<br>")}
        </div>
      `;
    }

    if (topProtective.length > 0) {
      html += `
        <div class="bedside-point">
          <strong>Protective Factors Working in Your Favor:</strong><br>
          ${topProtective.map(d => `• <strong>${d.label}</strong> (Current Value: <em>${d.display_value}</em>) is helping protect this vessel from narrowing.`).join("<br>")}
        </div>
      `;
    }

    html += `
      <div class="bedside-point" style="border-left-color:#10b981;">
        <strong>Empowerment Action:</strong> Using the <em>"Simulate Risk Control"</em> button on the left demonstrates how targeted adjustments in your blood pressure, LDL cholesterol, and smoking habits directly lower your cardiac risk.
      </div>
    `;

    dom.bedsideBody.innerHTML = html;
  }

  function selectTargetArtery(targetKey) {
    state.selectedTarget = targetKey;

    // Update risk cards active state
    const targets = ["cath", "lad", "lcx", "rca"];
    targets.forEach(t => {
      const card = document.getElementById(`card-risk-${t}`);
      const pill = document.getElementById(`pill-target-${t}`);
      if (card) card.classList.remove("active");
      if (pill) pill.classList.remove("active");
    });

    const activeCard = document.getElementById(`card-risk-${targetKey.toLowerCase()}`);
    if (activeCard) activeCard.classList.add("active");

    const activePill = document.getElementById(`pill-target-${targetKey.toLowerCase()}`);
    if (activePill) activePill.classList.add("active");

    // Update 3D Viewer highlight
    if (viewer3D) {
      viewer3D.selectVessel(targetKey);
    }

    // Update SHAP
    if (state.lastPredictionResponse) {
      renderSHAPAttributions(targetKey, state.lastPredictionResponse);
    }
  }

  // =========================================================================
  // COUNTERFACTUAL ACTIONS
  // =========================================================================
  function lockBaseline() {
    state.baselineFeatures = JSON.parse(JSON.stringify(state.currentFeatures));
    updateBaselineStatusUI();
    triggerPrediction();
  }

  function clearBaseline() {
    state.baselineFeatures = null;
    updateBaselineStatusUI();
    triggerPrediction();
  }

  function updateBaselineStatusUI() {
    if (state.baselineFeatures) {
      dom.btnLockBaseline.classList.add("locked");
      dom.lockBaselineLabel.textContent = "Baseline Locked";
      dom.baselineStatusBar.classList.remove("hidden");
    } else {
      dom.btnLockBaseline.classList.remove("locked");
      dom.lockBaselineLabel.textContent = "Lock Baseline";
      dom.baselineStatusBar.classList.add("hidden");
    }
  }

  function simulateOptimalRiskControl() {
    // If no baseline locked, lock current first
    if (!state.baselineFeatures) {
      state.baselineFeatures = JSON.parse(JSON.stringify(state.currentFeatures));
      updateBaselineStatusUI();
    }

    // Apply guideline-directed targets:
    // BP=120, LDL=70, TG=110, HDL=55, FBS=90, Current Smoker=0, BMI=23
    state.currentFeatures["BP"] = 120;
    state.currentFeatures["LDL"] = 70;
    state.currentFeatures["TG"] = 110;
    state.currentFeatures["HDL"] = 55;
    state.currentFeatures["FBS"] = 90;
    state.currentFeatures["Current Smoker"] = 0;
    state.currentFeatures["BMI"] = 23;
    state.currentFeatures["Obesity"] = 0;

    // Re-render form & predict
    renderFeatureForm(state.activeDomain);
    triggerPrediction();
  }

  // =========================================================================
  // SMART CLINICAL REPORT AUTO-FILLER & TOAST NOTIFICATION
  // =========================================================================
  function showToast(title, message, durationMs = 12000) {
    if (!dom.toastContainer) return;

    const toast = document.createElement("div");
    toast.className = "toast";
    toast.innerHTML = `
      <span class="toast-icon">🫀</span>
      <div class="toast-content">
        <div class="toast-title">${title}</div>
        <div class="toast-message">${message}</div>
      </div>
      <button class="toast-close-btn" title="Dismiss">&times;</button>
    `;

    const closeBtn = toast.querySelector(".toast-close-btn");
    closeBtn.addEventListener("click", () => {
      toast.classList.add("toast-fadeout");
      setTimeout(() => toast.remove(), 300);
    });

    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
      if (toast.parentElement) {
        toast.classList.add("toast-fadeout");
        setTimeout(() => toast.remove(), 300);
      }
    }, durationMs);
  }

  function updateReportExtractionPreview() {
    if (!dom.reportRawText) return;

    const text = dom.reportRawText.value || "";
    dom.reportCharCount.textContent = `${text.length} characters`;

    if (!text.trim()) {
      state.parsedReportFeatures = null;
      dom.detectedBiomarkersCount.textContent = "0 Detected";
      dom.detectedBiomarkersChips.innerHTML = `
        <span class="empty-chips-msg">Type or paste clinical text above to extract variables in real time.</span>
      `;
      dom.btnApplyReport.disabled = true;
      return;
    }

    // Call parser function from reportParser.js
    const res = (typeof parseClinicalReportClient === "function")
      ? parseClinicalReportClient(text, state.featureMetadata)
      : { extractedFeatures: {}, extractedCount: 0, details: [] };

    state.parsedReportFeatures = res.extractedFeatures;
    const count = res.extractedCount;

    dom.detectedBiomarkersCount.textContent = `${count} Detected`;
    dom.btnApplyReport.disabled = count === 0;

    if (count === 0) {
      dom.detectedBiomarkersChips.innerHTML = `
        <span class="empty-chips-msg">No recognized clinical biomarkers found yet. Try keywords like "BP", "LDL", "FBS", "Smoker", "LVEF".</span>
      `;
      return;
    }

    dom.detectedBiomarkersChips.innerHTML = res.details.map(d => `
      <div class="biomarker-chip ${d.modifiable ? 'modifiable' : ''}" title="${d.label} (${d.feature})">
        <span class="chip-var">${d.label}:</span>
        <span class="chip-val">${d.value}</span>
        ${d.modifiable ? '<span class="chip-mod-badge">Modifiable</span>' : ''}
      </div>
    `).join("");
  }

  function applyParsedReportToTwin() {
    if (!state.parsedReportFeatures || Object.keys(state.parsedReportFeatures).length === 0) return;

    const count = Object.keys(state.parsedReportFeatures).length;

    // Apply extracted variables into currentFeatures
    Object.assign(state.currentFeatures, state.parsedReportFeatures);

    // Sync pulse rate and ejection fraction to 3D heart
    const pr = state.currentFeatures["PR"] || 75;
    const ef = state.currentFeatures["EF-TTE"] || 50;
    const rwma = state.currentFeatures["Region RWMA"] || 0;
    viewer3D?.updatePhysiology(pr, ef, rwma);

    // Re-render active domain form inputs
    renderFeatureForm(state.activeDomain);

    // Trigger instant prediction
    triggerPrediction();

    // Close modal
    dom.importReportModal.classList.add("hidden");

    // Success Toast
    showToast(
      "Report Auto-Filled Successfully",
      `Successfully parsed ${count} clinical biomarkers from report. 3D Digital Twin updated.`
    );
  }

  // =========================================================================
  // EVENT LISTENERS SETUP
  // =========================================================================
  function setupEventListeners() {
    // Smart Clinical Report Auto-Filler
    dom.btnOpenImportModal.addEventListener("click", () => {
      dom.importReportModal.classList.remove("hidden");
      updateReportExtractionPreview();
      dom.reportRawText.focus();
    });

    dom.btnCloseImportModal.addEventListener("click", () => dom.importReportModal.classList.add("hidden"));
    dom.btnCancelImportModal.addEventListener("click", () => dom.importReportModal.classList.add("hidden"));
    dom.importReportModal.addEventListener("click", (e) => {
      if (e.target === dom.importReportModal) dom.importReportModal.classList.add("hidden");
    });

    dom.reportRawText.addEventListener("input", updateReportExtractionPreview);

    dom.btnClearReportText.addEventListener("click", () => {
      dom.reportRawText.value = "";
      updateReportExtractionPreview();
      dom.reportRawText.focus();
    });

    dom.btnApplyReport.addEventListener("click", applyParsedReportToTwin);

    // 1-Click Sample Reports
    document.querySelectorAll(".sample-chip-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const sampleKey = btn.dataset.sample;
        if (typeof SAMPLE_CLINICAL_REPORTS !== "undefined" && SAMPLE_CLINICAL_REPORTS[sampleKey]) {
          dom.reportRawText.value = SAMPLE_CLINICAL_REPORTS[sampleKey];
          updateReportExtractionPreview();
        }
      });
    });

    // Preset dropdown
    dom.presetSelector.addEventListener("change", (e) => loadPreset(e.target.value));

    // Triage sensitivity toggle
    dom.btnTriageStd.addEventListener("click", () => {
      state.triageThreshold = 0.50;
      dom.btnTriageStd.classList.add("active");
      dom.btnTriageEr.classList.remove("active");
      triggerPrediction();
    });

    dom.btnTriageEr.addEventListener("click", () => {
      state.triageThreshold = 0.35;
      dom.btnTriageEr.classList.add("active");
      dom.btnTriageStd.classList.remove("active");
      triggerPrediction();
    });

    // View mode toggle
    dom.btnViewClinician.addEventListener("click", () => {
      state.viewMode = "clinician";
      dom.btnViewClinician.classList.add("active");
      dom.btnViewBedside.classList.remove("active");
      dom.clinicianShapView.classList.remove("hidden");
      dom.bedsideShapView.classList.add("hidden");
    });

    dom.btnViewBedside.addEventListener("click", () => {
      state.viewMode = "bedside";
      dom.btnViewBedside.classList.add("active");
      dom.btnViewClinician.classList.remove("active");
      dom.clinicianShapView.classList.add("hidden");
      dom.bedsideShapView.classList.remove("hidden");
    });

    // 5-Fold CV Modal
    dom.btnOpenModal.addEventListener("click", () => dom.metricsModal.classList.remove("hidden"));
    dom.btnCloseModal.addEventListener("click", () => dom.metricsModal.classList.add("hidden"));
    dom.btnDismissModal.addEventListener("click", () => dom.metricsModal.classList.add("hidden"));
    dom.metricsModal.addEventListener("click", (e) => {
      if (e.target === dom.metricsModal) dom.metricsModal.classList.add("hidden");
    });

    // Domain Tabs
    dom.domainTabsNav.querySelectorAll(".domain-tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        dom.domainTabsNav.querySelectorAll(".domain-tab-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderFeatureForm(btn.dataset.domain);
      });
    });

    // Counterfactual Actions
    dom.btnLockBaseline.addEventListener("click", lockBaseline);
    dom.btnClearBaseline.addEventListener("click", clearBaseline);
    dom.btnSimulateLifestyle.addEventListener("click", simulateOptimalRiskControl);

    // Camera Angles
    dom.btnCamAp.addEventListener("click", () => {
      dom.btnCamAp.classList.add("active");
      dom.btnCamLao.classList.remove("active");
      dom.btnCamRao.classList.remove("active");
      viewer3D?.setCameraView("ap");
    });
    dom.btnCamLao.addEventListener("click", () => {
      dom.btnCamLao.classList.add("active");
      dom.btnCamAp.classList.remove("active");
      dom.btnCamRao.classList.remove("active");
      viewer3D?.setCameraView("lao");
    });
    dom.btnCamRao.addEventListener("click", () => {
      dom.btnCamRao.classList.add("active");
      dom.btnCamAp.classList.remove("active");
      dom.btnCamLao.classList.remove("active");
      viewer3D?.setCameraView("rao");
    });
    dom.btnCamReset.addEventListener("click", () => {
      viewer3D?.setCameraView("reset");
    });

    // 3D Viewport Layers
    dom.toggleChambers.addEventListener("change", (e) => viewer3D?.setLayerVisibility("chambers", e.target.checked));
    dom.togglePerfusion.addEventListener("change", (e) => viewer3D?.setLayerVisibility("perfusion", e.target.checked));
    dom.toggleFlow.addEventListener("change", (e) => viewer3D?.setLayerVisibility("flow", e.target.checked));
    dom.toggleBeat.addEventListener("change", (e) => viewer3D?.setLayerVisibility("beat", e.target.checked));

    // HUD Vessel Pills
    dom.pillCath.addEventListener("click", () => selectTargetArtery("Cath"));
    dom.pillLad.addEventListener("click", () => selectTargetArtery("LAD"));
    dom.pillLcx.addEventListener("click", () => selectTargetArtery("LCX"));
    dom.pillRca.addEventListener("click", () => selectTargetArtery("RCA"));

    // Risk Cards Click
    dom.cardCath.addEventListener("click", () => selectTargetArtery("Cath"));
    dom.cardLad.addEventListener("click", () => selectTargetArtery("LAD"));
    dom.cardLcx.addEventListener("click", () => selectTargetArtery("LCX"));
    dom.cardRca.addEventListener("click", () => selectTargetArtery("RCA"));
  }

  // Launch on document ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
  } else {
    initApp();
  }
})();

// === CLINICAL UI SANITIZER & AUTO-BMI CALCULATOR ===
function applyClinicalUIPolish() {
  // 1. Round any raw floating point numbers (e.g., 26.122448979591837 -> 26.1)
  document.querySelectorAll('span, div, label, strong, b').forEach((el) => {
    if (el.children.length === 0 && el.textContent) {
      const txt = el.textContent.trim();
      // Match raw decimals with more than 2 decimal places
      if (/^-?\d+\.\d{3,}(\s*[a-zA-Z/%²]*)?$/.test(txt)) {
        const match = txt.match(/^(-?\d+\.\d+)(\s*[a-zA-Z/%²]*)?$/);
        if (match) {
          const num = parseFloat(match[1]);
          const unit = match[2] || '';
          el.textContent = `${num.toFixed(1)}${unit}`;
        }
      }
    }
  });

  // 2. Fix "Biological Sex: Yes / No" -> "Male / Female"
  document.querySelectorAll('div, section, li').forEach((container) => {
    const labelText = container.textContent || '';
    if (
      labelText.includes('Biological Sex') &&
      container.querySelectorAll('button').length === 2
    ) {
      const btns = container.querySelectorAll('button');
      if (btns[0].textContent.trim() === 'Yes') btns[0].textContent = 'Male';
      if (btns[1].textContent.trim() === 'No') btns[1].textContent = 'Female';
      // Also update the top-right status badge inside the Sex card
      container.querySelectorAll('span, .badge').forEach((badge) => {
        if (badge.textContent.trim() === 'Yes') badge.textContent = 'Male';
        if (badge.textContent.trim() === 'No') badge.textContent = 'Female';
      });
    }
  });

  // 3. Auto-compute BMI when Weight or Height sliders change
  const weightInput = document.querySelector(
    'input[data-feature="Weight"], input[name="Weight"], #slider-Weight'
  );
  const heightInput = document.querySelector(
    'input[data-feature="Length"], input[data-feature="Height"], input[name="Length"], #slider-Length, #slider-Height'
  );
  const bmiInput = document.querySelector(
    'input[data-feature="BMI"], input[name="BMI"], #slider-BMI'
  );

  if (weightInput && heightInput && bmiInput && !weightInput.dataset.bmiBound) {
    const syncBMI = () => {
      const w = parseFloat(weightInput.value) || 75;
      const hCm = parseFloat(heightInput.value) || 170;
      const bmi = +(w / Math.pow(hCm / 100, 2)).toFixed(1);
      bmiInput.value = bmi;
      bmiInput.dispatchEvent(new Event('input', { bubbles: true }));
    };
    weightInput.addEventListener('input', syncBMI);
    heightInput.addEventListener('input', syncBMI);
    weightInput.dataset.bmiBound = 'true';
  }
}

// Run immediately and observe DOM updates when presets or sliders change
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      applyClinicalUIPolish();
      const observer = new MutationObserver(() => applyClinicalUIPolish());
      observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    });
  } else {
    applyClinicalUIPolish();
    const observer = new MutationObserver(() => applyClinicalUIPolish());
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
  }
}

