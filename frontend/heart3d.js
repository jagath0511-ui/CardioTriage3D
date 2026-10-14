/**
 * CardioTriage 3D — High-End Medical 3D Coronary & Myocardial Digital Twin
 * Multimodal AI Hackathon 2026 - Track A: Cardiovascular Risk Visualization & Prediction
 *
 * Enhanced Graphic Features:
 * 1. Solid Anatomical Ventricular Mass: Tapered conical muscular left/right ventricle with distinct apex pointing
 *    toward bottom-left, broad base at atria, and realistic epicardial grooves.
 * 2. Deep Clinical Myocardial Shading: Solid MeshPhysicalMaterial with burgundy/cardiac red base (#991b1b),
 *    high roughness (0.40), moist glistening clearcoat (0.35), and zero transparency.
 * 3. Prominent, Thicker Coronary Trees (LAD, LCX, RCA): CatmullRomCurve3 tubes (radius 0.082) hugging the
 *    epicardial surface tightly with major clinical branches (D1, D2, OM, AM, PDA).
 * 4. Anatomical Base Great Vessels: Aortic Arch with the 3 classic arterial trunks (Brachiocephalic, Left Common
 *    Carotid, Left Subclavian) + Pulmonary Trunk and Vena Cava.
 * 5. Professional 3-Point Medical Studio Lighting: Key light top-left, soft warm fill, and cyan rim light (#06b6d4).
 * 6. Dynamic Stenosis Coloration & Pulsing 3D Torus Plaque Constriction Rings.
 * 7. Real-Time Blood-Flow Particles & PR/EF-TTE Synced Ventricular Contraction.
 */

// === DROP-IN 3D ANATOMICAL & CAMERA FIXES FOR `frontend/heart3d.js` ===

// Preset Cath-Lab Projections (AP, LAO, RAO) with un-clipped distance:
const CATH_VIEWS = {
  AP:  { pos: [0, 0.35, 7.6],    target: [0, 0.25, 0] },
  LAO: { pos: [4.8, 0.85, 5.8],  target: [0, 0.25, 0] }, // Left Anterior Oblique (40° LAO, 15° Cranial)
  RAO: { pos: [-4.6, 0.30, 5.9], target: [0, 0.25, 0] }  // Right Anterior Oblique (35° RAO)
};

// 2. REPLACE FLAT BLUE PVC CYLINDER WITH SMOOTH PULMONARY ARTERY SPLINE
function createPulmonaryTrunk(THREE) {
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.18, 0.55, 0.65),
    new THREE.Vector3(0.05, 1.15, 0.52),
    new THREE.Vector3(-0.35, 1.48, 0.20),
    new THREE.Vector3(-0.85, 1.52, -0.15)
  ]);
  const geo = new THREE.TubeGeometry(curve, 48, 0.26, 24, false);
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0x2563eb,
    roughness: 0.38,
    metalness: 0.08,
    clearcoat: 0.4,
    clearcoatRoughness: 0.25
  });
  return new THREE.Mesh(geo, mat);
}

// 3. REPLACE BULKY PINK TORUS "DONUTS" WITH SLEEK GLOWING STENOSIS CUFFS
// Instead of new THREE.TorusGeometry(0.22, 0.08, ...), use a tight axial vascular cuff:
function createStenosisCuff(THREE, pointOnCurve, tangentVector, severityProb) {
  const group = new THREE.Group();

  // Inner dark-crimson plaque core (hugs the artery wall tightly)
  const coreGeo = new THREE.CylinderGeometry(0.065, 0.065, 0.22, 24, 1, false);
  const coreMat = new THREE.MeshPhysicalMaterial({
    color: severityProb >= 0.5 ? 0xdc2626 : 0xf59e0b,
    emissive: severityProb >= 0.5 ? 0x991b1b : 0xb45309,
    emissiveIntensity: 0.85,
    roughness: 0.25,
    clearcoat: 0.6
  });
  const coreMesh = new THREE.Mesh(coreGeo, coreMat);
  group.add(coreMesh);

  // Subtle translucent outer Halo (shows inflammation/ischemic focal zone without looking like a donut)
  const haloGeo = new THREE.CylinderGeometry(0.115, 0.115, 0.28, 24, 1, true);
  const haloMat = new THREE.MeshBasicMaterial({
    color: severityProb >= 0.5 ? 0xef4444 : 0xfbbf24,
    transparent: true,
    opacity: 0.32,
    side: THREE.DoubleSide,
    depthWrite: false
  });
  const haloMesh = new THREE.Mesh(haloGeo, haloMat);
  group.add(haloMesh);

  // Align cylinder along the exact artery tangent vector
  group.position.copy(pointOnCurve);
  const up = new THREE.Vector3(0, 1, 0);
  group.quaternion.setFromUnitVectors(up, tangentVector.clone().normalize());

  return group;
}

class Heart3DViewer {
  constructor() {
    this.container = null;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    // Scene groups
    this.rootGroup = new THREE.Group();
    this.ventriclesGroup = new THREE.Group();
    this.atriaGroup = new THREE.Group();
    this.greatVesselsGroup = new THREE.Group();
    this.coronaryGroup = new THREE.Group();
    this.plaqueGroup = new THREE.Group();
    this.perfusionGroup = new THREE.Group();
    this.particlesGroup = new THREE.Group();

    // Meshes, Curves & Materials
    this.vessels = {};
    this.vesselCurves = {};
    this.plaqueMeshes = {};
    this.perfusionMeshes = {};
    this.flowParticles = [];
    this.ventricleMesh = null;
    this.myocardialMat = null;

    // Physiology State
    this.pulseRate = 75; // BPM
    this.ejectionFraction = 50; // %
    this.rwma = 0; // Regional Wall Motion Abnormality
    this.triageThreshold = 0.50;
    this.selectedVessel = null;

    // View Layer Toggles
    this.showChambers = true;
    this.showPerfusion = true;
    this.showFlow = true;
    this.enableBeating = true;

    // Callbacks
    this.onVesselClicked = null;

    // Animation Clock
    this.clock = new THREE.Clock();
    this.timeAccumulator = 0;
  }

  init(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`Heart3DViewer: Container #${containerId} not found!`);
      return;
    }

    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x070b14); // Dark Cath-Lab cockpit background
    this.scene.fog = new THREE.FogExp2(0x070b14, 0.035);

    // 2. FIX CAMERA FRAMING (Prevents Aortic Arch Clipping at Top of Canvas)
    this.camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100);
    const camera = this.camera;
    camera.fov = 38;
    camera.position.set(0, 0.35, 7.6); // Pulled back from ~5.5 to 7.6 so full aorta & apex fit
    camera.updateProjectionMatrix();

    // 3. Renderer with high dynamic range ACES Filmic tonemapping
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.container.appendChild(this.renderer.domElement);

    // 4. OrbitControls
    if (THREE.OrbitControls) {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      const controls = this.controls;
      if (controls) {
        controls.target.set(0, 0.25, 0); // Centered slightly higher to balance top vessels
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.minDistance = 4.2;
        controls.maxDistance = 12.0;
        controls.maxPolarAngle = Math.PI * 0.88;
        controls.update();
      }
    }

    // 5. Professional 3-Point Medical Studio Lighting Setup
    this.setupMedicalLighting();

    // 6. Build High-Realism Anatomical Heart Model
    this.rootGroup.add(this.ventriclesGroup);
    this.rootGroup.add(this.atriaGroup);
    this.rootGroup.add(this.greatVesselsGroup);
    this.rootGroup.add(this.coronaryGroup);
    this.rootGroup.add(this.plaqueGroup);
    this.rootGroup.add(this.perfusionGroup);
    this.rootGroup.add(this.particlesGroup);
    this.scene.add(this.rootGroup);

    this.buildSolidMyocardialHeart();
    this.buildGreatVessels();
    this.buildThickCoronaryTree();
    this.buildPerfusionTerritories();
    this.buildBloodFlowParticles();

    // 7. Event Listeners
    window.addEventListener("resize", () => this.onWindowResize());
    this.renderer.domElement.addEventListener("pointerdown", (e) => this.onPointerDown(e));

    // 8. Animation Loop
    this.animate();
  }

  // =========================================================================
  // 1. PROFESSIONAL 3-POINT MEDICAL STUDIO LIGHTING SETUP
  // =========================================================================
  setupMedicalLighting() {
    // 1. Soft Ambient Medical Glow
    const ambientLight = new THREE.AmbientLight(0x0f172a, 0.55);
    this.scene.add(ambientLight);

    // 2. Bright Key Spotlight (Top-Left Surgical Theater Light)
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.45);
    keyLight.position.set(-4.5, 5.5, 4.5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    this.scene.add(keyLight);

    // 3. Soft Warm Fill Light (Bottom-Right) to soften deep shadows
    const fillLight = new THREE.DirectionalLight(0xfed7aa, 0.65);
    fillLight.position.set(4.0, -2.0, 3.0);
    this.scene.add(fillLight);

    // 4. Subtle Crisp Cyan Rim / Backlight (#06b6d4) to make cardiac silhouette pop against #070b14
    const rimLight = new THREE.DirectionalLight(0x06b6d4, 2.2);
    rimLight.position.set(0.5, 3.5, -4.5);
    this.scene.add(rimLight);

    // 5. Subtle Cardiac Accent Point Light near Apex
    const apexLight = new THREE.PointLight(0x991b1b, 0.55, 6);
    apexLight.position.set(-0.5, -1.6, 1.2);
    this.scene.add(apexLight);
  }

  // =========================================================================
  // 2. SOLID ANATOMICAL VENTRICULAR MASS & CLINICAL MYOCARDIUM SHADING
  // =========================================================================
  buildSolidMyocardialHeart() {
    // Rich, fleshy, solid clinical myocardial material (MeshPhysicalMaterial)
    // Deep burgundy/cardiac red (#881337), organic roughness (0.44), glistening clearcoat (0.35)
    this.myocardialMat = new THREE.MeshPhysicalMaterial({
      color: 0x881337,        // Deep burgundy / cardiac muscle red
      emissive: 0x2e0610,     // Deep blood emissive undertone
      emissiveIntensity: 0.18,
      roughness: 0.44,        // Natural muscular tissue texture
      metalness: 0.08,        // Proteinaceous wet sheen
      clearcoat: 0.35,        // Glistening epicardial serosa reflection
      clearcoatRoughness: 0.22,
      reflectivity: 0.55,
      transparent: false,     // 100% SOLID OPAQUE DEPTH — NO GLASS!
      depthWrite: true,
      side: THREE.FrontSide
    });

    // --- UNIFIED SCULPTED VENTRICULAR MASS (Conical, muscular, tapering to bottom-left) ---
    // Sculpted organic muscular core incorporating LV, RV, and anterior sulcus
    const lvGeometry = this.createAnatomicalVentricularGeometry();
    this.ventricleMesh = new THREE.Mesh(lvGeometry, this.myocardialMat);
    this.ventricleMesh.castShadow = true;
    this.ventricleMesh.receiveShadow = true;
    this.ventriclesGroup.add(this.ventricleMesh);

    // --- ATRIAL CHAMBERS (Superior muscular base) ---
    const atriumMat = new THREE.MeshPhysicalMaterial({
      color: 0x5b132b,        // Deep atrial myocardium
      emissive: 0x20050e,
      emissiveIntensity: 0.18,
      roughness: 0.48,
      metalness: 0.08,
      clearcoat: 0.25,
      depthWrite: true
    });

    // Left Atrium (Posterior-superior)
    const laGeo = new THREE.SphereGeometry(0.65, 28, 22);
    laGeo.scale(1.05, 0.90, 0.95);
    const laMesh = new THREE.Mesh(laGeo, atriumMat);
    laMesh.position.set(-0.35, 0.80, -0.32);
    this.atriaGroup.add(laMesh);

    // Right Atrium (Lateral-superior)
    const raGeo = new THREE.SphereGeometry(0.68, 28, 22);
    raGeo.scale(0.95, 0.95, 0.95);
    const raMesh = new THREE.Mesh(raGeo, atriumMat);
    raMesh.position.set(0.72, 0.72, 0.05);
    this.atriaGroup.add(raMesh);

    // Left Auricle / Atrial Appendage (curves around base of pulmonary artery)
    const laaCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.40, 0.75, -0.05),
      new THREE.Vector3(-0.65, 0.70, 0.22),
      new THREE.Vector3(-0.50, 0.52, 0.45)
    ]);
    const laaGeo = new THREE.TubeGeometry(laaCurve, 16, 0.13, 12, false);
    const laaMesh = new THREE.Mesh(laaGeo, atriumMat);
    this.atriaGroup.add(laaMesh);

    // Right Auricle (overlaps ascending aorta anteriorly)
    const raaCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.60, 0.75, 0.15),
      new THREE.Vector3(0.38, 0.76, 0.45),
      new THREE.Vector3(0.22, 0.65, 0.50)
    ]);
    const raaGeo = new THREE.TubeGeometry(raaCurve, 16, 0.13, 12, false);
    const raaMesh = new THREE.Mesh(raaGeo, atriumMat);
    this.atriaGroup.add(raaMesh);
  }

  // Sculpted custom organic muscular ventricular geometry from smooth closed sphere
  createAnatomicalVentricularGeometry() {
    const geo = new THREE.SphereGeometry(1.05, 54, 42);
    const pos = geo.attributes.position;
    const v = new THREE.Vector3();

    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);

      // Height normalized from apex (y ≈ -1.05) to base (y ≈ +1.05)
      const hRatio = (v.y + 1.05) / 2.10; // 0 at bottom, 1 at top

      // 1. Tapering to a conical point at the apex:
      if (v.y < 0) {
        const taper = 0.32 + 0.68 * Math.pow(hRatio / 0.50, 1.15);
        v.x *= taper;
        v.z *= taper;
        v.y *= 1.22; // elongate towards apex
      } else {
        v.x *= 1.02;
        v.z *= 0.92;
      }

      // 2. Anatomical cardiac axis tilt (apex points towards patient's left and anteriorly)
      const tiltFactor = Math.pow(1.0 - hRatio, 1.25);
      v.x -= 0.34 * tiltFactor;
      v.z += 0.22 * tiltFactor;

      // 3. Left ventricular lateral muscular contour
      if (v.x < 0) {
        const lvBulge = Math.sin(hRatio * Math.PI) * 0.15;
        v.x -= lvBulge;
      }

      // 4. Right ventricular anterolateral shell (integrated smoothly on X > 0)
      if (v.x > 0 && v.z > 0 && hRatio > 0.20 && hRatio < 0.85) {
        const rvBulge = Math.sin((hRatio - 0.20) / 0.65 * Math.PI) * 0.18;
        v.x += rvBulge * 0.7;
        v.z += rvBulge * 0.6;
      }

      // 5. Anterior interventricular sulcus (groove where LAD artery sits)
      const sulcusX = -0.18 - 0.14 * (1.0 - hRatio);
      const distToSulcus = Math.abs(v.x - sulcusX);
      if (distToSulcus < 0.28 && v.z > 0.4 && hRatio > 0.15 && hRatio < 0.85) {
        const sulcusDepth = Math.cos((distToSulcus / 0.28) * (Math.PI / 2)) * 0.06;
        v.z -= sulcusDepth;
      }

      pos.setXYZ(i, v.x, v.y - 0.25, v.z);
    }

    geo.computeVertexNormals();
    return geo;
  }

  // =========================================================================
  // 3. ANATOMICAL BASE GREAT VESSELS (Aorta, Branches & Pulmonary Trunk)
  // =========================================================================
  buildGreatVessels() {
    const aortaMat = new THREE.MeshPhysicalMaterial({
      color: 0xbe123c,        // Vibrant arterial crimson
      roughness: 0.28,
      metalness: 0.15,
      clearcoat: 0.40,
      clearcoatRoughness: 0.18,
      depthWrite: true
    });

    const pulmonaryMat = new THREE.MeshPhysicalMaterial({
      color: 0x1d4ed8,        // Deep venous blue
      roughness: 0.32,
      metalness: 0.12,
      clearcoat: 0.35,
      depthWrite: true
    });

    const venaCavaMat = new THREE.MeshPhysicalMaterial({
      color: 0x1e3a8a,        // Dark cava blue
      roughness: 0.38,
      depthWrite: true
    });

    // --- ASCENDING AORTA & AORTIC ARCH ---
    // Curves from central LV root superiorly, backward, and arching to the left
    const aortaCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.05, 0.70, 0.18),  // Aortic valve root
      new THREE.Vector3(0.15, 1.25, 0.10),  // Ascending aorta
      new THREE.Vector3(0.05, 1.68, -0.15), // Peak of aortic arch
      new THREE.Vector3(-0.35, 1.55, -0.38),// Arch turning left
      new THREE.Vector3(-0.48, 0.95, -0.55) // Descending aorta
    ]);
    const aortaGeo = new THREE.TubeGeometry(aortaCurve, 36, 0.24, 18, false);
    const aortaMesh = new THREE.Mesh(aortaGeo, aortaMat);
    this.greatVesselsGroup.add(aortaMesh);

    // The 3 Great Arterial Trunks off the Aortic Arch:
    // 1. Brachiocephalic Trunk (Innominate artery)
    const bctCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.12, 1.62, 0.02),
      new THREE.Vector3(0.28, 2.05, 0.05)
    ]);
    this.greatVesselsGroup.add(new THREE.Mesh(new THREE.TubeGeometry(bctCurve, 12, 0.09, 12, false), aortaMat));

    // 2. Left Common Carotid Artery
    const lccCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.05, 1.68, -0.12),
      new THREE.Vector3(-0.02, 2.10, -0.10)
    ]);
    this.greatVesselsGroup.add(new THREE.Mesh(new THREE.TubeGeometry(lccCurve, 12, 0.075, 12, false), aortaMat));

    // 3. Left Subclavian Artery
    const lscCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.22, 1.64, -0.25),
      new THREE.Vector3(-0.32, 2.08, -0.28)
    ]);
    this.greatVesselsGroup.add(new THREE.Mesh(new THREE.TubeGeometry(lscCurve, 12, 0.08, 12, false), aortaMat));

    // --- MAIN PULMONARY TRUNK & BIFURCATION ---
    // 2. REPLACE FLAT BLUE PVC CYLINDER WITH SMOOTH PULMONARY ARTERY SPLINE
    const paMesh = createPulmonaryTrunk(THREE);
    this.greatVesselsGroup.add(paMesh);

    // Right Pulmonary Artery (runs under aortic arch)
    const rpaCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.35, 1.48, 0.20),
      new THREE.Vector3(0.18, 1.35, -0.15),
      new THREE.Vector3(0.68, 1.20, -0.25)
    ]);
    this.greatVesselsGroup.add(new THREE.Mesh(new THREE.TubeGeometry(rpaCurve, 18, 0.14, 14, false), paMesh.material));

    // --- SUPERIOR VENA CAVA (SVC) ---
    const svcCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.82, 1.75, -0.05),
      new THREE.Vector3(0.78, 0.95, 0.05)
    ]);
    const svcGeo = new THREE.TubeGeometry(svcCurve, 16, 0.18, 14, false);
    const svcMesh = new THREE.Mesh(svcGeo, venaCavaMat);
    this.greatVesselsGroup.add(svcMesh);
  }

  // =========================================================================
  // 4. THICK, PROMINENT 3D CORONARY ARTERY TREES (LAD, LCX, RCA)
  // =========================================================================
  buildThickCoronaryTree() {
    // Left Main Coronary Artery (LMCA) trunk
    const lmCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.02, 0.75, 0.38),  // Left aortic sinus ostium
      new THREE.Vector3(-0.08, 0.68, 0.68)  // Bifurcation into LAD and LCX
    ]);
    const lmGeo = new THREE.TubeGeometry(lmCurve, 14, 0.088, 14, false);
    const lmMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      roughness: 0.22,
      metalness: 0.35,
      emissive: 0x0284c7,
      emissiveIntensity: 0.4
    });
    const lmMesh = new THREE.Mesh(lmGeo, lmMat);
    lmMesh.renderOrder = 10;
    this.coronaryGroup.add(lmMesh);

    // -----------------------------------------------------------------------
    // LAD: Left Anterior Descending Artery (Substantially Thicker, Radius 0.082)
    // Follows the anterior interventricular groove straight to the apical notch
    // Surface coordinates calibrated to sit prominently on epicardium
    // -----------------------------------------------------------------------
    const ladCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.08, 0.68, 0.68), // LMCA Bifurcation
      new THREE.Vector3(-0.16, 0.40, 0.98), // Proximal LAD (critical stenosis lesion site)
      new THREE.Vector3(-0.24, 0.05, 1.08), // Mid LAD
      new THREE.Vector3(-0.28, -0.38, 0.98),// Distal LAD
      new THREE.Vector3(-0.32, -0.85, 0.76),// Pre-apex
      new THREE.Vector3(-0.36, -1.35, 0.38) // True Cardiac Apex
    ]);
    this.vesselCurves["LAD"] = ladCurve;

    // Diagonal Branch 1 (D1) — Prominent lateral LV branch
    const d1Curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.16, 0.32, 0.98),
      new THREE.Vector3(-0.48, 0.15, 0.96),
      new THREE.Vector3(-0.82, -0.12, 0.72)
    ]);
    // Diagonal Branch 2 (D2) — Mid-apical lateral branch
    const d2Curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.25, -0.18, 1.05),
      new THREE.Vector3(-0.50, -0.42, 0.88),
      new THREE.Vector3(-0.72, -0.72, 0.60)
    ]);

    // -----------------------------------------------------------------------
    // LCX: Left Circumflex Artery (Radius 0.078)
    // Traverses left atrioventricular groove (coronary sulcus) toward posterior LV
    // Surface coordinates calibrated to hug lateral epicardium
    // -----------------------------------------------------------------------
    const lcxCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.08, 0.68, 0.68), // LMCA Bifurcation
      new THREE.Vector3(-0.50, 0.62, 0.58), // Proximal LCX (under left auricle)
      new THREE.Vector3(-0.95, 0.40, 0.22), // Mid LCX (wrapping left lateral border)
      new THREE.Vector3(-1.12, 0.08, -0.22),// Distal LCX
      new THREE.Vector3(-0.88, -0.38, -0.58)// Posterior atrioventricular course
    ]);
    this.vesselCurves["LCX"] = lcxCurve;

    // Obtuse Marginal 1 (OM1) — Thick branch down obtuse lateral margin
    const om1Curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.95, 0.35, 0.26),
      new THREE.Vector3(-1.08, 0.05, 0.32),
      new THREE.Vector3(-1.10, -0.42, 0.22)
    ]);
    // Obtuse Marginal 2 (OM2) — Posterolateral branch
    const om2Curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-1.12, 0.08, -0.18),
      new THREE.Vector3(-1.16, -0.25, -0.15)
    ]);

    // -----------------------------------------------------------------------
    // RCA: Right Coronary Artery (Radius 0.082)
    // Emerges from right coronary sinus, courses down right AV sulcus
    // Surface coordinates calibrated to sit on right AV border and acute margin
    // -----------------------------------------------------------------------
    const rcaCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.25, 0.70, 0.52),  // Right aortic sinus ostium
      new THREE.Vector3(0.68, 0.52, 0.70),  // Proximal RCA (Right AV Groove)
      new THREE.Vector3(0.98, 0.15, 0.65),  // Mid RCA (Acute margin)
      new THREE.Vector3(0.96, -0.32, 0.45), // Distal RCA
      new THREE.Vector3(0.72, -0.80, 0.05), // Crux cordis
      new THREE.Vector3(0.28, -1.22, -0.15) // Posterior descending artery (PDA)
    ]);
    this.vesselCurves["RCA"] = rcaCurve;

    // Acute Marginal (AM) Branch — Courses over anterior RV free wall
    const amCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.96, 0.10, 0.65),
      new THREE.Vector3(0.85, -0.18, 0.82),
      new THREE.Vector3(0.62, -0.52, 0.85)
    ]);

    // Create Rendered Thick Arterial Meshes
    this.createThickVesselMesh("LAD", ladCurve, 0.082, [d1Curve, d2Curve]);
    this.createThickVesselMesh("LCX", lcxCurve, 0.078, [om1Curve, om2Curve]);
    this.createThickVesselMesh("RCA", rcaCurve, 0.082, [amCurve]);
  }

  createThickVesselMesh(targetKey, mainCurve, radius, sideCurves = []) {
    const vesselGroup = new THREE.Group();
    vesselGroup.name = targetKey;

    // Clinical arterial material with dynamic emissive highlight
    const mat = new THREE.MeshStandardMaterial({
      color: 0x10b981,        // Default patent Emerald (#10b981)
      roughness: 0.18,
      metalness: 0.35,
      emissive: 0x10b981,
      emissiveIntensity: 0.65
    });

    // Main prominent tube geometry
    const mainGeo = new THREE.TubeGeometry(mainCurve, 54, radius, 16, false);
    const mainMesh = new THREE.Mesh(mainGeo, mat);
    mainMesh.castShadow = true;
    mainMesh.renderOrder = 10;
    mainMesh.userData = { target: targetKey, isVessel: true };
    vesselGroup.add(mainMesh);

    // Prominent branches
    sideCurves.forEach((c) => {
      const bGeo = new THREE.TubeGeometry(c, 28, radius * 0.72, 12, false);
      const bMesh = new THREE.Mesh(bGeo, mat);
      bMesh.renderOrder = 10;
      bMesh.userData = { target: targetKey, isVessel: true };
      vesselGroup.add(bMesh);
    });

    this.vessels[targetKey] = {
      group: vesselGroup,
      material: mat,
      radius: radius
    };
    this.coronaryGroup.add(vesselGroup);

    // 3. Create 3D Stenosis Constriction Plaque Cuff (pulsing when risk is elevated)
    this.createPlaqueRing(targetKey, mainCurve);
  }

  createPlaqueRing(targetKey, curve) {
    // Critical proximal lesion sites (e.g. 24-30% along proximal course)
    const t = targetKey === "LAD" ? 0.24 : (targetKey === "LCX" ? 0.30 : 0.26);
    const point = curve.getPointAt(t);
    const tangent = curve.getTangentAt(t);

    const cuff = createStenosisCuff(THREE, point, tangent, 0.5);
    cuff.renderOrder = 15;
    cuff.visible = false; // toggled on when stenosis risk >= threshold
    cuff.userData = {
      target: targetKey,
      updateSeverity: (prob) => {
        const isSevere = prob >= 0.5;
        if (cuff.children[0] && cuff.children[0].material) {
          cuff.children[0].material.color.setHex(isSevere ? 0xdc2626 : 0xf59e0b);
          cuff.children[0].material.emissive.setHex(isSevere ? 0x991b1b : 0xb45309);
        }
        if (cuff.children[1] && cuff.children[1].material) {
          cuff.children[1].material.color.setHex(isSevere ? 0xef4444 : 0xfbbf24);
        }
      }
    };

    this.plaqueGroup.add(cuff);
    this.plaqueMeshes[targetKey] = cuff;
  }

  // =========================================================================
  // 5. MYOCARDIAL PERFUSION TERRITORIES & RWMA HIGHLIGHT PATCHES
  // =========================================================================
  buildPerfusionTerritories() {
    // Perfusion patches: subtle epicardial glow that never obscures solid heart or vessels
    // 1. Anterior Territory (Under LAD)
    const antGeo = new THREE.SphereGeometry(1.05, 24, 20, 0, Math.PI * 0.55, 0, Math.PI * 0.65);
    antGeo.scale(0.88, 1.35, 0.92);
    const antMat = new THREE.MeshBasicMaterial({
      color: 0x06b6d4,
      transparent: true,
      opacity: 0.0,
      depthWrite: false,
      side: THREE.FrontSide
    });
    const antMesh = new THREE.Mesh(antGeo, antMat);
    antMesh.position.set(-0.25, -0.38, 0.10);
    antMesh.rotation.z = 0.14;
    antMesh.renderOrder = 2;
    this.perfusionGroup.add(antMesh);
    this.perfusionMeshes["LAD"] = antMesh;

    // 2. Lateral/Posterior Territory (Under LCX)
    const latGeo = new THREE.SphereGeometry(1.05, 24, 20, Math.PI * 0.55, Math.PI * 0.55, 0, Math.PI * 0.65);
    latGeo.scale(0.88, 1.35, 0.92);
    const latMat = new THREE.MeshBasicMaterial({
      color: 0xa855f7,
      transparent: true,
      opacity: 0.0,
      depthWrite: false,
      side: THREE.FrontSide
    });
    const latMesh = new THREE.Mesh(latGeo, latMat);
    latMesh.position.set(-0.25, -0.38, 0.10);
    latMesh.rotation.z = 0.14;
    latMesh.renderOrder = 2;
    this.perfusionGroup.add(latMesh);
    this.perfusionMeshes["LCX"] = latMesh;

    // 3. Inferior Territory (Under RCA)
    const infGeo = new THREE.SphereGeometry(0.92, 24, 20, Math.PI * 0.85, Math.PI * 0.65, Math.PI * 0.3, Math.PI * 0.55);
    infGeo.scale(0.94, 1.18, 0.76);
    const infMat = new THREE.MeshBasicMaterial({
      color: 0xf97316,
      transparent: true,
      opacity: 0.0,
      depthWrite: false,
      side: THREE.FrontSide
    });
    const infMesh = new THREE.Mesh(infGeo, infMat);
    infMesh.position.set(0.65, -0.28, 0.25);
    infMesh.rotation.z = -0.15;
    infMesh.renderOrder = 2;
    this.perfusionGroup.add(infMesh);
    this.perfusionMeshes["RCA"] = infMesh;
  }

  // =========================================================================
  // 6. ANIMATED BLOOD FLOW PARTICLES
  // =========================================================================
  buildBloodFlowParticles() {
    const particleGeo = new THREE.SphereGeometry(0.026, 8, 8);
    const particleMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.90
    });

    ["LAD", "LCX", "RCA"].forEach((key) => {
      const curve = this.vesselCurves[key];
      const count = 22;
      for (let i = 0; i < count; i++) {
        const pMesh = new THREE.Mesh(particleGeo, particleMat.clone());
        const u = i / count;
        pMesh.position.copy(curve.getPointAt(u));
        this.particlesGroup.add(pMesh);

        this.flowParticles.push({
          mesh: pMesh,
          curve: curve,
          target: key,
          u: u,
          baseSpeed: 0.22, // Faster, crisp baseline flow
          currentSpeed: 0.22
        });
      }
    });
  }

  // =========================================================================
  // DYNAMIC RISK & PHYSIOLOGY UPDATES
  // =========================================================================
  updateRisks(predictions, threshold = 0.50) {
    this.triageThreshold = threshold;

    ["LAD", "LCX", "RCA"].forEach((key) => {
      const pred = predictions[key];
      if (!pred) return;

      const prob = pred.probability;
      const vessel = this.vessels[key];
      const plaque = this.plaqueMeshes[key];
      const perfusion = this.perfusionMeshes[key];

      // Clinical color coding: Emerald (<25%), Amber (25-49%), Crimson (>=50% or >= threshold)
      let colorHex = 0x10b981; // Emerald
      let emissiveIntensity = 0.35;

      if (prob >= threshold || prob >= 0.50) {
        colorHex = 0xef4444; // Crimson
        emissiveIntensity = 0.85;
      } else if (prob >= 0.25) {
        colorHex = 0xf59e0b; // Amber
        emissiveIntensity = 0.55;
      }

      if (vessel) {
        vessel.material.color.setHex(colorHex);
        vessel.material.emissive.setHex(colorHex);
        vessel.material.emissiveIntensity = emissiveIntensity;
      }

      // Plaque constriction cuff visibility & severity color
      if (plaque) {
        plaque.visible = prob >= threshold;
        if (plaque.userData && typeof plaque.userData.updateSeverity === "function") {
          plaque.userData.updateSeverity(prob);
        }
      }

      // Flow particle speed (slows down proportionally as stenosis probability increases)
      this.flowParticles.filter(p => p.target === key).forEach(p => {
        const speedFactor = Math.max(0.15, 1.0 - (prob * 0.80));
        p.currentSpeed = p.baseSpeed * speedFactor;
        p.mesh.material.color.setHex(colorHex);
      });

      // Perfusion territory patch highlight
      if (perfusion) {
        if (prob >= threshold || this.rwma > 0) {
          perfusion.material.color.setHex(colorHex);
          perfusion.material.opacity = Math.min(0.16, 0.04 + prob * 0.12);
        } else {
          perfusion.material.opacity = 0.0;
        }
      }
    });
  }

  updatePhysiology(pulseRate, ejectionFraction, rwma = 0) {
    this.pulseRate = Math.max(40, Math.min(180, Number(pulseRate) || 75));
    this.ejectionFraction = Math.max(15, Math.min(80, Number(ejectionFraction) || 50));
    this.rwma = Number(rwma) || 0;
  }

  selectVessel(targetKey) {
    this.selectedVessel = targetKey;

    ["LAD", "LCX", "RCA"].forEach((key) => {
      const v = this.vessels[key];
      if (!v) return;

      if (!targetKey || targetKey === "Cath" || targetKey === key) {
        v.group.scale.set(1, 1, 1);
        if (key === targetKey) {
          v.material.emissiveIntensity = 1.0;
        } else {
          v.material.emissiveIntensity = 0.45;
        }
      } else {
        v.material.emissiveIntensity = 0.18;
      }
    });
  }

  setCameraView(viewName) {
    if (!this.camera || !this.controls) return;

    const key = (viewName || "").toUpperCase();
    const config = CATH_VIEWS[key] || (key === "RESET" ? CATH_VIEWS.AP : null);
    if (config) {
      this.camera.position.set(...config.pos);
      this.controls.target.set(...config.target);
      this.controls.update();
    }
  }

  setLayerVisibility(layer, isVisible) {
    if (layer === "chambers") {
      this.ventriclesGroup.visible = isVisible;
      this.atriaGroup.visible = isVisible;
      this.greatVesselsGroup.visible = isVisible;
    } else if (layer === "perfusion") {
      this.perfusionGroup.visible = isVisible;
    } else if (layer === "flow") {
      this.particlesGroup.visible = isVisible;
    } else if (layer === "beat") {
      this.enableBeating = isVisible;
    }
  }

  // =========================================================================
  // INTERACTION & RAYCASTING
  // =========================================================================
  onPointerDown(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.coronaryGroup.children, true);

    if (intersects.length > 0) {
      let hit = intersects[0].object;
      while (hit && !hit.userData.target && hit.parent) {
        hit = hit.parent;
      }
      if (hit && hit.userData.target) {
        const target = hit.userData.target;
        this.selectVessel(target);
        if (typeof this.onVesselClicked === "function") {
          this.onVesselClicked(target);
        }
      }
    }
  }

  onWindowResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  // =========================================================================
  // MAIN RENDER & NATURAL CARDIAC CYCLE CONTRACTION
  // =========================================================================
  animate() {
    requestAnimationFrame(() => this.animate());

    const delta = this.clock.getDelta();
    this.timeAccumulator += delta;

    // 1. Controls update
    if (this.controls) {
      this.controls.update();
    }

    // 2. Cardiac Cycle Contraction synced to PR (BPM) & EF-TTE (%)
    if (this.enableBeating) {
      const beatFrequency = (this.pulseRate / 60) * Math.PI * 2;
      const systolePhase = Math.sin(this.timeAccumulator * beatFrequency);

      // Anatomical ventricular contraction:
      // The ventricular body contracts radially and pulls towards apex during systole.
      // Amplitude scales with EF-TTE (normal 60% EF = ~7-9% radial excursion; depressed 30% = hypokinetic ~2-3%)
      const strokeAmplitude = (this.ejectionFraction / 100) * 0.085;
      const radialScale = 1.0 - strokeAmplitude * Math.max(0, systolePhase);
      const longScale = 1.0 - strokeAmplitude * 0.45 * Math.max(0, systolePhase);

      this.ventriclesGroup.scale.set(radialScale, longScale, radialScale);
    } else {
      this.ventriclesGroup.scale.set(1, 1, 1);
    }

    // 3. Animate Blood Flow Particles along Arterial Curves
    if (this.particlesGroup.visible) {
      this.flowParticles.forEach((p) => {
        p.u = (p.u + p.currentSpeed * delta) % 1.0;
        const pos = p.curve.getPointAt(p.u);
        p.mesh.position.copy(pos);
      });
    }

    // 4. Subtle Plaque Ring Pulsing Glow
    const pulseFactor = 1.0 + Math.sin(this.timeAccumulator * 4.5) * 0.16;
    Object.values(this.plaqueMeshes).forEach((pm) => {
      if (pm.visible) {
        pm.scale.set(pulseFactor, pulseFactor, pulseFactor);
      }
    });

    this.renderer.render(this.scene, this.camera);
  }
}

// Attach globally
window.Heart3DViewer = Heart3DViewer;
