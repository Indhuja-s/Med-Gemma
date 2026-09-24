/**
 * MedGemma AI — Enterprise Interactive Radiology Workstation
 * Single-Accordion Router Engine · DICOM Viewer · AI Chat · Controls
 */

document.addEventListener("DOMContentLoaded", () => {
    "use strict";

    // ──────────────────────────────────────────────────────────────────────────
    // 1. STATE
    // ──────────────────────────────────────────────────────────────────────────
    const state = {
        sidebarState: localStorage.getItem("medgemma_sidebar_state") || "expanded",
        theme: localStorage.getItem("medgemma_theme") || "dark",
        activeAccordion: localStorage.getItem("medgemma_active_accordion") || "dashboard",
        activePage: "dashboard",
        currentPatientId: "BraTS20_Training_001",
        patientName: "Emily Jackson",
        shapeX: 240,
        shapeY: 240,
        shapeZ: 155,
        axialSliceIndex: 78,
        coronalSliceIndex: 120,
        sagittalSliceIndex: 120,
        totalSlices: 155,
        brightness: 100,
        contrast: 100,
        opacity: 80,
        showOverlay: true,
        showOutline: true,
        isPlaying: false,
        playInterval: null,
        origImageObj: new Image(),
        coronalImageObj: new Image(),
        sagittalImageObj: new Image(),
        maskImageObj: new Image(),
        coronalMaskImageObj: new Image(),
        sagittalMaskImageObj: new Image(),
        touchStartX: 0
    };

    // Apply stored theme on load
    document.documentElement.setAttribute("data-theme", state.theme);

    // ──────────────────────────────────────────────────────────────────────────
    // 2. DOM REFS
    // ──────────────────────────────────────────────────────────────────────────
    const $ = (id) => document.getElementById(id);
    const appLayout     = $("appLayout");
    const sidebar       = $("sidebar");
    const sidebarToggleBtn = $("sidebarToggleBtn");
    const sidebarSearchInput = $("sidebarSearchInput");
    const profileCard   = $("profileCard");
    const profileDropdown = $("profileDropdown");
    const mobileHamburgerBtn = $("mobileHamburgerBtn");
    const mobileBackdrop = $("mobileBackdrop");
    const themeToggleBtn = $("themeToggleDropdownBtn");
    const toast         = $("toast");
    const clockText     = $("clockText");
    const wsCurrentName = $("wsCurrentName");
    const wsCurrentScan = $("wsCurrentScan");
    const wsCurrentPatient = $("wsCurrentPatient");
    const wsCurrentStatus  = $("wsCurrentStatus");
    const bcCategory    = $("bcCategory");
    const bcDetail      = $("bcDetail");
    const contextTitle  = $("contextTitle");
    const contextBody   = $("contextBody");
    const contextIcon   = $("contextIcon");

    // ──────────────────────────────────────────────────────────────────────────
    // 3. TOAST & CLOCK
    // ──────────────────────────────────────────────────────────────────────────
    const showToast = (msg, type = "info") => {
        if (!toast) return;
        const colors = { info: "#06B6D4", success: "#10B981", error: "#EF4444", warning: "#F59E0B" };
        toast.textContent = msg;
        toast.style.background = colors[type] || colors.info;
        toast.style.color = type === "warning" ? "#000" : "#fff";
        toast.classList.add("show");
        clearTimeout(toast._timer);
        toast._timer = setTimeout(() => toast.classList.remove("show"), 3000);
    };
    window.showToast = showToast;

    const updateClock = () => {
        if (!clockText) return;
        const now = new Date();
        const pad = (n) => String(n).padStart(2, "0");
        clockText.textContent = `${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())} UTC`;
    };
    setInterval(updateClock, 1000);
    updateClock();

    // ──────────────────────────────────────────────────────────────────────────
    // 4. WORKSPACE MAP
    // ──────────────────────────────────────────────────────────────────────────
    const workspaceMap = {
        "dashboard": { title: "Dashboard Overview", icon: "fa-chart-simple", bc: "Dashboard", status: "System Normal", cardTitle: "DASHBOARD SUMMARY", cardHTML: `<div class="mini-stat"><span>Today's Cases</span><strong class="text-cyan">12</strong></div><div class="mini-stat"><span>Pending Reviews</span><strong class="text-amber">4</strong></div><div class="mini-stat"><span>Critical Alerts</span><strong class="text-red">2</strong></div>` },
        "workstation": { title: "MRI Workspace", icon: "fa-brain", bc: "MRI Workspace", status: "Ready for Analysis", cardTitle: "MRI WORKSPACE", cardHTML: () => `<div class="mini-stat"><span>Current Scan</span><strong class="mono text-cyan">${state.currentPatientId}</strong></div><div class="mini-stat"><span>Slices</span><strong class="mono text-cyan">${state.totalSlices}</strong></div><div class="mini-stat"><span>AI Confidence</span><strong class="text-green">98.7%</strong></div>` },
        "patients": { title: "Patients Directory", icon: "fa-users", bc: "Patients", status: "Directory Active", cardTitle: "PATIENTS METRICS", cardHTML: `<div class="mini-stat"><span>Total Patients</span><strong class="text-cyan">369</strong></div><div class="mini-stat"><span>Critical</span><strong class="text-red">8</strong></div><div class="mini-stat"><span>High Risk</span><strong class="text-amber">17</strong></div>` },
        "upload": { title: "Upload MRI Scan", icon: "fa-cloud-arrow-up", bc: "Upload MRI", status: "Ingestion Ready", cardTitle: "UPLOAD QUEUE", cardHTML: `<div class="mini-stat"><span>Upload Queue</span><strong class="text-cyan">2 Scans</strong></div><div class="mini-stat"><span>Transfer Speed</span><strong class="mono text-green">45 MB/s</strong></div><div class="mini-stat"><span>Status</span><strong class="text-green">Active</strong></div>` },
        "analysis": { title: "AI Diagnostic Analysis", icon: "fa-robot", bc: "AI Analysis", status: "Inference Ready", cardTitle: "AI DIAGNOSTICS", cardHTML: `<div class="mini-stat"><span>Model</span><strong class="text-purple">MedGemma v2.4</strong></div><div class="mini-stat"><span>Confidence</span><strong class="text-green">98.7%</strong></div><div class="mini-stat"><span>Severity</span><strong class="text-red">WHO Grade IV</strong></div>` },
        "segmentation": { title: "3D Volumetric Segmentation", icon: "fa-bullseye", bc: "Segmentation", status: "3D U-Net Active", cardTitle: "SEGMENTATION STATS", cardHTML: `<div class="mini-stat"><span>Dice Score</span><strong class="mono text-green">0.932</strong></div><div class="mini-stat"><span>Tumor Volume</span><strong class="text-purple">34.2 cm³</strong></div><div class="mini-stat"><span>Overlay</span><strong class="text-cyan">Active</strong></div>` },
        "comparison": { title: "Scan Comparison Engine", icon: "fa-sliders", bc: "Comparison", status: "Side-by-Side Ready", cardTitle: "COMPARISON MODE", cardHTML: `<div class="mini-stat"><span>Tumor Growth</span><strong class="text-red">+10.3%</strong></div><div class="mini-stat"><span>Volume Delta</span><strong class="text-amber">+3.2 cm³</strong></div><div class="mini-stat"><span>Time Delta</span><strong class="text-cyan">75 days</strong></div>` },
        "reports": { title: "Radiology Reports Center", icon: "fa-chart-column", bc: "Reports", status: "Hub Active", cardTitle: "REPORTS SUMMARY", cardHTML: `<div class="mini-stat"><span>Generated Today</span><strong class="text-cyan">14</strong></div><div class="mini-stat"><span>Pending</span><strong class="text-amber">3</strong></div><div class="mini-stat"><span>Drafts</span><strong class="text-purple">1</strong></div>` },
        "history": { title: "Audit Log & History", icon: "fa-clock-rotate-left", bc: "History", status: "Logs Active", cardTitle: "AUDIT LOG STATS", cardHTML: `<div class="mini-stat"><span>Scan Logs</span><strong class="mono text-cyan">1,280</strong></div><div class="mini-stat"><span>Audit Trail</span><strong class="text-green">Active</strong></div><div class="mini-stat"><span>Last Activity</span><strong class="text-muted">10m ago</strong></div>` },
        "settings": { title: "System Settings", icon: "fa-gear", bc: "Settings", status: "Config Active", cardTitle: "SYSTEM CONFIG", cardHTML: `<div class="mini-stat"><span>Version</span><strong class="mono text-purple">v2.4.1</strong></div><div class="mini-stat"><span>License</span><strong class="text-cyan">Enterprise</strong></div><div class="mini-stat"><span>GPU Engine</span><strong class="text-green">TensorRT</strong></div>` }
    };

    function openMobileMenu() { appLayout.classList.add("mobile-open"); mobileBackdrop?.classList.add("show"); }
    function closeMobileMenu() { appLayout.classList.remove("mobile-open"); mobileBackdrop?.classList.remove("show"); }

    function navigateToPage(pageId, pushHistory = true) {
        if (!workspaceMap[pageId]) pageId = "dashboard";
        state.activePage = pageId;
        state.activeAccordion = pageId;
        localStorage.setItem("medgemma_active_accordion", pageId);

        const info = workspaceMap[pageId];
        const htmlContent = typeof info.cardHTML === "function" ? info.cardHTML() : info.cardHTML;

        document.querySelectorAll(".nav-accordion-item").forEach(item => {
            item.classList.toggle("active", item.getAttribute("data-page") === pageId);
        });

        if (wsCurrentName) wsCurrentName.textContent = info.title;
        if (wsCurrentScan) wsCurrentScan.textContent = `Current Scan: ${state.currentPatientId}`;
        if (wsCurrentPatient) wsCurrentPatient.textContent = `Patient: ${state.patientName}`;
        if (wsCurrentStatus) wsCurrentStatus.textContent = `Status: ${info.status}`;

        if (bcCategory) bcCategory.textContent = info.bc;
        if (bcDetail) bcDetail.textContent = `${state.patientName} (${state.currentPatientId})`;

        if (contextTitle) contextTitle.textContent = info.cardTitle;
        if (contextBody) contextBody.innerHTML = htmlContent;
        if (contextIcon) contextIcon.className = `fa-solid ${info.icon} text-cyan`;

        document.querySelectorAll(".page-view").forEach(view => {
            view.classList.toggle("active", view.id === `page-${pageId}`);
        });

        if (pushHistory) window.location.hash = pageId;

        if (["workstation", "segmentation", "comparison"].includes(pageId)) {
            setTimeout(renderAllCanvases, 80);
        }

        closeMobileMenu();
    }

    // Hash routing
    window.addEventListener("hashchange", () => {
        const hash = window.location.hash.replace("#", "");
        if (hash && workspaceMap[hash]) navigateToPage(hash, false);
    });

    (() => {
        const hash = window.location.hash.replace("#", "");
        const target = (hash && workspaceMap[hash]) ? hash : "dashboard";
        navigateToPage(target, false);
    })();

    // Accordion Nav Listeners
    document.querySelectorAll(".nav-item-header[data-page-trigger]").forEach(btn => {
        btn.addEventListener("click", () => navigateToPage(btn.getAttribute("data-page-trigger")));
    });

    document.querySelectorAll(".sub-item").forEach(subBtn => {
        subBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            document.querySelectorAll(".sub-item").forEach(s => s.classList.remove("active"));
            subBtn.classList.add("active");
            const parentPage = subBtn.getAttribute("data-page");
            if (parentPage) navigateToPage(parentPage);
        });
    });

    // ──────────────────────────────────────────────────────────────────────────
    // 5. SIDEBAR COLLAPSE & THEME TOGGLE
    // ──────────────────────────────────────────────────────────────────────────
    const setSidebarState = (s) => {
        state.sidebarState = s;
        localStorage.setItem("medgemma_sidebar_state", s);
        appLayout.classList.toggle("collapsed", s === "collapsed");
    };
    setSidebarState(state.sidebarState);

    sidebarToggleBtn?.addEventListener("click", () => setSidebarState(state.sidebarState === "expanded" ? "collapsed" : "expanded"));

    const toggleTheme = (themeName) => {
        state.theme = themeName || (state.theme === "dark" ? "light" : "dark");
        localStorage.setItem("medgemma_theme", state.theme);
        document.documentElement.setAttribute("data-theme", state.theme);
        showToast(`Theme: ${state.theme.toUpperCase()} Mode`, "info");
    };

    themeToggleBtn?.addEventListener("click", () => toggleTheme());

    profileCard?.addEventListener("click", (e) => {
        e.stopPropagation();
        profileDropdown?.classList.toggle("show");
    });
    document.addEventListener("click", () => profileDropdown?.classList.remove("show"));

    mobileHamburgerBtn?.addEventListener("click", openMobileMenu);
    mobileBackdrop?.addEventListener("click", closeMobileMenu);

    // Keyboard Shortcuts
    document.addEventListener("keydown", (e) => {
        if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
        if (!e.ctrlKey && !e.metaKey) return;
        const shortcuts = { "1": "dashboard", "2": "workstation", "3": "patients", "4": "upload", "5": "analysis" };
        if (shortcuts[e.key]) { e.preventDefault(); navigateToPage(shortcuts[e.key]); }
        else if (e.key.toLowerCase() === "b") { e.preventDefault(); setSidebarState(state.sidebarState === "expanded" ? "collapsed" : "expanded"); }
        else if (e.key.toLowerCase() === "k") { e.preventDefault(); $("searchInput")?.focus(); }
    });

    // ──────────────────────────────────────────────────────────────────────────
    // 6. SPARKLINE CHARTS
    // ──────────────────────────────────────────────────────────────────────────
    const drawSparkline = (canvasId, color, data) => {
        const canvas = $(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        canvas.width = canvas.parentElement?.clientWidth || 200;
        canvas.height = 24;
        const { width: w, height: h } = canvas;
        ctx.clearRect(0, 0, w, h);
        const min = Math.min(...data), max = Math.max(...data), range = max - min || 1;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        data.forEach((v, i) => {
            const x = (i / (data.length - 1)) * w;
            const y = h - ((v - min) / range) * (h - 6) - 3;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
    };

    setTimeout(() => {
        drawSparkline("sparkPatients", "#06B6D4", [120, 140, 135, 160, 180, 210, 250, 280]);
        drawSparkline("sparkScans",   "#7C3AED", [300, 320, 310, 350, 400, 420, 480, 520]);
        drawSparkline("sparkTumors",  "#EF4444", [40, 45, 42, 50, 58, 62, 70, 85]);
        drawSparkline("sparkAccuracy","#10B981", [96, 97, 96.5, 97.8, 98.1, 98.4, 98.7, 98.7]);
    }, 200);

    // ──────────────────────────────────────────────────────────────────────────
    // 7. DICOM CANVAS RENDERER & 3D OVERLAY PIPELINE
    // ──────────────────────────────────────────────────────────────────────────
    const drawSyntheticBrainSlice = (ctx, w, h) => {
        ctx.fillStyle = "#030508";
        ctx.fillRect(0, 0, w, h);
        ctx.beginPath();
        ctx.ellipse(w / 2, h / 2, w * 0.42, h * 0.46, 0, 0, Math.PI * 2);
        ctx.fillStyle = "#1a2330"; ctx.fill();
        ctx.beginPath();
        ctx.ellipse(w / 2, h / 2 + 4, w * 0.36, h * 0.40, 0, 0, Math.PI * 2);
        ctx.fillStyle = "#2a3a4d"; ctx.fill();
    };

    const renderCanvasView = (canvas, imgObj, maskObj = null, showSeg = false) => {
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        canvas.width = canvas.parentElement?.clientWidth || 240;
        canvas.height = canvas.parentElement?.clientHeight || 180;
        const { width: w, height: h } = canvas;

        ctx.save();
        ctx.clearRect(0, 0, w, h);
        ctx.filter = `brightness(${state.brightness}%) contrast(${state.contrast}%)`;

        if (imgObj?.complete && imgObj.naturalWidth) {
            ctx.drawImage(imgObj, 0, 0, w, h);
        } else {
            drawSyntheticBrainSlice(ctx, w, h);
        }
        ctx.restore();

        if (showSeg && state.showOverlay) {
            if (maskObj?.complete && maskObj.naturalWidth) {
                ctx.save();
                ctx.globalAlpha = state.opacity / 100;
                ctx.drawImage(maskObj, 0, 0, w, h);
                ctx.restore();
            }
            if (state.showOutline) {
                ctx.save();
                ctx.beginPath();
                ctx.arc(w / 2 - 18, h / 2 + 18, 28, 0, Math.PI * 2);
                ctx.strokeStyle = "rgba(6,182,212,0.9)";
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 3]);
                ctx.stroke();
                ctx.restore();
            }
        }
    };

    const renderAllCanvases = () => {
        renderCanvasView($("canvasAxial"),        state.origImageObj,    state.maskImageObj,         false);
        renderCanvasView($("canvasCoronal"),       state.coronalImageObj, state.coronalMaskImageObj,  false);
        renderCanvasView($("canvasSagittal"),      state.sagittalImageObj,state.sagittalMaskImageObj, false);
        renderCanvasView($("canvasSegmentation"),  state.origImageObj,    state.maskImageObj,         true);
    };

    window.addEventListener("resize", () => {
        if (["workstation", "segmentation", "comparison"].includes(state.activePage)) {
            renderAllCanvases();
        }
    });

    // ──────────────────────────────────────────────────────────────────────────
    // 8. PATIENT SCAN & ORTHOGONAL SLICE API
    // ──────────────────────────────────────────────────────────────────────────
    const loadSliceData = (patientId, sliceIdx, orientation = "axial") => {
        fetch("/load_slice", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ patient_id: patientId, slice_idx: sliceIdx, orientation: orientation })
        })
        .then(r => r.json())
        .then(data => {
            if (!data.success) return;
            if (orientation === "axial") {
                state.axialSliceIndex = data.active_slice;
                if (data.original_img) state.origImageObj.src = "data:image/png;base64," + data.original_img;
                if (data.mask_img) state.maskImageObj.src = "data:image/png;base64," + data.mask_img;
            } else if (orientation === "coronal") {
                state.coronalSliceIndex = data.active_slice;
                if (data.original_img) state.coronalImageObj.src = "data:image/png;base64," + data.original_img;
                if (data.mask_img) state.coronalMaskImageObj.src = "data:image/png;base64," + data.mask_img;
            } else if (orientation === "sagittal") {
                state.sagittalSliceIndex = data.active_slice;
                if (data.original_img) state.sagittalImageObj.src = "data:image/png;base64," + data.original_img;
                if (data.mask_img) state.sagittalMaskImageObj.src = "data:image/png;base64," + data.mask_img;
            }
            renderAllCanvases();
        });
    };

    const loadPatientScan = (patientId, sliceIdx = null) => {
        const payload = { patient_id: patientId };
        if (sliceIdx !== null) payload.slice_idx = sliceIdx;

        fetch("/load_patient", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(r => r.json())
        .then(data => {
            if (!data.success) return;

            state.shapeX = data.shape_x || 240;
            state.shapeY = data.shape_y || 240;
            state.shapeZ = data.shape_z || 155;

            state.axialSliceIndex   = data.active_slice_axial ?? data.active_slice ?? 78;
            state.coronalSliceIndex = data.active_slice_coronal ?? 120;
            state.sagittalSliceIndex= data.active_slice_sagittal ?? 120;
            state.totalSlices       = state.shapeZ;

            state.patientName = data.patient?.name || `Patient (${patientId})`;
            state.currentPatientId = data.patient?.id || patientId;

            if (patientSelect) patientSelect.value = state.currentPatientId;

            const setTxt = (id, val) => { const el = $(id); if (el) el.textContent = val; };
            setTxt("metaName", state.patientName);
            setTxt("metaId", `${state.currentPatientId}`);
            setTxt("metaAgeGender", `${data.patient?.age || 54} yrs • ${data.patient?.gender || 'Female'}`);
            setTxt("neoplasmVolumeTag", `Neoplasm Core: ${data.volume || 34.2} cm³`);

            // Configure slider max ranges dynamically for 3D shape
            const sAx = $("sliderAxial"), sCor = $("sliderCoronal"), sSag = $("sliderSagittal");
            if (sAx) { sAx.max = state.shapeZ - 1; sAx.value = state.axialSliceIndex; }
            if (sCor) { sCor.max = state.shapeY - 1; sCor.value = state.coronalSliceIndex; }
            if (sSag) { sSag.max = state.shapeX - 1; sSag.value = state.sagittalSliceIndex; }

            setTxt("axialTag", `Slice ${state.axialSliceIndex}/${state.shapeZ - 1}`);
            setTxt("coronalTag", `Slice ${state.coronalSliceIndex}/${state.shapeY - 1}`);
            setTxt("sagittalTag", `Slice ${state.sagittalSliceIndex}/${state.shapeX - 1}`);

            if (wsCurrentScan) wsCurrentScan.textContent = `Current Scan: ${state.currentPatientId}`;
            if (wsCurrentPatient) wsCurrentPatient.textContent = `Patient: ${state.patientName}`;
            if (bcDetail) bcDetail.textContent = `${state.patientName} (${state.currentPatientId})`;

            const risk = data.status || "Critical";
            const riskEl = $("riskBadge");
            if (riskEl) {
                riskEl.textContent = risk.toUpperCase();
                riskEl.className = `badge ${risk === "Critical" ? "red" : risk === "High Risk" ? "yellow" : "green"}-badge`;
            }

            if (data.original_img) state.origImageObj.src = "data:image/png;base64," + data.original_img;
            if (data.coronal_img) state.coronalImageObj.src = "data:image/png;base64," + data.coronal_img;
            if (data.sagittal_img) state.sagittalImageObj.src = "data:image/png;base64," + data.sagittal_img;

            if (data.mask_img) state.maskImageObj.src = "data:image/png;base64," + data.mask_img;
            if (data.coronal_mask_img) state.coronalMaskImageObj.src = "data:image/png;base64," + data.coronal_mask_img;
            if (data.sagittal_mask_img) state.sagittalMaskImageObj.src = "data:image/png;base64," + data.sagittal_mask_img;

            state.origImageObj.onload = renderAllCanvases;
            renderAllCanvases();
            showToast(`Loaded MRI Scan: ${state.currentPatientId}`, "success");
        })
        .catch(() => {
            renderAllCanvases();
        });
    };

    const loadPatientBtn = $("loadPatientBtn");
    const patientSelect  = $("patientSelect");
    if (patientSelect) {
        patientSelect.addEventListener("change", () => {
            state.currentPatientId = patientSelect.value;
            loadPatientScan(state.currentPatientId);
        });
    }
    if (loadPatientBtn && patientSelect) {
        loadPatientBtn.addEventListener("click", () => {
            state.currentPatientId = patientSelect.value;
            loadPatientScan(state.currentPatientId);
        });
    }

    loadPatientScan(state.currentPatientId);

    // ──────────────────────────────────────────────────────────────────────────
    // 9. VIEWER CONTROLS & SLIDERS
    // ──────────────────────────────────────────────────────────────────────────
    const brightnessRange = $("brightnessRange");
    const contrastRange   = $("contrastRange");
    const opacityRange    = $("opacityRange");
    const bVal = $("bVal"), cVal = $("cVal");

    brightnessRange?.addEventListener("input", () => {
        state.brightness = brightnessRange.value;
        if (bVal) bVal.textContent = `${state.brightness}%`;
        renderAllCanvases();
    });

    contrastRange?.addEventListener("input", () => {
        state.contrast = contrastRange.value;
        if (cVal) cVal.textContent = `${state.contrast}%`;
        renderAllCanvases();
    });

    opacityRange?.addEventListener("input", () => {
        state.opacity = opacityRange.value;
        renderAllCanvases();
    });

    $("toggleOverlay")?.addEventListener("change", (e) => {
        state.showOverlay = e.target.checked;
        renderAllCanvases();
    });

    $("toggleOutline")?.addEventListener("change", (e) => {
        state.showOutline = e.target.checked;
        renderAllCanvases();
    });

    // 3-Plane Slice Sliders
    $("sliderAxial")?.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        state.axialSliceIndex = val;
        const tag = $("axialTag"); if (tag) tag.textContent = `Slice ${val}/${state.shapeZ - 1}`;
        loadSliceData(state.currentPatientId, val, "axial");
    });

    $("sliderCoronal")?.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        state.coronalSliceIndex = val;
        const tag = $("coronalTag"); if (tag) tag.textContent = `Slice ${val}/${state.shapeY - 1}`;
        loadSliceData(state.currentPatientId, val, "coronal");
    });

    $("sliderSagittal")?.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        state.sagittalSliceIndex = val;
        const tag = $("sagittalTag"); if (tag) tag.textContent = `Slice ${val}/${state.shapeX - 1}`;
        loadSliceData(state.currentPatientId, val, "sagittal");
    });

    // Reset Tool
    $("toolReset")?.addEventListener("click", () => {
        state.brightness = 100; state.contrast = 100; state.opacity = 80;
        if (brightnessRange) brightnessRange.value = 100;
        if (contrastRange) contrastRange.value = 100;
        if (opacityRange) opacityRange.value = 80;
        if (bVal) bVal.textContent = "100%";
        if (cVal) cVal.textContent = "100%";
        renderAllCanvases();
        showToast("Viewer reset", "info");
    });

    // ──────────────────────────────────────────────────────────────────────────
    // 10. AI CHAT INTERFACE
    // ──────────────────────────────────────────────────────────────────────────
    const chatHistoryEl  = $("chatHistoryPage");
    const chatInputEl    = $("chatInputPage");
    const sendBtnPageEl  = $("sendBtnPage");

    const appendChatMsg = (text, isUser = false) => {
        if (!chatHistoryEl) return;
        const div = document.createElement("div");
        div.className = `c-msg ${isUser ? "user" : "bot"}`;
        div.style.cssText = `
            padding: 0.4rem 0.5rem; margin-bottom: 0.35rem; border-radius: 6px;
            background: ${isUser ? "rgba(79,141,255,0.15)" : "rgba(0,0,0,0.2)"};
            border-left: 2px solid ${isUser ? "var(--accent-blue)" : "var(--accent-cyan)"};
        `;
        div.innerHTML = `<p style="font-size:0.72rem;line-height:1.4;color:${isUser ? "var(--text-primary)" : "var(--text-secondary)"}">${text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>")}</p>`;
        chatHistoryEl.appendChild(div);
        chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
    };

    const sendChat = () => {
        if (!chatInputEl) return;
        const q = chatInputEl.value.trim();
        if (!q) return;
        appendChatMsg(q, true);
        chatInputEl.value = "";

        const typingDiv = document.createElement("div");
        typingDiv.id = "typingIndicator";
        typingDiv.style.cssText = "padding:0.3rem 0.5rem;font-size:0.7rem;color:var(--accent-cyan);font-style:italic;";
        typingDiv.textContent = "MedGemma AI is analyzing...";
        chatHistoryEl?.appendChild(typingDiv);
        chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;

        fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ patient_id: state.currentPatientId, question: q })
        })
        .then(r => r.json())
        .then(data => {
            $("typingIndicator")?.remove();
            appendChatMsg(data.response || "I was unable to process that query.", false);
        })
        .catch(() => {
            $("typingIndicator")?.remove();
            appendChatMsg("⚠️ Network error. Please check server connection.", false);
        });
    };

    sendBtnPageEl?.addEventListener("click", sendChat);
    chatInputEl?.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } });

    document.querySelectorAll(".chip-btn").forEach(chip => {
        chip.addEventListener("click", () => {
            if (chatInputEl) chatInputEl.value = chip.getAttribute("data-query") || chip.textContent.trim();
            sendChat();
        });
    });

    // ──────────────────────────────────────────────────────────────────────────
    // 11. UPLOAD SYSTEM WITH VALIDATION
    // ──────────────────────────────────────────────────────────────────────────
    const dropzone = $("dropzoneLarge");
    const fileInput = $("fileInputUpload");
    const browseBtn = $("browseUploadBtn");

    browseBtn?.addEventListener("click", () => fileInput?.click());

    // Drag & Drop Event Handlers
    if (dropzone) {
        ["dragenter", "dragover"].forEach(evtName => {
            dropzone.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });
        ["dragleave", "drop"].forEach(evtName => {
            dropzone.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });
        dropzone.addEventListener("drop", (e) => {
            const files = e.dataTransfer?.files;
            if (files && files.length > 0) {
                handleFileUpload(files[0]);
            }
        });
    }

    const handleFileUpload = (file) => {
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);
        showToast(`Uploading & processing ${file.name}...`, "info");

        if (dropzone) {
            dropzone.innerHTML = `
                <i class="fa-solid fa-spinner fa-spin fa-3x text-cyan"></i>
                <h3>Processing & Ingesting ${file.name}...</h3>
                <p>Registering file asset & generating volume preview...</p>
            `;
        }

        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                const pid = data.patient_id || file.name.replace(/\.[^/.]+$/, "");
                showToast(`File uploaded & registered: ${file.name}`, "success");

                if (patientSelect) {
                    let optExists = Array.from(patientSelect.options).some(o => o.value === pid);
                    if (!optExists) {
                        const newOpt = document.createElement("option");
                        newOpt.value = pid;
                        newOpt.textContent = `${pid} (${file.name})`;
                        patientSelect.prepend(newOpt);
                    }
                    patientSelect.value = pid;
                }
                state.currentPatientId = pid;
                loadPatientScan(pid);
                navigateToPage("workstation");
            } else {
                const errMsg = data.error?.message || "Upload failed. Invalid file format.";
                showToast("Upload Error: " + errMsg, "error");
                if (dropzone) {
                    dropzone.innerHTML = `
                        <i class="fa-solid fa-triangle-exclamation fa-3x text-red"></i>
                        <h3 class="text-red">Upload Validation Failed</h3>
                        <p>${errMsg}</p>
                        <button class="secondary-btn mt-3" id="retryUploadBtn">Try Another File</button>
                    `;
                    $("retryUploadBtn")?.addEventListener("click", () => fileInput?.click());
                }
            }
        })
        .catch(() => {
            showToast("Upload error. Check server connectivity.", "error");
        });
    };

    fileInput?.addEventListener("change", (e) => {
        if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
    });
});
