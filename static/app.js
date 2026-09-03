// =====================================================
// CANVAS & STATE
// =====================================================

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const imageObj = new Image();

let detections = [];
let visibleClasses = {};
let hoveredDetectionId = null;

let imageWidth = 0;
let imageHeight = 0;

// =====================================================
// CONSTANTS
// =====================================================

const LEFT_MARGIN = 220;
const TOP_MARGIN = 100;
const EXTRA_BOTTOM = 160;
const BADGE_RADIUS = 15;
const MIN_BADGE_GAP = 36; // 30px diameter + 6px gap

// =====================================================
// EVENT LISTENERS
// =====================================================

document.getElementById("predictBtn").addEventListener("click", uploadAndPredict);
document.getElementById("showAllBtn").addEventListener("click", showAllClasses);
document.getElementById("hideAllBtn").addEventListener("click", hideAllClasses);

// Canvas Mouse Hover Listener
canvas.addEventListener("mousemove", handleCanvasMouseMove);
canvas.addEventListener("mouseleave", handleCanvasMouseLeave);

// =====================================================
// PREDICT
// =====================================================

async function uploadAndPredict() {
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    document.getElementById("loadingText").style.display = "inline";

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned status ${response.status}`);
        }

        const data = await response.json();
        detections = data.detections || [];
        imageWidth = data.width || 0;
        imageHeight = data.height || 0;

        buildCheckboxes();

        imageObj.onload = () => {
            updateCanvasDimensions();
            drawScene();
        };

        imageObj.src = data.image_url;

    } catch (err) {
        console.error("Prediction Error:", err);
        alert("Prediction failed. Please check the console for details.");
    } finally {
        document.getElementById("loadingText").style.display = "none";
    }
}

// Dynamic Canvas Resizing
function updateCanvasDimensions() {
    const visibleDets = detections.filter(d => visibleClasses[d.class_name]);
    const maxPerCol = 16;
    const numCols = Math.max(1, Math.ceil(visibleDets.length / maxPerCol));
    const legendWidth = Math.max(320, numCols * 210 + 40);

    canvas.width = imageWidth + LEFT_MARGIN + legendWidth;
    canvas.height = Math.max(imageHeight + TOP_MARGIN + EXTRA_BOTTOM, 750);
}

// =====================================================
// CHECKBOXES & SIDEBAR
// =====================================================

function buildCheckboxes() {
    const anatomyDiv = document.getElementById("anatomyClassList");
    const dentalDiv = document.getElementById("dentalClassList");

    anatomyDiv.innerHTML = "";
    dentalDiv.innerHTML = "";

    visibleClasses = {};

    const anatomyDets = detections.filter(d => d.source === "anatomy");
    const dentalDets = detections.filter(d => d.source === "dental");

    const anatomyClasses = [...new Set(anatomyDets.map(d => d.class_name))];
    const dentalClasses = [...new Set(dentalDets.map(d => d.class_name))];

    anatomyClasses.forEach(cls => {
        const count = anatomyDets.filter(d => d.class_name === cls).length;
        addCheckbox(cls, count, anatomyDiv);
    });

    dentalClasses.forEach(cls => {
        const count = dentalDets.filter(d => d.class_name === cls).length;
        addCheckbox(cls, count, dentalDiv);
    });

    renderDetectionCards();
}

function addCheckbox(className, count, parent) {
    visibleClasses[className] = true;

    const div = document.createElement("div");
    div.className = "class-item";

    div.innerHTML = `
        <label>
            <input type="checkbox" checked data-class="${className}">
            <span class="class-label-text">${className}</span>
            <span class="class-badge-count">${count}</span>
        </label>
    `;

    parent.appendChild(div);

    div.querySelector("input").addEventListener("change", function () {
        visibleClasses[className] = this.checked;
        updateCanvasDimensions();
        drawScene();
        renderDetectionCards();
    });
}

function renderDetectionCards() {
    let cardsContainer = document.getElementById("detectionCardsList");
    if (!cardsContainer) {
        const classPanel = document.getElementById("classPanel");
        const section = document.createElement("div");
        section.className = "group-section";
        section.innerHTML = `
            <h3>Individual Detections</h3>
            <div id="detectionCardsList" class="detection-cards-container"></div>
        `;
        classPanel.appendChild(section);
        cardsContainer = document.getElementById("detectionCardsList");
    }

    cardsContainer.innerHTML = "";

    detections.forEach(det => {
        if (!visibleClasses[det.class_name]) return;

        const card = document.createElement("div");
        card.className = `detection-card ${det.id === hoveredDetectionId ? "active" : ""}`;
        card.dataset.id = det.id;

        const confPct = Math.round(det.confidence * 100);

        card.innerHTML = `
            <span class="det-id-badge" style="background-color: ${det.color}">${det.id}</span>
            <div class="det-info">
                <div class="det-title">${det.class_name}</div>
                <div class="det-meta">Conf: ${confPct}%</div>
            </div>
        `;

        card.addEventListener("mouseenter", () => {
            hoveredDetectionId = det.id;
            highlightCard(det.id);
            drawScene();
        });

        card.addEventListener("mouseleave", () => {
            hoveredDetectionId = null;
            highlightCard(null);
            drawScene();
        });

        cardsContainer.appendChild(card);
    });
}

function highlightCard(id) {
    document.querySelectorAll(".detection-card").forEach(card => {
        if (id && parseInt(card.dataset.id) === id) {
            card.classList.add("active");
            card.scrollIntoView({ behavior: "smooth", block: "nearest" });
        } else {
            card.classList.remove("active");
        }
    });
}

// =====================================================
// SHOW / HIDE ALL
// =====================================================

function showAllClasses() {
    document.querySelectorAll("input[type='checkbox']").forEach(cb => {
        cb.checked = true;
        visibleClasses[cb.dataset.class] = true;
    });
    updateCanvasDimensions();
    drawScene();
    renderDetectionCards();
}

function hideAllClasses() {
    document.querySelectorAll("input[type='checkbox']").forEach(cb => {
        cb.checked = false;
        visibleClasses[cb.dataset.class] = false;
    });
    updateCanvasDimensions();
    drawScene();
    renderDetectionCards();
}

// =====================================================
// CALLOUT POSITIONING (ANTI-COLLISION)
// =====================================================

function computeBadgePositions() {
    const visibleDets = detections.filter(d => visibleClasses[d.class_name]);

    const sides = { 0: [], 1: [], 2: [], 3: [] };
    visibleDets.forEach(d => {
        const s = (d.side !== undefined && d.side >= 0 && d.side <= 3) ? d.side : 0;
        sides[s].push(d);
    });

    // --- LEFT SIDE (0) & RIGHT SIDE (1) ---
    [0, 1].forEach(side => {
        const group = sides[side];
        if (group.length === 0) return;

        group.sort((a, b) => a.cy - b.cy);

        const yMin = TOP_MARGIN + BADGE_RADIUS;
        const yMax = TOP_MARGIN + imageHeight - BADGE_RADIUS;

        let pos = group.map(d => d.cy + TOP_MARGIN);

        // Forward relaxation
        for (let i = 1; i < pos.length; i++) {
            if (pos[i] < pos[i - 1] + MIN_BADGE_GAP) {
                pos[i] = pos[i - 1] + MIN_BADGE_GAP;
            }
        }

        // Backward relaxation
        if (pos[pos.length - 1] > yMax) {
            pos[pos.length - 1] = yMax;
            for (let i = pos.length - 2; i >= 0; i--) {
                if (pos[i] > pos[i + 1] - MIN_BADGE_GAP) {
                    pos[i] = pos[i + 1] - MIN_BADGE_GAP;
                }
            }
        }

        // Forward boundary fix
        if (pos[0] < yMin) {
            pos[0] = yMin;
            for (let i = 1; i < pos.length; i++) {
                if (pos[i] < pos[i - 1] + MIN_BADGE_GAP) {
                    pos[i] = pos[i - 1] + MIN_BADGE_GAP;
                }
            }
        }

        const badgeX = (side === 0) 
            ? LEFT_MARGIN - 65 
            : LEFT_MARGIN + imageWidth + 65;

        const elbowX = (side === 0)
            ? LEFT_MARGIN - 25
            : LEFT_MARGIN + imageWidth + 25;

        group.forEach((d, idx) => {
            d.tx = badgeX;
            d.ty = pos[idx];
            d.elbowX = elbowX;
            d.elbowY = d.cy + TOP_MARGIN;
        });
    });

    // --- TOP SIDE (2) & BOTTOM SIDE (3) ---
    [2, 3].forEach(side => {
        const group = sides[side];
        if (group.length === 0) return;

        group.sort((a, b) => a.cx - b.cx);

        const xMin = LEFT_MARGIN + BADGE_RADIUS;
        const xMax = LEFT_MARGIN + imageWidth - BADGE_RADIUS;

        let pos = group.map(d => d.cx + LEFT_MARGIN);

        // Forward relaxation
        for (let i = 1; i < pos.length; i++) {
            if (pos[i] < pos[i - 1] + MIN_BADGE_GAP) {
                pos[i] = pos[i - 1] + MIN_BADGE_GAP;
            }
        }

        // Backward relaxation
        if (pos[pos.length - 1] > xMax) {
            pos[pos.length - 1] = xMax;
            for (let i = pos.length - 2; i >= 0; i--) {
                if (pos[i] > pos[i + 1] - MIN_BADGE_GAP) {
                    pos[i] = pos[i + 1] - MIN_BADGE_GAP;
                }
            }
        }

        // Forward boundary fix
        if (pos[0] < xMin) {
            pos[0] = xMin;
            for (let i = 1; i < pos.length; i++) {
                if (pos[i] < pos[i - 1] + MIN_BADGE_GAP) {
                    pos[i] = pos[i - 1] + MIN_BADGE_GAP;
                }
            }
        }

        const badgeY = (side === 2)
            ? TOP_MARGIN - 45
            : TOP_MARGIN + imageHeight + 45;

        const elbowY = (side === 2)
            ? TOP_MARGIN - 20
            : TOP_MARGIN + imageHeight + 20;

        group.forEach((d, idx) => {
            d.tx = pos[idx];
            d.ty = badgeY;
            d.elbowX = d.cx + LEFT_MARGIN;
            d.elbowY = elbowY;
        });
    });
}

// =====================================================
// DRAW SCENE
// =====================================================

function drawScene() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (imageObj.src && imageObj.complete) {
        ctx.drawImage(imageObj, LEFT_MARGIN, TOP_MARGIN);
    }

    computeBadgePositions();

    const visibleDets = detections.filter(d => visibleClasses[d.class_name]);

    // 1. Draw polygons
    visibleDets.forEach(det => {
        drawPolygon(det);
    });

    // 2. Draw callout lines & badges
    visibleDets.forEach(det => {
        drawCallout(det);
    });

    // 3. Draw canvas legend
    drawLegend();
}

// =====================================================
// POLYGON
// =====================================================

function drawPolygon(det) {
    const poly = det.polygon;
    if (!poly || poly.length < 2) return;

    const isHovered = (det.id === hoveredDetectionId);

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(poly[0][0] + LEFT_MARGIN, poly[0][1] + TOP_MARGIN);

    for (let i = 1; i < poly.length; i++) {
        ctx.lineTo(poly[i][0] + LEFT_MARGIN, poly[i][1] + TOP_MARGIN);
    }

    ctx.closePath();

    // Fill with semi-transparent color on hover
    if (isHovered) {
        ctx.fillStyle = hexToRgba(det.color, 0.35);
        ctx.fill();
    } else {
        ctx.fillStyle = hexToRgba(det.color, 0.12);
        ctx.fill();
    }

    ctx.strokeStyle = det.color;
    ctx.lineWidth = isHovered ? 3.5 : 2;
    ctx.stroke();
    ctx.restore();
}

// =====================================================
// CALLOUTS
// =====================================================

function drawCallout(det) {
    const cx = det.cx + LEFT_MARGIN;
    const cy = det.cy + TOP_MARGIN;
    const isHovered = (det.id === hoveredDetectionId);

    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    ctx.beginPath();
    ctx.moveTo(cx, cy);

    if (det.side === 0 || det.side === 1) {
        ctx.lineTo(det.elbowX, cy);
        ctx.lineTo(det.elbowX, det.ty);
        ctx.lineTo(det.tx, det.ty);
    } else {
        ctx.lineTo(cx, det.elbowY);
        ctx.lineTo(det.tx, det.elbowY);
        ctx.lineTo(det.tx, det.ty);
    }

    ctx.strokeStyle = det.color;
    ctx.lineWidth = isHovered ? 3.5 : 2;
    ctx.stroke();

    // Centroid marker dot
    ctx.beginPath();
    ctx.arc(cx, cy, isHovered ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle = det.color;
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1;
    ctx.stroke();

    drawNumber(det.id, det.tx, det.ty, det.color, isHovered);
    ctx.restore();
}

// =====================================================
// NUMBER BADGE
// =====================================================

function drawNumber(id, x, y, color, isHovered = false) {
    const r = isHovered ? BADGE_RADIUS + 3 : BADGE_RADIUS;

    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = isHovered ? 3.5 : 2.5;
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.font = isHovered ? "bold 15px Arial" : "bold 13px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(id, x, y);
    ctx.restore();
}

// =====================================================
// LEGEND
// =====================================================

function drawLegend() {
    const visibleDets = detections.filter(d => visibleClasses[d.class_name]);
    if (visibleDets.length === 0) return;

    const startX = LEFT_MARGIN + imageWidth + 120;
    let y = TOP_MARGIN;

    ctx.save();

    // Title
    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 16px Arial";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("Legend", startX, y);

    y += 28;

    const maxPerCol = 16;
    const colWidth = 205;
    const rowHeight = 24;

    visibleDets.forEach((det, idx) => {
        const col = Math.floor(idx / maxPerCol);
        const row = idx % maxPerCol;

        const itemX = startX + col * colWidth;
        const itemY = y + row * rowHeight;

        const isHovered = (det.id === hoveredDetectionId);

        if (isHovered) {
            ctx.fillStyle = "#f1f5f9";
            ctx.fillRect(itemX - 4, itemY - 2, colWidth - 10, rowHeight - 2);
        }

        // Color circle badge
        ctx.beginPath();
        ctx.arc(itemX + 10, itemY + 10, 9, 0, Math.PI * 2);
        ctx.fillStyle = det.color;
        ctx.fill();

        // ID number
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 10px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(det.id, itemX + 10, itemY + 10);

        // Class text & confidence
        ctx.fillStyle = isHovered ? "#0284c7" : "#334155";
        ctx.font = isHovered ? "bold 12px Arial" : "12px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";

        let label = det.class_name;
        if (label.length > 18) {
            label = label.substring(0, 16) + "..";
        }
        ctx.fillText(label, itemX + 26, itemY + 10);
    });

    ctx.restore();
}

// =====================================================
// CANVAS MOUSE EVENTS
// =====================================================

function handleCanvasMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    let foundId = null;

    const visibleDets = detections.filter(d => visibleClasses[d.class_name]);
    for (let det of visibleDets) {
        if (!det.tx || !det.ty) continue;
        const dist = Math.hypot(mouseX - det.tx, mouseY - det.ty);
        if (dist <= BADGE_RADIUS + 5) {
            foundId = det.id;
            break;
        }
    }

    if (hoveredDetectionId !== foundId) {
        hoveredDetectionId = foundId;
        drawScene();
        highlightCard(foundId);
    }
}

function handleCanvasMouseLeave() {
    if (hoveredDetectionId !== null) {
        hoveredDetectionId = null;
        drawScene();
        highlightCard(null);
    }
}

// =====================================================
// UTILS
// =====================================================

function hexToRgba(hex, alpha) {
    hex = hex.replace("#", "");
    if (hex.length === 3) {
        hex = hex.split("").map(c => c + c).join("");
    }
    const num = parseInt(hex, 16);
    const r = (num >> 16) & 255;
    const g = (num >> 8) & 255;
    const b = num & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}