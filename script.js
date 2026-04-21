/* ═══════════════════════════════════════════════════════════════
   Marker Analyzer by Lookiar
   Pure-JS browser pipeline — no backend, no OpenCV dependency
   ═══════════════════════════════════════════════════════════════ */

const MAX_DIM = 512;
const CIRC    = 2 * Math.PI * 50; // SVG arc circumference ≈ 314.16

let state = {
  heatmapCanvas: null,
  scores:        null,
  opacity:       0.60,
};

/* ── DOMContentLoaded ──────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const banner = document.getElementById('cvBanner');
  if (banner) banner.hidden = true;
  document.getElementById('btnSelect').disabled = false;

  setupUploadZone();
  setupDownloads();

  document.getElementById('opacitySlider').addEventListener('input', e => {
    state.opacity = e.target.value / 100;
    renderCanvas();
  });

  document.getElementById('btnBack').addEventListener('click', resetToUpload);

  document.getElementById('btnLookiar').addEventListener('click', () => {
    window.open('https://lookiar.com', '_blank', 'noopener');
  });
});

/* ════════════════════════════════════════════════════════════════
   UPLOAD / NAVIGATION
   ════════════════════════════════════════════════════════════════ */
function setupUploadZone() {
  const zone  = document.getElementById('dropZone');
  const input = document.getElementById('fileInput');
  const btn   = document.getElementById('btnSelect');

  btn.addEventListener('click',  e => { e.stopPropagation(); input.click(); });
  zone.addEventListener('click', ()  => input.click());
  input.addEventListener('change', e => { if (e.target.files[0]) loadFile(e.target.files[0]); });

  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) loadFile(f);
  });
}

function loadFile(file) {
  if (file.size > 15 * 1024 * 1024) {
    alert('La imagen supera el límite de 15 MB. Usa una imagen más pequeña.');
    return;
  }
  showScreen('loading');
  setStatus('Cargando imagen…');

  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    URL.revokeObjectURL(url);
    document.getElementById('resFilename').textContent = file.name;
    setTimeout(() => runAnalysis(img), 60);
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    showScreen('upload');
    alert('No se pudo cargar la imagen. Prueba con un archivo JPG o PNG.');
  };
  img.src = url;
}

function showScreen(name) {
  const ids = { upload: 'screenUpload', loading: 'screenLoading', results: 'screenResults' };
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('visible'));
  document.getElementById(ids[name]).classList.add('visible');
}

function setStatus(msg) {
  const el = document.getElementById('loadingStatus');
  if (el) el.textContent = msg;
}

function resetToUpload() {
  document.getElementById('fileInput').value = '';
  state = { heatmapCanvas: null, scores: null, opacity: 0.60 };
  showScreen('upload');
}

/* ════════════════════════════════════════════════════════════════
   ANALYSIS PIPELINE  (async so loading status messages are visible)
   ════════════════════════════════════════════════════════════════ */
const tick = () => new Promise(r => requestAnimationFrame(r));

async function runAnalysis(img) {
  const scale = Math.min(1, MAX_DIM / Math.max(img.width, img.height));
  const aw = Math.round(img.width  * scale);
  const ah = Math.round(img.height * scale);

  const aCanvas = document.createElement('canvas');
  aCanvas.width = aw; aCanvas.height = ah;
  const aCtx = aCanvas.getContext('2d');
  aCtx.drawImage(img, 0, 0, aw, ah);
  const imageData = aCtx.getImageData(0, 0, aw, ah);

  setStatus('Convirtiendo a escala de grises…'); await tick();
  const gray = toGrayscale(imageData.data, aw, ah);

  setStatus('Calculando gradientes de imagen…'); await tick();
  const blurred      = gaussBlur(gray, aw, ah, 1.0);
  const { gx, gy, mag } = sobel(blurred, aw, ah);

  setStatus('Detectando esquinas y puntos de tracking…'); await tick();
  const harris = harrisResponse(gx, gy, aw, ah);
  const localV = localVariance(gray, aw, ah, 4);

  setStatus('Generando mapa de densidad de features…'); await tick();
  const trackMap = buildTrackMap(harris, mag, localV, aw, ah);

  setStatus('Calculando score de calidad…'); await tick();
  const keypoints = extractKeypoints(harris, aw, ah);
  const scores    = computeScores(gray, mag, harris, trackMap, keypoints, aw, ah);
  state.scores = scores;

  setStatus('Renderizando heatmap…'); await tick();
  state.heatmapCanvas  = buildHeatmapCanvas(imageData, trackMap, keypoints, aw, ah);

  displayResults(scores);
}

/* ════════════════════════════════════════════════════════════════
   IMAGE PROCESSING PRIMITIVES
   ════════════════════════════════════════════════════════════════ */

function toGrayscale(data, w, h) {
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++)
    gray[i] = 0.299 * data[i*4] + 0.587 * data[i*4+1] + 0.114 * data[i*4+2];
  return gray;
}

function gaussBlur(src, w, h, sigma) {
  const r    = Math.max(1, Math.ceil(3 * sigma));
  const size = 2 * r + 1;
  const k    = new Float32Array(size);
  let ksum = 0;
  for (let i = 0; i < size; i++) { k[i] = Math.exp(-((i-r)**2) / (2*sigma*sigma)); ksum += k[i]; }
  for (let i = 0; i < size; i++) k[i] /= ksum;

  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);

  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = -r; j <= r; j++)
        s += src[y * w + clamp(x + j, 0, w - 1)] * k[j + r];
      tmp[y * w + x] = s;
    }

  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = -r; j <= r; j++)
        s += tmp[clamp(y + j, 0, h - 1) * w + x] * k[j + r];
      out[y * w + x] = s;
    }

  return out;
}

function sobel(gray, w, h) {
  const gx  = new Float32Array(w * h);
  const gy  = new Float32Array(w * h);
  const mag = new Float32Array(w * h);
  const p   = (y, x) => gray[y * w + x];

  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const dx = -p(y-1,x-1) + p(y-1,x+1) - 2*p(y,x-1) + 2*p(y,x+1) - p(y+1,x-1) + p(y+1,x+1);
      const dy = -p(y-1,x-1) - 2*p(y-1,x) - p(y-1,x+1) + p(y+1,x-1) + 2*p(y+1,x) + p(y+1,x+1);
      const i  = y * w + x;
      gx[i] = dx; gy[i] = dy;
      mag[i] = Math.sqrt(dx*dx + dy*dy);
    }
  return { gx, gy, mag };
}

function harrisResponse(gx, gy, w, h) {
  const Ix2  = new Float32Array(w * h);
  const Iy2  = new Float32Array(w * h);
  const IxIy = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    Ix2[i]  = gx[i] * gx[i];
    Iy2[i]  = gy[i] * gy[i];
    IxIy[i] = gx[i] * gy[i];
  }

  const sIx2  = gaussBlur(Ix2,  w, h, 2.0);
  const sIy2  = gaussBlur(Iy2,  w, h, 2.0);
  const sIxIy = gaussBlur(IxIy, w, h, 2.0);

  const R = new Float32Array(w * h);
  const K = 0.04;
  for (let i = 0; i < w * h; i++) {
    const det   = sIx2[i] * sIy2[i] - sIxIy[i] * sIxIy[i];
    const trace = sIx2[i] + sIy2[i];
    R[i] = Math.max(0, det - K * trace * trace);
  }
  return R;
}

function localVariance(gray, w, h, r) {
  const out = new Float32Array(w * h);
  for (let y = r; y < h - r; y++)
    for (let x = r; x < w - r; x++) {
      let sum = 0, sum2 = 0, n = 0;
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          const v = gray[(y+dy)*w + (x+dx)];
          sum += v; sum2 += v*v; n++;
        }
      const mean = sum / n;
      out[y * w + x] = Math.max(0, sum2 / n - mean * mean);
    }
  return out;
}

function buildTrackMap(harris, mag, localV, w, h) {
  let maxH = 0, maxM = 0, maxV = 0;
  for (let i = 0; i < w * h; i++) {
    if (harris[i] > maxH) maxH = harris[i];
    if (mag[i]    > maxM) maxM = mag[i];
    if (localV[i] > maxV) maxV = localV[i];
  }

  const raw = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++)
    raw[i] = 0.45 * (maxH > 0 ? harris[i] / maxH : 0)
           + 0.35 * (maxM > 0 ? mag[i]    / maxM : 0)
           + 0.20 * (maxV > 0 ? localV[i] / maxV : 0);

  const sigma    = Math.max(4, Math.min(w, h) / 30);
  const smoothed = gaussBlur(raw, w, h, sigma);

  let maxS = 0;
  for (let i = 0; i < w * h; i++) if (smoothed[i] > maxS) maxS = smoothed[i];
  if (maxS > 0) for (let i = 0; i < w * h; i++) smoothed[i] /= maxS;

  return smoothed;
}

/* ════════════════════════════════════════════════════════════════
   KEYPOINT EXTRACTION — proper NMS in 7px radius window
   ════════════════════════════════════════════════════════════════ */
function extractKeypoints(harris, w, h) {
  let maxH = 0;
  for (let i = 0; i < w * h; i++) if (harris[i] > maxH) maxH = harris[i];
  if (maxH === 0) return [];

  const threshold = maxH * 0.05;
  const NMS = 7;
  const pts = [];

  for (let y = NMS; y < h - NMS; y++) {
    for (let x = NMS; x < w - NMS; x++) {
      const val = harris[y * w + x];
      if (val < threshold) continue;
      let isMax = true;
      outer: for (let dy = -NMS; dy <= NMS; dy++) {
        for (let dx = -NMS; dx <= NMS; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (harris[(y + dy) * w + (x + dx)] >= val) { isMax = false; break outer; }
        }
      }
      if (isMax) pts.push({ x, y, strength: val / maxH });
    }
  }
  return pts;
}

/* ════════════════════════════════════════════════════════════════
   SCORING
   ════════════════════════════════════════════════════════════════ */
function computeScores(gray, mag, harris, trackMap, keypoints, w, h) {
  const sharpness = scoreSharpness(gray, w, h);          // 0–25
  const features  = scoreFeatures(harris, w, h);          // 0–25
  const contrast  = scoreContrast(gray, w, h);            // 0–25
  const distData  = analyzeDistribution(trackMap, w, h);  // object
  const distribution = distData.score;                    // 0–25

  // Coverage penalty: multiplier 0.55–1.0 applied to total
  // Images with large empty areas are penalized even if their active zones are strong
  const coveragePenalty = computeCoveragePenalty(distData.coverageRatio);

  const raw   = sharpness + features + distribution + contrast;
  const total = Math.round(Math.max(0, raw * coveragePenalty));

  return {
    sharpness,
    features,
    distribution,
    contrast,
    total,
    kp:            keypoints.length,
    coverageRatio: distData.coverageRatio,
    gridScore:     distData.gridScore,
    edgeScore:     distData.edgeScore,
    goodCells:     distData.goodCells,
    totalCells:    distData.totalCells,
  };
}

/* Laplacian variance — proxy for sharpness / texture richness */
function scoreSharpness(gray, w, h) {
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const lap = gray[(y-1)*w+x] + gray[(y+1)*w+x] + gray[y*w+x-1] + gray[y*w+x+1] - 4*gray[y*w+x];
      sum += lap; sum2 += lap*lap; n++;
    }
  const mean = sum / n;
  return 25 * Math.min(1, (sum2 / n - mean * mean) / 1200);
}

/* Fraction of pixels with strong Harris response */
function scoreFeatures(harris, w, h) {
  let maxH = 0;
  for (let i = 0; i < w * h; i++) if (harris[i] > maxH) maxH = harris[i];
  if (maxH === 0) return 0;
  const threshold = maxH * 0.05;
  let count = 0;
  for (let i = 0; i < w * h; i++) if (harris[i] > threshold) count++;
  return 25 * Math.min(1, count / (w * h) / 0.06);
}

/* Standard deviation of pixel luminance */
function scoreContrast(gray, w, h) {
  let sum = 0;
  for (let i = 0; i < w * h; i++) sum += gray[i];
  const mean = sum / (w * h);
  let sq = 0;
  for (let i = 0; i < w * h; i++) sq += (gray[i] - mean) ** 2;
  return 25 * Math.min(1, Math.sqrt(sq / (w * h)) / 75);
}

/*
 * Distribution analysis — combines three spatial metrics into a 0–25 score.
 *
 * 1. Grid score:     6×6 grid, counts cells where the MEAN density exceeds a
 *                    threshold. Mean (not max) prevents a single strong pixel
 *                    from falsely marking an otherwise empty cell as covered.
 *
 * 2. Coverage score: fraction of the total image area with trackMap > 0.10.
 *                    Penalizes images where features cluster in a small zone.
 *
 * 3. Edge score:     mean density in the 15% border ring. AR trackers need
 *                    features near the edges to compute stable pose.
 */
function analyzeDistribution(trackMap, w, h) {
  // 1 — Grid coverage (6×6)
  const GRID = 6;
  const cw = w / GRID, ch = h / GRID;
  let goodCells = 0;

  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      let sum = 0, count = 0;
      const x0 = Math.floor(gx * cw), x1 = Math.floor((gx + 1) * cw);
      const y0 = Math.floor(gy * ch), y1 = Math.floor((gy + 1) * ch);
      for (let y = y0; y < y1; y++)
        for (let x = x0; x < x1; x++) { sum += trackMap[y * w + x]; count++; }
      if (count > 0 && sum / count > 0.18) goodCells++;
    }
  }
  const totalCells  = GRID * GRID;
  const gridScore   = goodCells / totalCells;  // 0–1

  // 2 — Pixel coverage (fraction of image with meaningful density)
  let coveredPx = 0;
  for (let i = 0; i < w * h; i++) if (trackMap[i] > 0.10) coveredPx++;
  const coverageRatio  = coveredPx / (w * h);
  const coverageScore  = Math.min(1, coverageRatio / 0.50);  // 50% = full score

  // 3 — Edge/border strip coverage (15% ring from each edge)
  const bw = Math.max(1, Math.round(w * 0.15));
  const bh = Math.max(1, Math.round(h * 0.15));
  let edgeSum = 0, edgeCount = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (y < bh || y >= h - bh || x < bw || x >= w - bw) {
        edgeSum += trackMap[y * w + x];
        edgeCount++;
      }
    }
  }
  const edgeMean  = edgeCount > 0 ? edgeSum / edgeCount : 0;
  const edgeScore = Math.min(1, edgeMean / 0.25);  // mean 0.25 in border ring = full score

  const combined = 0.40 * gridScore + 0.35 * coverageScore + 0.25 * edgeScore;

  return {
    score: 25 * combined,
    gridScore,
    coverageScore,
    edgeScore,
    coverageRatio,
    goodCells,
    totalCells,
  };
}

/*
 * Coverage penalty multiplier applied to the total score.
 * An image with only 15% of its area covered gets a 45% penalty.
 * No penalty once coverage reaches 45%.
 */
function computeCoveragePenalty(coverageRatio) {
  if (coverageRatio >= 0.45) return 1.0;
  if (coverageRatio <= 0.15) return 0.55;
  return 0.55 + ((coverageRatio - 0.15) / 0.30) * 0.45;
}

/* ════════════════════════════════════════════════════════════════
   HEATMAP CANVAS
   Stores original ImageData + overlay canvas as properties
   so renderCanvas can compose them at any opacity.
   Colormap: Blue (low) → Yellow (medium) → Red (high/good)
   ════════════════════════════════════════════════════════════════ */
function featureDensityColor(t) {
  let r, g, b;
  if (t < 0.5) {
    const s = t * 2;
    r = Math.round(59  + s * (234 - 59));
    g = Math.round(130 + s * (179 - 130));
    b = Math.round(246 + s * (8   - 246));
  } else {
    const s = (t - 0.5) * 2;
    r = Math.round(234 + s * (239 - 234));
    g = Math.round(179 + s * (68  - 179));
    b = Math.round(8   + s * (68  - 8));
  }
  return [r, g, b];
}

function buildHeatmapCanvas(imageData, trackMap, keypoints, w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;

  // Overlay: feature density colormap
  const overlayData = new ImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const t = trackMap[i];
    const [r, g, b] = featureDensityColor(t);
    overlayData.data[i*4]   = r;
    overlayData.data[i*4+1] = g;
    overlayData.data[i*4+2] = b;
    overlayData.data[i*4+3] = t > 0.04 ? Math.min(215, Math.round((0.25 + t * 0.75) * 215)) : 0;
  }
  const overlayCanvas = document.createElement('canvas');
  overlayCanvas.width = w; overlayCanvas.height = h;
  overlayCanvas.getContext('2d').putImageData(overlayData, 0, 0);

  // Keypoints layer: dark outer ring + white dot
  const kpCanvas = document.createElement('canvas');
  kpCanvas.width = w; kpCanvas.height = h;
  const kpCtx = kpCanvas.getContext('2d');
  keypoints.forEach(({ x, y, strength }) => {
    const r = Math.max(2, Math.round(2 + strength * 2));
    kpCtx.beginPath();
    kpCtx.arc(x, y, r + 1, 0, Math.PI * 2);
    kpCtx.fillStyle = 'rgba(0,0,0,0.45)';
    kpCtx.fill();
    kpCtx.beginPath();
    kpCtx.arc(x, y, r, 0, Math.PI * 2);
    kpCtx.fillStyle = '#ffffff';
    kpCtx.fill();
  });

  canvas._overlay   = overlayCanvas;
  canvas._kpCanvas  = kpCanvas;
  canvas._imageData = imageData;
  return canvas;
}

/* ════════════════════════════════════════════════════════════════
   DISPLAY RESULTS
   ════════════════════════════════════════════════════════════════ */
function displayResults(scores) {
  updateGauge(scores.total);
  updateKeypoints(scores);
  updateMetrics(scores);
  updateDiagnosis(scores);
  updateRecommendations(scores);
  document.getElementById('heatLegend').style.display = 'flex';
  renderCanvas();
  showScreen('results');
}

function updateGauge(total) {
  const arc   = document.getElementById('scoreArc');
  const color = scoreColor(total);

  arc.setAttribute('stroke-dasharray', `${(total / 100) * CIRC} ${CIRC}`);
  arc.style.stroke = color;

  const numEl   = document.getElementById('scoreNum');
  const labelEl = document.getElementById('scoreLabel');
  numEl.textContent   = total;
  numEl.style.color   = color;
  labelEl.textContent = scoreCategory(total);
  labelEl.style.color = color;
}

function updateKeypoints(scores) {
  document.getElementById('kpCount').textContent = scores.kp.toLocaleString('es');
  const q = scores.features / 25;
  let sub;
  if (q < 0.3)      sub = 'Pocos puntos — imagen con poco detalle visual';
  else if (q < 0.6) sub = 'Cantidad moderada de puntos de tracking';
  else              sub = 'Buena densidad de puntos para AR estable';
  document.getElementById('kpSub').textContent = sub;
}

function updateMetrics(scores) {
  setBar('barDist',     'valDist',     scores.distribution);
  setBar('barContrast', 'valContrast', scores.contrast);
  setBar('barTexture',  'valTexture',  (scores.sharpness + scores.features) / 2);
}

function setBar(barId, valId, raw25) {
  const pct = (raw25 / 25) * 100;
  const el  = document.getElementById(barId);
  if (!el) return;
  el.style.width      = pct + '%';
  el.style.background = metricColor(pct);
  document.getElementById(valId).textContent = Math.round(pct) + '%';
}

function metricColor(pct) {
  if (pct >= 75) return '#22c55e';
  if (pct >= 50) return '#29ABE2';
  if (pct >= 30) return '#eab308';
  return '#ef4444';
}

/* ════════════════════════════════════════════════════════════════
   DIAGNOSIS
   Structural problems (concentration, empty areas, weak borders)
   take priority over the generic score-range messages.
   ════════════════════════════════════════════════════════════════ */
function updateDiagnosis(scores) {
  const { badge, cls, text } = getDiagnosis(scores);
  const el = document.getElementById('diagBadge');
  el.textContent = badge;
  el.className   = 'diag-badge badge-' + cls;
  document.getElementById('diagText').textContent = text;
}

function getDiagnosis(scores) {
  const { total, coverageRatio, gridScore, edgeScore, goodCells, totalCells } = scores;
  const gridCoverage = goodCells / totalCells;

  // Concentrated information / large empty areas
  if (coverageRatio < 0.25 || gridCoverage < 0.30) {
    const cls = total >= 50 ? 'average' : 'poor';
    return {
      badge: total >= 50 ? 'Regular' : 'Mala',
      cls,
      text: 'La información visual está concentrada en una zona pequeña de la imagen. '
          + 'Grandes áreas sin textura pueden afectar gravemente la estabilidad del '
          + 'tracking AR, especialmente cuando la cámara se mueve o el marcador '
          + 'está parcialmente visible.',
    };
  }

  // Weak borders
  if (edgeScore < 0.30) {
    const isBorderGood = total >= 60;
    return {
      badge: isBorderGood ? 'Buena' : 'Regular',
      cls:   isBorderGood ? 'good'  : 'average',
      text: 'Los bordes de la imagen tienen baja densidad de información visual. '
          + 'Los sistemas AR necesitan features en toda la superficie —incluyendo '
          + 'los bordes— para calcular la posición y orientación del marcador con precisión.',
    };
  }

  // Generic score-based messages
  if (total >= 80) return {
    badge: 'Excelente', cls: 'excellent',
    text: 'Imagen de alta calidad para AR tracking. Tiene buena densidad de features, '
        + 'distribución equilibrada en toda la superficie y buen contraste. '
        + 'Los sistemas de WebAR la detectarán de forma rápida y estable.',
  };
  if (total >= 65) return {
    badge: 'Buena', cls: 'good',
    text: 'Buena imagen para tracking AR. Funcionará correctamente en la mayoría de '
        + 'condiciones. La distribución de features es adecuada, con margen menor de mejora.',
  };
  if (total >= 45) return {
    badge: 'Regular', cls: 'average',
    text: 'La imagen puede funcionar como marcador pero con limitaciones. El tracking '
        + 'puede ser inestable con iluminación variable o desde distancias mayores. '
        + 'Aplica las mejoras sugeridas para mayor estabilidad.',
  };
  if (total >= 25) return {
    badge: 'Mala', cls: 'poor',
    text: 'Esta imagen tiene problemas significativos para tracking AR. El sistema '
        + 'posiblemente no la detecte de forma confiable. Revisa cada punto '
        + 'específico y considera rediseñar la imagen.',
  };
  return {
    badge: 'Muy mala', cls: 'poor',
    text: 'Esta imagen no es apta para tracking AR. Carece de las características '
        + 'visuales mínimas necesarias. Usa una imagen con más texturas, bordes '
        + 'nítidos y zonas de alto contraste distribuidas por toda la superficie.',
  };
}

/* ════════════════════════════════════════════════════════════════
   RECOMMENDATIONS
   ════════════════════════════════════════════════════════════════ */
function updateRecommendations(scores) {
  const recs = getRecommendations(scores);
  const ul   = document.getElementById('recoList');
  ul.innerHTML = '';
  if (recs.length === 0) {
    const li = document.createElement('li');
    li.textContent = '¡Imagen óptima! No se requieren mejoras adicionales.';
    ul.appendChild(li);
    return;
  }
  recs.forEach(text => {
    const li = document.createElement('li');
    li.textContent = text;
    ul.appendChild(li);
  });
}

function getRecommendations(scores) {
  const recs = [];
  const pS = scores.sharpness    / 25 * 100;
  const pF = scores.features     / 25 * 100;
  const pD = scores.distribution / 25 * 100;
  const pC = scores.contrast     / 25 * 100;
  const { coverageRatio, gridScore, edgeScore, goodCells, totalCells } = scores;
  const gridCoverage = goodCells / totalCells;

  // Coverage / empty areas
  if (coverageRatio < 0.25) {
    recs.push(
      'Más del ' + Math.round((1 - coverageRatio) * 100) + '% de la imagen tiene muy baja '
      + 'densidad de información. Añade textura, patrones o detalles en las zonas vacías '
      + 'para cubrir la superficie completa.'
    );
  } else if (coverageRatio < 0.40) {
    recs.push(
      'Grandes áreas sin textura pueden afectar el tracking. Distribuye el contenido '
      + 'visual por toda la imagen, no solo en el centro o en una zona concreta.'
    );
  }

  // Grid concentration
  if (gridCoverage < 0.35) {
    recs.push(
      'La información visual está concentrada en pocas zonas ('
      + goodCells + ' de ' + totalCells + ' sectores con buena densidad). '
      + 'Un marcador ideal llena uniformemente toda su superficie.'
    );
  } else if (gridCoverage < 0.55) {
    recs.push(
      'Varios sectores de la imagen tienen baja densidad de features. '
      + 'Intenta llenar las zonas más vacías con detalles visuales adicionales.'
    );
  }

  // Edge quality
  if (edgeScore < 0.30) {
    recs.push(
      'Los bordes de la imagen tienen poca información visual. Agrega elementos '
      + 'cerca de los bordes y esquinas — son críticos para que el tracker '
      + 'calcule la orientación del marcador con precisión.'
    );
  } else if (edgeScore < 0.50) {
    recs.push(
      'Refuerza los bordes con más detalles visuales para mejorar la '
      + 'estabilidad del tracking en movimiento.'
    );
  }

  // Sharpness / texture
  if (pS < 50)
    recs.push('Usa una imagen más nítida. Evita fotos desenfocadas o con motion blur. La versión "Mejorada" adjunta aplica un filtro de enfoque automático.');
  else if (pS < 75)
    recs.push('La nitidez es aceptable pero puede mejorar. Fotografía con buena iluminación y sin movimiento de cámara.');

  // Feature density
  if (pF < 40)
    recs.push('Agrega más detalles visuales: texto, líneas irregulares, patrones asimétricos o fotografías con mucha textura. Las zonas de color sólido son invisibles para el tracker.');
  else if (pF < 65)
    recs.push('Enriquece la imagen con más detalles finos. Bordes, esquinas y texturas variadas mejoran la detección.');

  // Distribution bar (only if coverage is acceptable, avoiding duplicate messages)
  if (pD < 50 && coverageRatio >= 0.40)
    recs.push('La distribución general puede mejorar. Asegúrate de que los puntos de tracking se extiendan por toda la imagen, incluyendo esquinas.');

  // Contrast
  if (pC < 40)
    recs.push('El contraste es muy bajo. Aumenta la diferencia entre zonas claras y oscuras. Evita fondos planos o colores muy similares.');
  else if (pC < 60)
    recs.push('Mejora el contraste global para facilitar la detección en distintas condiciones de luz.');

  if (scores.total >= 75 && recs.length === 0)
    recs.push('Para máximo rendimiento, descarga y usa la imagen mejorada: incluye enfoque optimizado y contraste aumentado.');

  return recs;
}

/* ════════════════════════════════════════════════════════════════
   CANVAS RENDERER
   ════════════════════════════════════════════════════════════════ */
function renderCanvas() {
  const canvas = document.getElementById('displayCanvas');
  const ctx    = canvas.getContext('2d');
  const src    = state.heatmapCanvas;
  if (!src) return;
  canvas.width = src.width; canvas.height = src.height;
  ctx.putImageData(src._imageData, 0, 0);
  ctx.globalAlpha = state.opacity;
  ctx.drawImage(src._overlay, 0, 0);
  ctx.globalAlpha = 1;
  ctx.drawImage(src._kpCanvas, 0, 0);
}

/* ════════════════════════════════════════════════════════════════
   DOWNLOADS
   ════════════════════════════════════════════════════════════════ */
function setupDownloads() {
  document.getElementById('btnDlHeatmap').addEventListener('click', () => {
    if (!state.heatmapCanvas) return;
    renderCanvas();
    downloadCanvas(document.getElementById('displayCanvas'), 'marker-feature-density.png');
  });
}

function downloadCanvas(canvas, filename, mime = 'image/png', quality = 1) {
  const a = document.createElement('a');
  a.download = filename;
  a.href = canvas.toDataURL(mime, quality);
  a.click();
}

/* ════════════════════════════════════════════════════════════════
   UTILITIES
   ════════════════════════════════════════════════════════════════ */
const clamp   = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const clamp01 = v => Math.max(0, Math.min(1, v));

function scoreColor(score) {
  if (score >= 80) return '#22c55e';
  if (score >= 65) return '#29ABE2';
  if (score >= 45) return '#eab308';
  return '#ef4444';
}

function scoreCategory(score) {
  if (score >= 80) return 'Excelente';
  if (score >= 65) return 'Bueno';
  if (score >= 45) return 'Regular';
  if (score >= 25) return 'Malo';
  return 'Muy malo';
}
