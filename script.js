/* ═══════════════════════════════════════════════════════════════
   Marker Analyzer by Lookiar
   Pure-JS browser pipeline — no backend, no OpenCV dependency
   ═══════════════════════════════════════════════════════════════ */

const MAX_DIM  = 512;               // working resolution (px)
const CIRC     = 2 * Math.PI * 50;  // SVG arc circumference (r=50 → ≈314.16)

/* ── State ─────────────────────────────────────────────────────── */
let state = {
  heatmapCanvas:  null,   // canvas with original + overlay references
  improvedCanvas: null,   // canvas with sharpened + contrast-stretched image
  scores:         null,   // { sharpness, features, distribution, contrast, total, kp }
  currentMode:    'heatmap',
  opacity:        0.60,
};

/* ── DOMContentLoaded ──────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  // Pure JS — no OpenCV needed, hide loading banner and enable button immediately
  const banner = document.getElementById('cvBanner');
  if (banner) banner.hidden = true;
  document.getElementById('btnSelect').disabled = false;

  setupUploadZone();
  setupTabs();
  setupDownloads();

  document.getElementById('opacitySlider').addEventListener('input', e => {
    state.opacity = e.target.value / 100;
    if (state.currentMode === 'heatmap') renderCanvas('heatmap');
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
    // yield to browser so the loading screen actually paints before heavy work
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
  state = { heatmapCanvas: null, improvedCanvas: null, scores: null,
            currentMode: 'heatmap', opacity: 0.60 };
  showScreen('upload');
}

/* ════════════════════════════════════════════════════════════════
   ANALYSIS PIPELINE  (async so loading status messages are visible)
   ════════════════════════════════════════════════════════════════ */
const tick = () => new Promise(r => requestAnimationFrame(r));

async function runAnalysis(img) {
  // Scale to working resolution
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
  const blurred     = gaussBlur(gray, aw, ah, 1.0);
  const { gx, gy, mag } = sobel(blurred, aw, ah);

  setStatus('Detectando esquinas y puntos de tracking…'); await tick();
  const harris = harrisResponse(gx, gy, aw, ah);
  const localV = localVariance(gray, aw, ah, 4);

  setStatus('Generando mapa de trackabilidad…'); await tick();
  const trackMap = buildTrackMap(harris, mag, localV, aw, ah);

  setStatus('Calculando score de calidad…'); await tick();
  const scores = computeScores(gray, mag, harris, trackMap, aw, ah);
  state.scores = scores;

  setStatus('Renderizando heatmap…'); await tick();
  state.heatmapCanvas  = buildHeatmapCanvas(imageData, trackMap, aw, ah);

  setStatus('Generando imagen mejorada…'); await tick();
  state.improvedCanvas = buildImprovedCanvas(imageData, aw, ah);

  displayResults(scores);
}

/* ════════════════════════════════════════════════════════════════
   IMAGE PROCESSING PRIMITIVES
   ════════════════════════════════════════════════════════════════ */

/* Luminance-weighted grayscale */
function toGrayscale(data, w, h) {
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++)
    gray[i] = 0.299 * data[i*4] + 0.587 * data[i*4+1] + 0.114 * data[i*4+2];
  return gray;
}

/* Separable Gaussian blur */
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

/* Sobel gradients + gradient magnitude */
function sobel(gray, w, h) {
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  const mag = new Float32Array(w * h);
  const p = (y, x) => gray[y * w + x];

  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const dx = -p(y-1,x-1) + p(y-1,x+1) - 2*p(y,x-1) + 2*p(y,x+1) - p(y+1,x-1) + p(y+1,x+1);
      const dy = -p(y-1,x-1) - 2*p(y-1,x) - p(y-1,x+1) + p(y+1,x-1) + 2*p(y+1,x) + p(y+1,x+1);
      const i = y * w + x;
      gx[i] = dx; gy[i] = dy;
      mag[i] = Math.sqrt(dx*dx + dy*dy);
    }
  return { gx, gy, mag };
}

/* Harris corner detector R = det(M) − k·trace²(M) */
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

/* Local variance — measures texture richness */
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

/* Weighted blend of Harris + gradient magnitude + local variance → 0–1 map */
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

  // Smooth for visually pleasant gradient heatmap
  const sigma    = Math.max(4, Math.min(w, h) / 30);
  const smoothed = gaussBlur(raw, w, h, sigma);

  let maxS = 0;
  for (let i = 0; i < w * h; i++) if (smoothed[i] > maxS) maxS = smoothed[i];
  if (maxS > 0) for (let i = 0; i < w * h; i++) smoothed[i] /= maxS;

  return smoothed;
}

/* ════════════════════════════════════════════════════════════════
   SCORING  — each component contributes 0–25 pts, total 0–100
   ════════════════════════════════════════════════════════════════ */
function computeScores(gray, mag, harris, trackMap, w, h) {
  const sharpness    = scoreSharpness(gray, w, h);
  const features     = scoreFeatures(harris, w, h);
  const distribution = scoreDistribution(trackMap, w, h);
  const contrast     = scoreContrast(gray, w, h);
  const total        = Math.round(sharpness + features + distribution + contrast);
  const kp           = estimateKeypoints(harris, w, h);
  return { sharpness, features, distribution, contrast, total, kp };
}

/* Laplacian variance — proxy for image sharpness */
function scoreSharpness(gray, w, h) {
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const lap = gray[(y-1)*w+x] + gray[(y+1)*w+x] + gray[y*w+x-1] + gray[y*w+x+1] - 4*gray[y*w+x];
      sum += lap; sum2 += lap*lap; n++;
    }
  const mean = sum / n;
  const variance = sum2 / n - mean * mean;
  return 25 * Math.min(1, variance / 1200);
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

/* How evenly trackable zones cover a 6×6 grid */
function scoreDistribution(trackMap, w, h) {
  const GRID = 6;
  const cw = w / GRID, ch = h / GRID;
  let covered = 0;
  for (let gy = 0; gy < GRID; gy++)
    for (let gx = 0; gx < GRID; gx++) {
      let cellMax = 0;
      const x0 = Math.floor(gx * cw), x1 = Math.floor((gx + 1) * cw);
      const y0 = Math.floor(gy * ch), y1 = Math.floor((gy + 1) * ch);
      for (let y = y0; y < y1; y++)
        for (let x = x0; x < x1; x++)
          if (trackMap[y * w + x] > cellMax) cellMax = trackMap[y * w + x];
      if (cellMax > 0.2) covered++;
    }
  return 25 * (covered / (GRID * GRID));
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

/* Approximate keypoint count (Harris with implicit NMS) */
function estimateKeypoints(harris, w, h) {
  let maxH = 0;
  for (let i = 0; i < w * h; i++) if (harris[i] > maxH) maxH = harris[i];
  if (maxH === 0) return 0;
  const threshold = maxH * 0.05;
  let count = 0;
  for (let i = 0; i < w * h; i++) if (harris[i] > threshold) count++;
  return Math.max(0, Math.round(count / 9)); // /9 approximates NMS suppression
}

/* ════════════════════════════════════════════════════════════════
   HEATMAP CANVAS
   Stores original ImageData + overlay canvas as properties
   so renderCanvas can compose them at any opacity.
   ════════════════════════════════════════════════════════════════ */
function jetColor(t) {
  return [
    clamp01(1.5 - Math.abs(4*t - 3)) * 255,
    clamp01(1.5 - Math.abs(4*t - 2)) * 255,
    clamp01(1.5 - Math.abs(4*t - 1)) * 255,
  ];
}

function buildHeatmapCanvas(imageData, trackMap, w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);

  const overlayData = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const t = trackMap[i];
    const [r, g, b] = jetColor(t);
    overlayData.data[i*4]   = r;
    overlayData.data[i*4+1] = g;
    overlayData.data[i*4+2] = b;
    overlayData.data[i*4+3] = t > 0.05 ? Math.min(255, t * 1.4 * 255) : 0;
  }

  const overlayCanvas = document.createElement('canvas');
  overlayCanvas.width = w; overlayCanvas.height = h;
  overlayCanvas.getContext('2d').putImageData(overlayData, 0, 0);

  canvas._overlay   = overlayCanvas;
  canvas._imageData = imageData;
  return canvas;
}

/* ════════════════════════════════════════════════════════════════
   IMPROVED IMAGE  — unsharp mask + per-channel contrast stretch
   ════════════════════════════════════════════════════════════════ */
function buildImprovedCanvas(imageData, w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');

  const src = imageData.data;
  const out = ctx.createImageData(w, h);
  const d   = out.data;

  const R = new Float32Array(w * h);
  const G = new Float32Array(w * h);
  const B = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) { R[i] = src[i*4]; G[i] = src[i*4+1]; B[i] = src[i*4+2]; }

  const sigma = 1.2, amount = 0.65;
  const bR = gaussBlur(R, w, h, sigma);
  const bG = gaussBlur(G, w, h, sigma);
  const bB = gaussBlur(B, w, h, sigma);

  // First pass: find range after unsharp mask for contrast stretching
  let minR=255, maxR=0, minG=255, maxG=0, minB=255, maxB=0;
  for (let i = 0; i < w * h; i++) {
    const sr = clamp(R[i] + amount * (R[i] - bR[i]), 0, 255);
    const sg = clamp(G[i] + amount * (G[i] - bG[i]), 0, 255);
    const sb = clamp(B[i] + amount * (B[i] - bB[i]), 0, 255);
    if (sr < minR) minR = sr; if (sr > maxR) maxR = sr;
    if (sg < minG) minG = sg; if (sg > maxG) maxG = sg;
    if (sb < minB) minB = sb; if (sb > maxB) maxB = sb;
  }

  const stretch = (v, mn, mx) => (mx - mn) > 10 ? (v - mn) / (mx - mn) * 255 : v;

  // Second pass: apply + stretch
  for (let i = 0; i < w * h; i++) {
    d[i*4]   = Math.round(stretch(clamp(R[i] + amount*(R[i]-bR[i]), 0, 255), minR, maxR));
    d[i*4+1] = Math.round(stretch(clamp(G[i] + amount*(G[i]-bG[i]), 0, 255), minG, maxG));
    d[i*4+2] = Math.round(stretch(clamp(B[i] + amount*(B[i]-bB[i]), 0, 255), minB, maxB));
    d[i*4+3] = src[i*4+3];
  }

  ctx.putImageData(out, 0, 0);
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

  state.currentMode = 'heatmap';
  syncTabs('heatmap');
  document.getElementById('heatLegend').style.display = 'flex';
  renderCanvas('heatmap');

  showScreen('results');
}

/* Score ring — animates via CSS transition on stroke-dasharray */
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

/* Keypoint counter + quality subtitle */
function updateKeypoints(scores) {
  document.getElementById('kpCount').textContent = scores.kp.toLocaleString('es');
  const q = scores.features / 25;
  let sub;
  if (q < 0.3)      sub = 'Pocos puntos — imagen con poco detalle visual';
  else if (q < 0.6) sub = 'Cantidad moderada de puntos de tracking';
  else              sub = 'Buena densidad de puntos para AR estable';
  document.getElementById('kpSub').textContent = sub;
}

/* Three metric bars: Distribución, Contraste, Textura/bordes */
function updateMetrics(scores) {
  setBar('barDist',     'valDist',     scores.distribution);
  setBar('barContrast', 'valContrast', scores.contrast);
  // Texture = blend of sharpness (edge quality) + feature density
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

/* Diagnosis badge + explanatory text */
function updateDiagnosis(scores) {
  const { badge, cls, text } = getDiagnosis(scores);
  const el = document.getElementById('diagBadge');
  el.textContent = badge;
  el.className   = 'diag-badge badge-' + cls;
  document.getElementById('diagText').textContent = text;
}

function getDiagnosis({ total }) {
  if (total >= 80) return {
    badge: 'Excelente', cls: 'excellent',
    text: 'Esta imagen es un marcador AR de alta calidad. Tiene excelente nitidez, '
        + 'puntos de tracking bien distribuidos y buen contraste. Los sistemas de '
        + 'WebAR la detectarán de forma rápida y estable.',
  };
  if (total >= 65) return {
    badge: 'Buena', cls: 'good',
    text: 'Buena imagen para tracking AR. Funcionará correctamente en la mayoría de '
        + 'condiciones. Hay pequeñas áreas de mejora, pero no son críticas para el '
        + 'funcionamiento básico.',
  };
  if (total >= 45) return {
    badge: 'Regular', cls: 'average',
    text: 'La imagen puede funcionar como marcador pero con limitaciones. El tracking '
        + 'puede ser inestable con iluminación cambiante o desde distancias mayores. '
        + 'Se recomienda aplicar las mejoras sugeridas.',
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
        + 'visuales mínimas que los algoritmos necesitan. Usa una imagen con más '
        + 'texturas, bordes nítidos y zonas de alto contraste.',
  };
}

/* Actionable recommendations based on individual scores */
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

  if (pS < 50)
    recs.push('Usa una imagen más nítida. Evita fotos desenfocadas o con motion blur. La versión "Mejorada" adjunta aplica un filtro de enfoque automático.');
  else if (pS < 75)
    recs.push('La nitidez es aceptable pero puede mejorar. Fotografía con buena iluminación y sin movimiento de cámara.');

  if (pF < 40)
    recs.push('Agrega más detalles visuales: texto, líneas irregulares, patrones asimétricos o fotografías con mucha textura. Las zonas de color sólido son invisibles para el tracker.');
  else if (pF < 65)
    recs.push('Enriquece la imagen con más detalles finos. Bordes, esquinas y texturas variadas mejoran la detección.');

  if (pD < 50)
    recs.push('Los puntos de tracking están concentrados en pocas zonas. Distribuye el contenido visual por toda la imagen, incluyendo esquinas y bordes.');
  else if (pD < 70)
    recs.push('Coloca elementos de interés también en los bordes y esquinas de la imagen para mejorar la distribución.');

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
function renderCanvas(mode) {
  const canvas = document.getElementById('displayCanvas');
  const ctx    = canvas.getContext('2d');
  const src    = state.heatmapCanvas;
  if (!src) return;

  const iw = src.width, ih = src.height;

  if (mode === 'original') {
    canvas.width = iw; canvas.height = ih;
    ctx.putImageData(src._imageData, 0, 0);

  } else if (mode === 'heatmap') {
    canvas.width = iw; canvas.height = ih;
    ctx.putImageData(src._imageData, 0, 0);
    ctx.globalAlpha = state.opacity;
    ctx.drawImage(src._overlay, 0, 0);
    ctx.globalAlpha = 1;

  } else if (mode === 'improved') {
    // Side-by-side comparison: original | divider | improved
    const gap = 6;
    canvas.width  = iw * 2 + gap;
    canvas.height = ih;

    ctx.putImageData(src._imageData, 0, 0);

    // Accent-colored divider
    ctx.fillStyle = 'rgba(56,189,248,0.6)';
    ctx.fillRect(iw + 1, 0, gap - 2, ih);

    ctx.drawImage(state.improvedCanvas, iw + gap, 0);

    // Labels
    const labelH = 22, labelPad = 8;
    ctx.font = 'bold 11px "Segoe UI", system-ui, sans-serif';

    ['Original', 'Mejorada'].forEach((label, idx) => {
      const offsetX = idx === 0 ? 0 : iw + gap;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(offsetX + labelPad, ih - labelH - labelPad, tw + 16, labelH);
      ctx.fillStyle = '#e2f4ff';
      ctx.fillText(label, offsetX + labelPad + 8, ih - labelPad - 7);
    });
  }
}

/* ════════════════════════════════════════════════════════════════
   TABS
   ════════════════════════════════════════════════════════════════ */
function setupTabs() {
  document.querySelectorAll('.vtab').forEach(btn =>
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      state.currentMode = mode;
      syncTabs(mode);
      renderCanvas(mode);
      document.getElementById('heatLegend').style.display = mode === 'heatmap' ? 'flex' : 'none';
    })
  );
}

function syncTabs(active) {
  document.querySelectorAll('.vtab').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === active)
  );
}

/* ════════════════════════════════════════════════════════════════
   DOWNLOADS
   ════════════════════════════════════════════════════════════════ */
function setupDownloads() {
  document.getElementById('btnDlHeatmap').addEventListener('click', () => {
    if (!state.heatmapCanvas) return;
    // Ensure heatmap is rendered to displayCanvas at current opacity
    if (state.currentMode !== 'heatmap') {
      state.currentMode = 'heatmap';
      syncTabs('heatmap');
      renderCanvas('heatmap');
    }
    downloadCanvas(document.getElementById('displayCanvas'), 'marker-heatmap.png');
  });

  document.getElementById('btnDlImproved').addEventListener('click', () => {
    if (!state.improvedCanvas) return;
    downloadCanvas(state.improvedCanvas, 'marker-improved.jpg', 'image/jpeg', 0.92);
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
