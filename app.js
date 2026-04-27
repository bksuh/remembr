// NavQA trajectory viewer — pure browser, no build step.
// Loads site/index.json once, then site/meta/<idx>.json on demand.

const els = {
  picker: document.getElementById('row-picker'),
  prev: document.getElementById('prev'),
  next: document.getElementById('next'),
  list: document.getElementById('row-list'),
  cursorToggle: document.getElementById('cursor-toggle'),
  fovToggle: document.getElementById('fov-toggle'),
  cursorOverlay: document.getElementById('cursor-overlay'),
  plot: document.getElementById('plot'),
  plotWrap: document.getElementById('plot-wrap'),
  frames: document.getElementById('frames-grid'),
  info: document.getElementById('info-panel'),
};

const state = {
  index: [],
  currentIdx: null,
  cursorMoveHandler: null,
  currentMeta: null,
  frameFovTraceIndices: [],
};

// ------- color helpers -------

// Approximate matplotlib jet colormap (early=dark blue, late=dark red)
function jet(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 1)));
  return `rgb(${Math.round(r*255)}, ${Math.round(g*255)}, ${Math.round(b*255)})`;
}

function jetWithAlpha(t, a) {
  const c = jet(t);
  return c.replace('rgb(', 'rgba(').replace(')', `, ${a})`);
}

// ------- wedge polygon -------

function wedgePolygon(cx, cy, r, yawDeg, fovDeg, n = 24) {
  const xs = [cx];
  const ys = [cy];
  const a0 = (yawDeg - fovDeg / 2) * Math.PI / 180;
  const a1 = (yawDeg + fovDeg / 2) * Math.PI / 180;
  for (let i = 0; i <= n; i++) {
    const a = a0 + (a1 - a0) * i / n;
    xs.push(cx + r * Math.cos(a));
    ys.push(cy + r * Math.sin(a));
  }
  xs.push(cx);
  ys.push(cy);
  return { xs, ys };
}

// ------- plot building -------

function buildTraces(meta) {
  const traces = [];
  const frameFovIndices = [];

  // Trajectory (gray) — full episode path. Drawn first so colored layers go on top.
  if (meta.trajectory_xy && meta.trajectory_xy.length) {
    const xs = meta.trajectory_xy.map(p => p[0]);
    const ys = meta.trajectory_xy.map(p => p[1]);
    traces.push({
      x: xs, y: ys, type: 'scatter', mode: 'lines+markers',
      name: 'Trajectory',
      line: { color: 'rgba(110,110,110,0.85)', width: 1.2 },
      marker: { color: 'rgba(110,110,110,0.85)', size: 4 },
      hovertemplate: 'traj<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>',
    });
  }

  // In-window scatter colored by index → use Rainbow colorscale (red→violet)
  if (meta.in_window_xy && meta.in_window_xy.length) {
    const n = meta.in_window_xy.length;
    const xs = meta.in_window_xy.map(p => p[0]);
    const ys = meta.in_window_xy.map(p => p[1]);
    const ts = meta.in_window_xy.map((_, i) => i);
    // gray connector below scatter
    traces.push({
      x: xs, y: ys, type: 'scatter', mode: 'lines',
      line: { color: 'rgba(180,180,180,0.6)', width: 1 },
      hoverinfo: 'skip', showlegend: false,
    });
    traces.push({
      x: xs, y: ys, type: 'scatter', mode: 'markers',
      name: 'Trajectory [start(blue) → end(red)]',
      marker: {
        size: 7,
        color: ts,
        colorscale: 'Rainbow',
        cmin: 0, cmax: Math.max(1, n - 1),
        line: { width: 0 },
      },
      customdata: ts,
      hovertemplate: 'in-window #%{customdata}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>',
    });
  }

  // 6 frame wedges (jet)
  if (meta.frame_poses && meta.frame_poses.length) {
    const n = meta.frame_poses.length;
    meta.frame_poses.forEach(([fx, fy, fyaw], i) => {
      const t = n > 1 ? i / (n - 1) : 0;
      const fillColor = jetWithAlpha(t, 0.35);
      const lineColor = jet(t);
      const w = wedgePolygon(fx, fy, meta.frame_fov_radius, fyaw, meta.fov_deg);
      frameFovIndices.push(traces.length);
      traces.push({
        x: w.xs, y: w.ys, type: 'scatter', mode: 'lines',
        fill: 'toself', fillcolor: fillColor,
        line: { color: lineColor, width: 1 },
        name: i === 0 ? `Per Frame View [early(blue) → late(red)]` : '',
        showlegend: i === 0,
        legendgroup: 'frame_fov',
        hoverinfo: 'skip',
      });
    });
  }

  // GT FOV wedge (orange)
  if (meta.gt_caption && meta.gt_caption.yaw_deg !== null && meta.gt_caption.yaw_deg !== undefined) {
    const w = wedgePolygon(meta.gt_caption.x, meta.gt_caption.y,
      meta.fov_radius, meta.gt_caption.yaw_deg, meta.fov_deg);
    traces.push({
      x: w.xs, y: w.ys, type: 'scatter', mode: 'lines',
      fill: 'toself', fillcolor: 'rgba(255,165,0,0.2)',
      line: { color: 'rgba(255,140,0,0.9)', width: 1 },
      name: `Averaged Memory View (${meta.fov_deg.toFixed(0)}°)`,
      hoverinfo: 'skip',
    });
  }

  // GT diamond
  if (meta.gt_caption) {
    traces.push({
      x: [meta.gt_caption.x], y: [meta.gt_caption.y],
      type: 'scatter', mode: 'markers',
      name: 'Memory Centric GT (Remembr)',
      marker: { symbol: 'diamond', size: 12, color: 'orange', line: { color: '#a35400', width: 1 } },
      hovertemplate: `GT (avg)<br>x=%{x:.2f}<br>y=%{y:.2f}<br>yaw=${(meta.gt_caption.yaw_deg ?? 0).toFixed(1)}°<extra></extra>`,
    });
  }

  // Text-answer star
  if (meta.gt_truth_xyz && meta.gt_truth_xyz.length) {
    traces.push({
      x: meta.gt_truth_xyz.map(p => p[0]),
      y: meta.gt_truth_xyz.map(p => p[1]),
      type: 'scatter', mode: 'markers',
      name: 'Object Centric GT (DAAAM)',
      marker: { symbol: 'star', size: 16, color: 'red', line: { color: '#700', width: 1 } },
      hovertemplate: 'text-answer<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>',
    });
  }

  return { traces, frameFovIndices };
}

function buildArrowAnnotations(meta, includeFrames = true) {
  const ann = [];
  // Frame arrows (jet)
  if (includeFrames && meta.frame_poses && meta.frame_poses.length) {
    const n = meta.frame_poses.length;
    meta.frame_poses.forEach(([fx, fy, fyaw], i) => {
      const t = n > 1 ? i / (n - 1) : 0;
      const yawRad = fyaw * Math.PI / 180;
      const r = meta.frame_fov_radius;
      ann.push({
        x: fx + r * Math.cos(yawRad), y: fy + r * Math.sin(yawRad),
        ax: fx, ay: fy,
        xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
        showarrow: true, arrowhead: 3, arrowsize: 1, arrowwidth: 1.5,
        arrowcolor: jet(t),
        standoff: 0,
      });
    });
  }
  // GT arrow (orange)
  if (meta.gt_caption && meta.gt_caption.yaw_deg !== null && meta.gt_caption.yaw_deg !== undefined) {
    const yawRad = meta.gt_caption.yaw_deg * Math.PI / 180;
    const r = meta.fov_radius;
    ann.push({
      x: meta.gt_caption.x + r * Math.cos(yawRad),
      y: meta.gt_caption.y + r * Math.sin(yawRad),
      ax: meta.gt_caption.x, ay: meta.gt_caption.y,
      xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
      showarrow: true, arrowhead: 3, arrowsize: 1.2, arrowwidth: 2,
      arrowcolor: 'rgba(255,140,0,0.95)',
      standoff: 0,
    });
  }
  return ann;
}

function plotRow(meta) {
  state.currentMeta = meta;
  const { traces, frameFovIndices } = buildTraces(meta);
  state.frameFovTraceIndices = frameFovIndices;
  const includeFrames = els.fovToggle.checked;
  // Hide frame wedges on initial render if toggle is off
  if (!includeFrames) {
    frameFovIndices.forEach(i => { traces[i].visible = false; });
  }
  const annotations = buildArrowAnnotations(meta, includeFrames);
  const titlePrefix = meta.in_window ? '' : '[SNAPPED] ';
  const layout = {
    title: { text: `${titlePrefix}${meta.question}<br><span style="font-size:11px;color:#888">${meta.uuid}</span>`, font: { size: 14 } },
    xaxis: { title: 'x', zeroline: false, scaleanchor: 'y', scaleratio: 1 },
    yaxis: { title: 'y', zeroline: false },
    hovermode: 'closest',
    margin: { l: 60, r: 30, t: 70, b: 50 },
    legend: { font: { size: 11 } },
    annotations,
  };
  Plotly.newPlot(els.plot, traces, layout, {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,            // mouse wheel: up = zoom in, down = zoom out
    doubleClick: 'reset',         // double-click to reset zoom
  }).then(() => {
    attachCursorReadout();
  });
}

// ------- cursor coord overlay (works over empty space too) -------

function attachCursorReadout() {
  detachCursorReadout();
  const gd = els.plot;
  const drag = gd.querySelector('.nsewdrag');
  if (!drag) return;
  const update = (evt) => {
    if (!els.cursorToggle.checked) return;
    const rect = drag.getBoundingClientRect();
    const px = evt.clientX - rect.left;
    const py = evt.clientY - rect.top;
    const xa = gd._fullLayout.xaxis;
    const ya = gd._fullLayout.yaxis;
    const x = xa.p2c(px);
    const y = ya.p2c(py);
    els.cursorOverlay.textContent = `x, y = ${x.toFixed(2)}, ${y.toFixed(2)}`;
  };
  const leave = () => { els.cursorOverlay.textContent = 'x, y = —'; };
  drag.addEventListener('mousemove', update);
  drag.addEventListener('mouseleave', leave);
  state.cursorMoveHandler = { drag, update, leave };
  applyCursorVisibility();
}

function detachCursorReadout() {
  if (!state.cursorMoveHandler) return;
  const { drag, update, leave } = state.cursorMoveHandler;
  drag.removeEventListener('mousemove', update);
  drag.removeEventListener('mouseleave', leave);
  state.cursorMoveHandler = null;
}

function applyCursorVisibility() {
  els.cursorOverlay.classList.toggle('hidden', !els.cursorToggle.checked);
}

function applyFrameFovVisibility() {
  if (!state.currentMeta || !state.frameFovTraceIndices.length) return;
  const visible = els.fovToggle.checked;
  Plotly.restyle(els.plot, { visible: visible ? true : false }, state.frameFovTraceIndices);
  Plotly.relayout(els.plot, { annotations: buildArrowAnnotations(state.currentMeta, visible) });
}

// ------- frames panel -------

function renderFrames(meta) {
  els.frames.innerHTML = '';
  (meta.frames || []).forEach((path, i) => {
    const cell = document.createElement('div');
    cell.className = 'frame-cell';
    if (path) {
      const img = document.createElement('img');
      img.loading = 'lazy';
      img.src = path;
      img.alt = `frame ${i}`;
      cell.appendChild(img);
    } else {
      const ph = document.createElement('div');
      ph.style.height = '120px';
      ph.style.background = '#fee';
      ph.style.color = '#a00';
      ph.style.display = 'flex';
      ph.style.alignItems = 'center';
      ph.style.justifyContent = 'center';
      ph.style.fontSize = '11px';
      ph.textContent = `[${i}] missing`;
      cell.appendChild(ph);
    }
    const title = document.createElement('div');
    title.className = 'frame-title';
    const name = path ? path.split('/').pop() : `[${i}] missing`;
    title.textContent = `[${i}] ${name}`;
    cell.appendChild(title);
    els.frames.appendChild(cell);
  });
}

// ------- info panel -------

function renderInfo(meta) {
  const lines = [];
  const c = (cls, text) => `<span class="${cls}">${text}</span>`;
  lines.push(`${c('k', 'Q:')} ${meta.question}`);
  lines.push(`${c('k', 'UUID:')} ${meta.uuid}`);
  lines.push(`${c('k', 'CSV HMS:')} ${meta.csv_hms}  →  gt_epoch=${meta.gt_epoch.toFixed(0)} (${meta.gt_iso})`);
  lines.push(`${c('k', 'Window UTC:')} ${meta.window_iso[0]}  →  ${meta.window_iso[1]}`);
  if (meta.status === 'in_window') {
    lines.push(`${c('ok', 'RESULT: in-window')}  caption_idx=${meta.caption_idx}  caption_t=${meta.caption_time.toFixed(0)}`);
  } else if (meta.status === 'snapped') {
    lines.push(`${c('warn', 'RESULT: snapped')}  caption_idx=${meta.caption_idx}  caption_t=${meta.caption_time.toFixed(0)}  gt_offset_from_window=${meta.snap_offset_seconds.toFixed(0)}s`);
  } else {
    lines.push(`${c('err', 'RESULT: ' + meta.status)}`);
  }
  if (meta.yaw_check) {
    const yc = meta.yaw_check;
    const fmt = v => v == null ? '—' : v.toFixed(1) + '°';
    lines.push(`${c('k', 'Yaw check:')} theta[0]=${fmt(yc.theta0_deg)}, true_yaw(wxyz)=${fmt(yc.true_yaw_deg)}, motion_heading=${fmt(yc.motion_heading_deg)}`);
  }
  lines.push(`${c('k', 'Used files:')}`);
  (meta.frames || []).forEach((p, i) => {
    lines.push(`  [${i}] ${p ?? '(missing)'}`);
  });
  els.info.innerHTML = lines.join('\n');
}

// ------- row navigation -------

async function loadRow(idx) {
  if (idx == null) return;
  state.currentIdx = idx;
  els.picker.value = String(idx);
  document.querySelectorAll('.row-item').forEach(el => {
    el.classList.toggle('active', Number(el.dataset.idx) === idx);
  });
  updateHash(idx);
  els.info.textContent = `Loading row ${idx}…`;
  let meta;
  try {
    const resp = await fetch(`meta/${idx}.json`, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    meta = await resp.json();
  } catch (e) {
    els.info.textContent = `Failed to load row ${idx}: ${e.message}`;
    Plotly.purge(els.plot);
    els.frames.innerHTML = '';
    return;
  }
  plotRow(meta);
  renderFrames(meta);
  renderInfo(meta);
}

function updateHash(idx) {
  const h = `#row=${idx}`;
  if (location.hash !== h) history.replaceState(null, '', h);
}

function readHashRow() {
  const m = location.hash.match(/row=(\d+)/);
  return m ? Number(m[1]) : null;
}

function neighborIdx(delta) {
  if (state.index.length === 0) return null;
  const ids = state.index.map(r => r.idx);
  const cur = state.currentIdx;
  const pos = ids.indexOf(cur);
  if (pos === -1) return ids[0];
  const next = pos + delta;
  if (next < 0 || next >= ids.length) return cur;
  return ids[next];
}

// ------- bootstrap -------

async function init() {
  // load index
  let idxData;
  try {
    const resp = await fetch('index.json', { cache: 'no-store' });
    idxData = await resp.json();
  } catch (e) {
    els.info.textContent = `Failed to load index.json: ${e.message}`;
    return;
  }
  state.index = idxData;

  // populate picker + row list
  els.picker.innerHTML = '';
  els.list.innerHTML = '';
  idxData.forEach(r => {
    const opt = document.createElement('option');
    opt.value = String(r.idx);
    opt.textContent = `[${r.idx}] seq ${r.seq_id} — ${r.question.slice(0, 50)}`;
    els.picker.appendChild(opt);

    const li = document.createElement('div');
    li.className = 'row-item';
    li.dataset.idx = String(r.idx);
    const badge = `<span class="badge ${r.status}">${r.status === 'in_window' ? '✓' : r.status === 'snapped' ? '~' : '!'}</span>`;
    li.innerHTML = `${badge}<b>[${r.idx}]</b> seq ${r.seq_id}<br><span style="color:#666">${r.question.slice(0, 70)}</span>`;
    li.onclick = () => loadRow(r.idx);
    els.list.appendChild(li);
  });

  els.picker.onchange = (e) => loadRow(Number(e.target.value));
  els.prev.onclick = () => loadRow(neighborIdx(-1));
  els.next.onclick = () => loadRow(neighborIdx(+1));

  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft') { e.preventDefault(); loadRow(neighborIdx(-1)); }
    if (e.key === 'ArrowRight') { e.preventDefault(); loadRow(neighborIdx(+1)); }
  });

  els.cursorToggle.onchange = applyCursorVisibility;
  els.fovToggle.onchange = applyFrameFovVisibility;

  window.addEventListener('hashchange', () => {
    const idx = readHashRow();
    if (idx != null && idx !== state.currentIdx) loadRow(idx);
  });

  const startIdx = readHashRow() ?? (idxData[0] && idxData[0].idx);
  if (startIdx != null) loadRow(startIdx);
}

init();
