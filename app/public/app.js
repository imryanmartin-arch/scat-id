"use strict";
const $ = (s) => document.querySelector(s);
const state = { img: null, cx: .5, cy: .5, zoom: 1, size: null, shape: null, probs: null, classes: null, session: null, busy: false, pending: false };

// ---------- offline + model ----------
const status = (t, cls = "") => { const s = $("#status"); s.textContent = t; s.className = "pill " + cls; };

async function boot() {
  if ("serviceWorker" in navigator) {
    try {
      const hadController = !!navigator.serviceWorker.controller;
      const reg = await navigator.serviceWorker.register("sw.js");
      reg.update().catch(() => {});
      // A new version finished downloading: reload once so it's used (not on first install).
      navigator.serviceWorker.addEventListener("controllerchange", () => { if (hadController) location.reload(); });
    } catch (e) { console.warn(e); }
  }
  try { if (navigator.storage && navigator.storage.persist) await navigator.storage.persist(); } catch (e) {}
  try {
    status("Loading model…");
    ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;
    ort.env.wasm.numThreads = 1;
    const meta = await (await fetch("model/model.json")).json();
    state.classes = meta.classes;
    state.session = await ort.InferenceSession.create("model/scat.onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    status(await offlineReady() ? "Ready · works offline" : "Ready", "ok");
    if (!(await offlineReady())) setTimeout(async () => { if (await offlineReady()) status("Ready · works offline", "ok"); }, 8000);
  } catch (e) {
    console.error(e); status("Model failed to load", "bad");
  }
}
async function offlineReady() {
  try { return !!(window.caches && await caches.match("model/scat.onnx")); } catch (e) { return false; }
}

// ---------- photo + crop ----------
function loadFile(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    state.img = img; state.cx = .5; state.cy = .5; state.zoom = 1; $("#zoom").value = 100;
    state.size = null; state.shape = null; renderChips();
    $("#start").hidden = true; $("#work").hidden = false;
    draw(); classify();
  };
  img.src = url;
}
$("#cam").onchange = (e) => loadFile(e.target.files[0]);
$("#lib").onchange = (e) => loadFile(e.target.files[0]);
$("#again").onclick = () => { $("#work").hidden = true; $("#start").hidden = false; $("#cam").value = ""; $("#lib").value = ""; window.scrollTo(0, 0); };

function cropRect() {
  const { img, cx, cy, zoom } = state, W = img.naturalWidth, H = img.naturalHeight;
  const s = Math.min(W, H) * zoom;
  const x = Math.min(Math.max(cx * W - s / 2, 0), W - s), y = Math.min(Math.max(cy * H - s / 2, 0), H - s);
  return { x, y, s };
}
function draw() {
  const c = $("#view"), img = state.img, W = img.naturalWidth, H = img.naturalHeight;
  const scale = Math.min(1, 1200 / Math.max(W, H));
  c.width = Math.round(W * scale); c.height = Math.round(H * scale);
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0, c.width, c.height);
  const r = cropRect();
  g.fillStyle = "rgba(0,0,0,.55)";
  g.beginPath(); g.rect(0, 0, c.width, c.height);
  g.rect(r.x * scale, r.y * scale + r.s * scale, r.s * scale, -r.s * scale); g.fill("evenodd");
  g.strokeStyle = "#e0a43a"; g.lineWidth = Math.max(2, c.width / 250);
  g.strokeRect(r.x * scale, r.y * scale, r.s * scale, r.s * scale);
}
$("#view").addEventListener("click", (e) => {
  const b = e.target.getBoundingClientRect();
  state.cx = (e.clientX - b.left) / b.width; state.cy = (e.clientY - b.top) / b.height;
  if (state.zoom === 1) { state.zoom = .6; $("#zoom").value = 60; }
  draw(); classify();
});
$("#zoom").oninput = (e) => { state.zoom = e.target.value / 100; draw(); };
$("#zoom").onchange = () => classify();

// Square crop -> 224x224 RGB in [0,1], plus mirrored copy (test-time augmentation).
function toTensor() {
  const r = cropRect();
  const mid = document.createElement("canvas"); mid.width = mid.height = 448;
  const m = mid.getContext("2d"); m.imageSmoothingQuality = "high";
  m.drawImage(state.img, r.x, r.y, r.s, r.s, 0, 0, 448, 448);
  const c = document.createElement("canvas"); c.width = c.height = 224;
  const g = c.getContext("2d"); g.imageSmoothingQuality = "high";
  g.drawImage(mid, 0, 0, 224, 224);
  const d = g.getImageData(0, 0, 224, 224).data, N = 224 * 224;
  const out = new Float32Array(2 * 3 * N);
  for (let y = 0; y < 224; y++) for (let x = 0; x < 224; x++) {
    const i = (y * 224 + x) * 4, p = y * 224 + x, q = y * 224 + (223 - x);
    for (let ch = 0; ch < 3; ch++) {
      const v = d[i + ch] / 255;
      out[ch * N + p] = v;               // original
      out[3 * N + ch * N + q] = v;       // mirrored
    }
  }
  return new ort.Tensor("float32", out, [2, 3, 224, 224]);
}

async function classify() {
  if (!state.session) { $("#results").innerHTML = `<div class="spinner">Model still loading…</div>`; setTimeout(classify, 800); return; }
  if (state.busy) { state.pending = true; return; }
  state.busy = true;
  $("#results").innerHTML = `<div class="spinner">Analyzing…</div>`;
  await new Promise((r) => setTimeout(r, 30));
  try {
    const out = await state.session.run({ image: toTensor() });
    const p = out.probs.data, K = state.classes.length;
    state.probs = state.classes.map((_, k) => (p[k] + p[K + k]) / 2);
    renderResults();
  } catch (e) {
    console.error(e); $("#results").innerHTML = `<div class="alert">Something went wrong analyzing this photo.</div>`;
  }
  state.busy = false;
  if (state.pending) { state.pending = false; classify(); }
}

// ---------- combine model + hints + local abundance ----------
function ranked() {
  const scores = state.classes.map((k, i) => {
    const sp = SPECIES[k] || SPECIES.notscat;
    let s = state.probs[i] * Math.sqrt(sp.local);           // gentle local prior
    if (state.size) s *= sp.size[state.size];
    if (state.shape) s *= sp.shape[state.shape];
    return { k, s };
  });
  const tot = scores.reduce((a, b) => a + b.s, 0) || 1;
  return scores.map((x) => ({ k: x.k, p: x.s / tot })).sort((a, b) => b.p - a.p);
}

function card(k, p, top) {
  const sp = SPECIES[k];
  return `<div class="result ${top ? "top" : ""}">
    <div class="result-head"><span class="result-name">${sp.name}</span><span class="result-pct">${Math.round(p * 100)}%</span></div>
    <div class="bar"><i style="width:${Math.max(2, p * 100)}%"></i></div>
    <details ${top ? "open" : ""}><summary>What it looks like</summary>
      <p>${sp.looks}</p>
      ${sp.season ? `<p><b>Season:</b> ${sp.season}</p>` : ""}
      ${sp.where ? `<p><b>Where:</b> ${sp.where}</p>` : ""}
      <p><b>Look-alikes:</b> ${sp.lookalikes}</p>
    </details></div>`;
}

function renderResults() {
  const r = ranked(), top = r.slice(0, 3);
  let html = "";
  if (r[0].k === "notscat") html += `<div class="alert">This doesn't look much like scat. Try a closer, sharper photo with the droppings filling the box.</div>`;
  const wolf = r.find((x) => x.k === "wolf");
  if (wolf && wolf.p > .15) html += `<div class="alert">🐺 ${SPECIES.wolf.alert}</div>`;
  if (r[0].p < .4 && r[0].k !== "notscat") html += `<p class="note">Low confidence — these are the most likely candidates. Adding size and shape below helps a lot.</p>`;
  html += top.filter((x) => x.k !== "notscat" || x === r[0]).map((x, i) => card(x.k, x.p, i === 0)).join("");
  $("#results").innerHTML = html;
}

function renderChips() {
  const mk = (list, key, el) => {
    $(el).innerHTML = list.map((o) => `<button class="chip ${state[key] === o.key ? "on" : ""}" data-v="${o.key}">${o.label}<small>${o.hint}</small></button>`).join("");
    $(el).querySelectorAll(".chip").forEach((b) => b.onclick = () => {
      state[key] = state[key] === b.dataset.v ? null : b.dataset.v; renderChips(); if (state.probs) renderResults();
    });
  };
  mk(SIZES, "size", "#sizes"); mk(SHAPES, "shape", "#shapes");
}

// ---------- log (IndexedDB) ----------
function db() {
  return new Promise((res, rej) => {
    const q = indexedDB.open("scat-log", 1);
    q.onupgradeneeded = () => q.result.createObjectStore("finds", { keyPath: "id" });
    q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error);
  });
}
async function tx(mode, fn) {
  const d = await db();
  return new Promise((res, rej) => {
    const t = d.transaction("finds", mode), s = t.objectStore("finds"), r = fn(s);
    t.oncomplete = () => res(r && r.result); t.onerror = () => rej(t.error);
  });
}
function getPos() {
  return new Promise((res) => {
    if (!navigator.geolocation) return res(null);
    navigator.geolocation.getCurrentPosition((p) => res({ lat: p.coords.latitude, lon: p.coords.longitude, acc: Math.round(p.coords.accuracy) }),
      () => res(null), { enableHighAccuracy: true, timeout: 15000, maximumAge: 60000 });
  });
}
$("#save").onclick = async () => {
  if (!state.probs) return;
  const btn = $("#save"); btn.disabled = true; btn.textContent = "Getting GPS…";
  const r = cropRect(), c = document.createElement("canvas"); c.width = c.height = 320;
  c.getContext("2d").drawImage(state.img, r.x, r.y, r.s, r.s, 0, 0, 320, 320);
  const pos = await getPos();
  const entry = { id: Date.now(), t: new Date().toISOString(), thumb: c.toDataURL("image/jpeg", .8), pos,
    size: state.size, shape: state.shape, top: ranked().slice(0, 3) };
  await tx("readwrite", (s) => s.put(entry));
  btn.textContent = "Saved ✓"; setTimeout(() => { btn.textContent = "Save to log"; btn.disabled = false; }, 1500);
};
async function renderLog() {
  const all = (await tx("readonly", (s) => s.getAll())) || [];
  all.sort((a, b) => b.id - a.id);
  $("#log").innerHTML = all.length ? all.map((e) => `<div class="entry">
      <img src="${e.thumb}" alt="">
      <div><div class="who">${SPECIES[e.top[0].k].name} · ${Math.round(e.top[0].p * 100)}%</div>
        <div class="meta">${e.top.slice(1).map((x) => `${SPECIES[x.k].name} ${Math.round(x.p * 100)}%`).join(" · ")}</div>
        <div class="meta">${new Date(e.t).toLocaleString()}</div>
        <div class="meta">${e.pos ? `<a href="https://maps.apple.com/?ll=${e.pos.lat},${e.pos.lon}&q=Scat">${e.pos.lat.toFixed(5)}, ${e.pos.lon.toFixed(5)}</a> ±${e.pos.acc}m` : "No GPS"}
          <button data-del="${e.id}">Delete</button></div></div></div>`).join("")
    : `<div class="empty">No saved finds yet.</div>`;
  $("#log").querySelectorAll("[data-del]").forEach((b) => b.onclick = async () => {
    if (!confirm("Delete this find?")) return;
    await tx("readwrite", (s) => s.delete(Number(b.dataset.del))); renderLog();
  });
}

// ---------- guide ----------
function renderGuide() {
  $("#guide").innerHTML = Object.entries(SPECIES).filter(([k]) => k !== "notscat").map(([k, sp]) => `<div class="guide-item">
    <h3>${sp.name}</h3>
    ${PHOTOS[k] ? `<figure><img src="guide/${k}.jpg" alt="${sp.name} scat" loading="lazy">
      <figcaption><a href="${PHOTOS[k].obs}">${PHOTOS[k].credit}</a></figcaption></figure>` : ""}
    <p>${sp.looks}</p><p><b>Season:</b> ${sp.season}</p><p><b>Where:</b> ${sp.where}</p><p><b>Look-alikes:</b> ${sp.lookalikes}</p></div>`).join("") +
    `<p class="fine">Model trained on ~23,000 Creative Commons scat photos from iNaturalist contributors —
     <a href="https://github.com/imryanmartin-arch/scat-id/blob/main/training/credits.csv" style="color:var(--accent)">photo credits</a>.
     Non-commercial use only. Report possible wolf sign to ODFW.</p>`;
}

// ---------- tabs ----------
document.querySelectorAll("nav button").forEach((b) => b.onclick = () => {
  document.querySelectorAll("nav button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "log") renderLog();
  window.scrollTo(0, 0);
});

// ---------- install prompt (only when opened in a browser tab) ----------
const standalone = navigator.standalone || matchMedia("(display-mode: standalone)").matches;
let skipped = false; try { skipped = localStorage.getItem("skipInstall") === "1"; } catch (e) {}
if (!standalone && !skipped) { $("#install").hidden = false; $("#start").hidden = true; }
$("#skip-install").onclick = () => {
  try { localStorage.setItem("skipInstall", "1"); } catch (e) {}
  $("#install").hidden = true; $("#start").hidden = false;
};

renderChips(); renderGuide(); boot();
