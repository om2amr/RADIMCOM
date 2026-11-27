// ---------- DECODE › ATC (atc.py): speech to text of the airband voice; the aircraft whose callsign is heard is
// selected on the COMPOSITE map. The key blinks while a transmission is transcribed (loading the model: slowly). ----------
// Loaded after app.js and uses its globals ($, st, vbase, escHtml, renderAd). app.js calls atcStatus from each state
// of the VFO, atcText from the "atc" event (one transcribed transmission) and atcAdOn / atcAdRender from renderAd
// (nothing in the AD now). The transcripts: GET /api/atc (history).

var atcLast = null;                // the last transmission of the active VFO: {t, hz, text, calls}

/** Airband AM: the only place where the key is shown. */
function atcHere(s) {
  return !!s && s.mode === "AM" && s.tune_hz >= 117975000 && s.tune_hz <= 137000000 && !s.dab;
}

function atcStatus(s) {
  const k = $("atcKey"), here = atcHere(s);
  k.hidden = false;
  // only in AM on the airband; elsewhere shown, but inactive (greyed, not disabled: a disabled key shows no tooltip)
  k.classList.toggle("atc-off", !here);
  k.setAttribute("aria-disabled", here ? "false" : "true");
  k.classList.toggle("on", here && !!s.atc);
  k.classList.toggle("atc-load", !!s.atc && s.atc_state === "loading");
  k.classList.toggle("atc-err", !!s.atc && s.atc_state === "error");
  k.classList.toggle("atc-busy", here && !!s.atc && s.atc_state === "busy");   // transcribing: blinks
  // the tooltip: its own text kept apart (tooltip.js takes title off while the mouse is over), written only on a change
  if (k.dataset.tip0 == null) k.dataset.tip0 = k.getAttribute("title") || "";
  const tip = k.dataset.tip0 + (s.atc && s.atc_state === "error" ? `\n\nNow: ${s.atc_error || "error"}` : "");
  if (k._tip !== tip) { k._tip = tip; k.title = tip; }
  atcDbgStatus(s);
}

/** ATC on / off for this VFO (the ATC key and DEBUG › ATC-AI › START / STOP). */
async function atcSet(on) {
  st.atc = on;
  $("atcKey").classList.toggle("on", on);
  if (!on) atcLast = null;
  atcDbgStatus(st);
  try {
    await fetch(vbase + "/api/atc", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ on }) });
  } catch { /* the state from the server corrects the key */ }
}

$("atcKey").addEventListener("click", () => {
  if (st && atcHere(st)) atcSet(!st.atc);
});

// ----- COMPOSITE: the aircraft of the callsign heard is selected (red, its details on the right) -----

/** "RYR028KK" / "RYR 28KK" -> "RYR28KK" (leading zeros of the number dropped, as they are not spoken). */
function atcNorm(f) {
  const s = String(f || "").toUpperCase().replace(/\s+/g, "");
  const m = /^([A-Z]{3})0*([0-9A-Z]+)$/.exec(s);
  return m ? m[1] + m[2] : s;
}

/** The COMPOSITE object of an ICAO callsign (its flight as ADS-B / FR24 / adsb.fi / VDL2 give it), or null. */
function atcComFind(cs) {
  if (typeof comMerged === "undefined" || !comMerged.length) return null;
  const want = atcNorm(cs);
  return comMerged.find((m) => m.air && [m.air.flight, m.label].some((f) => f && (atcNorm(f) === want ||
    (typeof comFlight === "function" && atcNorm(comFlight(f)) === want)))) || null;
}

/** Selects the aircraft on the COMPOSITE map; the map does not move (pan = moves it there). */
function atcComSelect(cs, pan) {
  const m = atcComFind(cs);
  if (!m) return false;
  comSel = m.id;
  if (comMap && pan) comMap.panTo([m.lat, m.lon]);
  renderComMap();
  return true;
}

/** The COMPOSITE map loads its aircraft only while it is open (comShown): otherwise "on the map" is not known. */
function atcMapKnown() {
  return typeof comTimer !== "undefined" && !!comTimer;
}

/** The ICAO callsigns of the aircraft on the COMPOSITE map (normalised by atcNorm). */
function atcMapCalls() {
  const out = new Set();
  if (typeof comMerged === "undefined") return out;
  for (const m of comMerged) {
    if (!m.air) continue;
    for (const f of [m.air.flight, m.label]) {
      const n = atcNorm(f && typeof comFlight === "function" ? comFlight(f) || f : f);
      if (/^[A-Z]{3}/.test(n)) out.add(n);
    }
  }
  return out;
}

/** Letters and digits the speech recognition mixes up ("zulu zulu" heard as "zero zero"). */
function atcSound(s) {
  return s.replace(/Z/g, "0");
}

/** The letters a in this order in b ("FL" in "FTL"): a designator spelled with a letter swallowed. */
function atcLettersIn(a, b) {
  let i = 0;
  for (const ch of b) if (ch === a[i]) i++;
  return i === a.length;
}

function atcLev(a, b) {
  let p = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const c = [i];
    for (let j = 1; j <= b.length; j++) c.push(Math.min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (a[i - 1] !== b[j - 1])));
    p = c;
  }
  return p[b.length];
}

/** How well the heard flight number h fits the number m of an aircraft on the map (0 = not at all). */
function atcSufScore(h, m) {
  if (!h || !m) return 0;
  if (h === m) return 10;
  if (atcSound(h) === atcSound(m)) return 9;                    // RYR900 heard for RYR9ZZ
  if (h.length < 2) return 0;
  if (m.endsWith(h)) return 4 + Math.min(h.length, 4);          // the start lost: "234" of 1234
  if (m.startsWith(h)) return 3 + Math.min(h.length, 4);       // the end lost: "123" of 1234, "5K" of 5KE
  if (h.length >= 3 && m.length >= 3 && atcLev(atcSound(h), atcSound(m)) === 1) return 6;   // one character wrong
  if (h.length >= 3 && m.includes(h)) return 5;
  return 0;
}

/** How well the heard airline letters p (3 from the telephony or spelled, 2 spelled with one swallowed) fit d. */
function atcPreScore(p, d) {
  if (!p) return 0;
  if (p === d) return 6;
  let same = 0;
  for (let i = 0; i < 3; i++) if (p[i] && p[i] === d[i]) same++;
  return Math.max(2 * same, p.length === 2 && atcLettersIn(p, d) ? 4 : 0);
}

var ATC_MATCH_MIN = 9;             // the score of a partial match to be taken (a whole number alone is 10)

/** The aircraft on the map that fits a heard callsign best, even partly (AC.234 for AB1234: the number is heard
 *  better than the airline, and an aircraft that is there is far more likely than one that is not):
 *  [{cs, score, alt}]: the best one, or the two that fit alike (alt), or [] when none is good enough. */
function atcBestMatch(c) {
  const heard = c.cs ? atcNorm(c.cs) : "";
  const pre = heard ? heard.slice(0, 3) : c.pre || "";
  const suf = heard ? heard.slice(3) : String(c.suf || "").replace(/^0+(?=.)/, "");
  const scores = [];
  for (const n of atcMapCalls()) {
    const s = atcSufScore(suf, n.slice(3));
    if (!s) continue;
    if (suf.length < 2 && n.slice(0, 3) !== pre) continue;      // one digit ("ups three"): the airline must be the same
    scores.push([n, s + atcPreScore(pre, n.slice(0, 3))]);
  }
  scores.sort((a, b) => b[1] - a[1]);
  if (!scores.length || scores[0][1] < ATC_MATCH_MIN) return [];
  // two alike: both (one of them is very likely it); more than two alike: none
  const close = scores.filter((s) => s[1] >= ATC_MATCH_MIN && scores[0][1] - s[1] < 2);
  return close.length <= 2 ? close.map(([cs, score]) => ({ cs, score, alt: close.length > 1 })) : [];
}

/** A callsign not on the map, or a number said alone ("four one seven eight"), joined to the aircraft on the map it
 *  fits best, even partly (atcBestMatch): no "Scandinavian 4178" guessed when Ryanair 4178 is there.
 *  c.heard keeps what was recognised; a number without such an aircraft is dropped. */
function atcResolve(d) {
  const out = [];
  for (const c of d.calls || []) {
    if (c.cs && atcComFind(c.cs)) { out.push(c); continue; }
    const bs = atcBestMatch(c);
    for (const b of bs) out.push({ ...c, heard: c.cs || (c.pre || "") + c.suf, cs: b.cs, exact: false, bynum: true, alt: b.alt });
    if (!bs.length && !c.num && !c.short) out.push(c);   // one digit only ("ups three"): only an aircraft on the map
  }
  d.calls = out;
}

/** One transmission from the server. */
function atcText(d) {
  atcResolve(d);
  // the first callsign found on the map (an exact one first) is selected there
  const calls = [...(d.calls || [])].sort((a, b) => b.exact - a.exact);
  d.onMap = calls.map((c) => c.cs).filter((cs) => atcComFind(cs));
  d.mapKnown = atcMapKnown();
  if (d.onMap.length) atcComSelect(d.onMap[0], false);
  atcLast = d;
  atcDbgAdd(d);
}

// ----- SETTINGS › DEBUG › ATC-AI: every transcript of this VFO whole: UTC, MHz, the text → the callsign found -----

var ATC_DBG_MAX = 300;             // rows kept on the page
var atcDbgBase = null;             // the VFO whose history is shown (vbase)
var atcDbgCut = 0;                 // CLR: rows up to this id are not shown again

function atcDbgVisible() {
  const p = document.querySelector('[data-set-subpanel="dbg:atc"]');
  return !!p && !p.hidden && !p.closest("[hidden]");
}

/** One row: time, frequency, the whole text, the callsign(s) at the end (matched to an aircraft on the map: ≈). */
function atcDbgRow(d) {
  const t = new Date(d.t * 1000).toISOString().slice(11, 19);
  const calls = (d.calls || []).filter((c) => c.cs);
  const cs = calls.map((c) => {
    const tip = c.bynum ? `heard ${c.heard || "?"}, the aircraft on the map that fits best${c.alt ? " (one of two alike)" : ""}`
      : c.exact ? "said whole" : "found in the text";
    return `<span class="atc-dbg-cs${c.bynum ? " near" : ""}" title="${escHtml(tip)}">${c.bynum ? "≈" : ""}${escHtml(c.cs)}</span>`;
  }).join(" ");
  const map = !calls.length ? '<span class="muted" title="no callsign to look for">—</span>'
    : !d.mapKnown ? '<span class="muted" title="The COMPOSITE map was not open at that time: its aircraft were not loaded, so it is not known">map ?</span>'
    : (d.onMap || []).length ? `<span class="atc-dbg-yes" title="On the COMPOSITE map: ${escHtml(d.onMap.join(", "))}">map yes</span>`
    : '<span class="atc-dbg-no" title="No aircraft with this callsign on the COMPOSITE map (not heard by ADS-B / FR24 / the other sources)">map no</span>';
  const info = `${d.dur} s of voice, ${d.rms} dBFS, transcribed in ${d.cpu} s${d.opt ? `\nWhisper: ${d.opt}` : ""}`;
  return `<div class="atc-dbg-row" data-id="${d.id}"><span class="atc-dbg-t" title="${escHtml(info)}">${t}</span>` +
    `<span class="atc-dbg-f">${(d.hz / 1e6).toFixed(3)}</span><span class="atc-dbg-x">${escHtml(d.text)}</span>` +
    `<span class="atc-dbg-c">→ ${cs || '<span class="muted" title="no callsign found in the text">—</span>'}</span>` +
    `<span class="atc-dbg-m">${map}</span></div>`;
}

/** A new transcript (the "atc" event): on top of the list. */
function atcDbgAdd(d) {
  const box = $("atcDbgList");
  if (!box || atcDbgBase !== vbase || d.id <= atcDbgCut) return;
  box.querySelector(".atc-dbg-empty")?.remove();
  box.insertAdjacentHTML("afterbegin", atcDbgRow(d));
  while (box.children.length > ATC_DBG_MAX) box.lastElementChild.remove();
}

/** The sub-page was opened (or the VFO changed): the history of this VFO from the server. */
async function atcDbgOpened() {
  atcDbgBase = vbase;
  atcDbgStatus(st);
  let h = [], r = null;
  try { r = await fetch(vbase + "/api/atc").then((x) => x.json()); h = r.history || []; } catch { /* the server is away */ }
  if (atcDbgBase !== vbase) return;
  if (r && r.settings) { atcSetDefaults = r.defaults || atcSetDefaults; atcSetShow(r.settings); }
  if (h.length && h[h.length - 1].id < atcDbgCut) atcDbgCut = 0;     // the server started again: its ids from 1
  h = h.filter((d) => d.id > atcDbgCut);
  for (const d of h) {             // the map of now (the history has no "on map" of its time)
    atcResolve(d);
    d.onMap = (d.calls || []).map((c) => c.cs).filter((cs) => cs && atcComFind(cs));
    d.mapKnown = atcMapKnown();
  }
  $("atcDbgList").innerHTML = h.length ? h.reverse().map(atcDbgRow).join("")
    : '<div class="atc-dbg-empty muted">No transcript yet. START, then tune an airband channel in AM with the squelch set.</div>';
}

/** The START / STOP key, the state and the border of the block from the state of the VFO. */
function atcDbgStatus(s) {
  const k = $("atcDbgRun");
  if (!k || !s) return;
  if (atcDbgBase !== null && atcDbgBase !== vbase && atcDbgVisible()) { atcDbgCut = 0; atcDbgOpened(); return; }
  const here = atcHere(s), on = !!s.atc, err = on && s.atc_state === "error";
  k.textContent = on ? "STOP" : "START";
  k.classList.toggle("on", on);
  const txt = !on ? "off" : err ? `error: ${s.atc_error || "?"}` : !here ? "waiting: only in AM on the airband 118–137 MHz"
    : { loading: "loading the model…", busy: "transcribing…", ready: "listening" }[s.atc_state] || s.atc_state || "";
  const el = $("atcDbgState");
  if (el.textContent !== txt) { el.textContent = txt; el.title = txt; }
  const b = $("atcDbgB");
  b.classList.toggle("lit", on && !err && here);
  b.classList.toggle("warn", on && !err && !here);
  b.classList.toggle("bad", err);
}

$("atcDbgRun").addEventListener("click", () => { if (st) atcSet(!st.atc); });

// ----- the Whisper settings (atc.py transcribe_args, data\atc.json): taken at the next transcript, on every VFO -----

var atcSetDefaults = null;

function atcSetShow(s) {
  $("atcSetBeam").value = String(s.beam);
  $("atcSetFallback").checked = !!s.fallback;
  $("atcSetNoTs").checked = !!s.no_ts;
  if (document.activeElement !== $("atcSetGap")) atcSetGapShow(s.gap);
  $("atcSetPromptOn").checked = !!s.prompt_on;
  if (document.activeElement !== $("atcSetPrompt")) $("atcSetPrompt").value = s.prompt || "";
  $("atcSetHotOn").checked = !!s.hotwords_on;
  if (document.activeElement !== $("atcSetHot")) $("atcSetHot").value = s.hotwords || "";
}

async function atcSetSend(part) {
  try {
    const r = await fetch(vbase + "/api/atc", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings: part }) }).then((x) => x.json());
    if (r.settings) atcSetShow(r.settings);
  } catch { /* the server is away: the next opening shows what is saved */ }
}

$("atcSetBeam").addEventListener("change", (e) => atcSetSend({ beam: Number(e.target.value) }));
$("atcSetFallback").addEventListener("change", (e) => atcSetSend({ fallback: e.target.checked }));
$("atcSetNoTs").addEventListener("change", (e) => atcSetSend({ no_ts: e.target.checked }));
$("atcSetPromptOn").addEventListener("change", (e) => atcSetSend({ prompt_on: e.target.checked }));
$("atcSetPrompt").addEventListener("change", (e) => atcSetSend({ prompt: e.target.value }));
$("atcSetHotOn").addEventListener("change", (e) => atcSetSend({ hotwords_on: e.target.checked }));
$("atcSetHot").addEventListener("change", (e) => atcSetSend({ hotwords: e.target.value }));
$("atcSetDef").addEventListener("click", () => { if (atcSetDefaults) atcSetSend(atcSetDefaults); });
function atcSetGapShow(v) {
  $("atcSetGap").value = v;
  $("atcSetGapText").textContent = `${Number(v).toFixed(1)} s`;
}
$("atcSetGap").addEventListener("input", (e) => atcSetGapShow(e.target.value));
$("atcSetGap").addEventListener("change", (e) => { atcSetSend({ gap: Number(e.target.value) }); e.target.blur(); });
$("atcSetGap").addEventListener("dblclick", () => {
  const d = atcSetDefaults ? atcSetDefaults.gap : 0.1;
  atcSetGapShow(d);
  atcSetSend({ gap: d });
});
$("atcDbgClr").addEventListener("click", () => {
  const top = $("atcDbgList").querySelector(".atc-dbg-row");
  if (top) atcDbgCut = Math.max(atcDbgCut, Number(top.dataset.id));
  $("atcDbgList").innerHTML = '<div class="atc-dbg-empty muted">Cleared. New transcripts come here.</div>';
});

/** The AD shows nothing of ATC (the key blinks while transcribing; the texts: GET /api/atc). */
function atcAdOn() {
  return false;
}
function atcAdRender() { /* nothing in the AD */ }

