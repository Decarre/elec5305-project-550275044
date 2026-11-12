const audio = document.querySelector("#audio");
const run = document.querySelector("#run");
const upload = document.querySelector("#upload");
const status = document.querySelector("#status");
const detail = document.querySelector("#detail");
const tracks = document.querySelector("#tracks");
const ticks = document.querySelector("#ticks");
const wave = document.querySelector("#wave");
const canvas = wave.querySelector("canvas");
const finding = document.querySelector("#finding");
const instruments = ["drums", "bass", "guitar", "piano"];
let cases = {};
let source = "success";
let file;
let objectURL;
let result;
let revision = 0;
let pending = false;
let audioContext;
let waveform;

function seconds(value) {
  return Number(value.toFixed(2)) + " s";
}

function seek(value, description) {
  if (!Number.isFinite(audio.duration)) return;
  audio.currentTime = Math.min(value, audio.duration);
  detail.textContent = description;
}

function updatePlayhead() {
  const duration = result?.duration_seconds ?? audio.duration;
  const position = duration > 0 ? Math.min(audio.currentTime / duration, 1) : 0;
  for (const line of document.querySelectorAll(".playhead")) {
    line.style.left = position * 100 + "%";
  }
}

function drawWave() {
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * scale);
  canvas.height = Math.round(canvas.clientHeight * scale);
  const ctx = canvas.getContext("2d");
  const middle = canvas.height / 2;
  ctx.strokeStyle = "#344548";
  ctx.beginPath();
  ctx.moveTo(0, middle);
  ctx.lineTo(canvas.width, middle);
  ctx.stroke();
  if (!waveform) return;
  const step = canvas.width / waveform.length;
  ctx.strokeStyle = "#85c8b3";
  ctx.lineWidth = Math.max(1, step * 0.68);
  ctx.beginPath();
  for (const [index, [low, high]] of waveform.entries()) {
    const x = (index + 0.5) * step;
    ctx.moveTo(x, middle - high * middle * 0.88);
    ctx.lineTo(x, middle - low * middle * 0.88);
  }
  ctx.stroke();
}

async function loadWave() {
  const requestRevision = revision;
  waveform = undefined;
  drawWave();
  try {
    const bytes = file ? await file.arrayBuffer() : await (await fetch(audio.src)).arrayBuffer();
    if (requestRevision !== revision) return;
    audioContext ??= new AudioContext();
    const decoded = await audioContext.decodeAudioData(bytes);
    if (requestRevision !== revision) return;
    const signal = decoded.getChannelData(0);
    const bins = Math.min(240, signal.length);
    const peaks = [];
    let largest = 0;
    for (let bin = 0; bin < bins; bin++) {
      let low = 0;
      let high = 0;
      const end = Math.floor((bin + 1) * signal.length / bins);
      for (let sample = Math.floor(bin * signal.length / bins); sample < end; sample++) {
        low = Math.min(low, signal[sample]);
        high = Math.max(high, signal[sample]);
      }
      largest = Math.max(largest, high, -low);
      peaks.push([low, high]);
    }
    waveform = largest ? peaks.map(([low, high]) => [low / largest, high / largest]) : peaks;
    drawWave();
  } catch {
    // The native player and inference can still operate if waveform decoding fails.
  }
}

wave.addEventListener("click", event => {
  if (!(audio.duration > 0)) return;
  const box = wave.getBoundingClientRect();
  const time = Math.max(0, Math.min(audio.duration, (event.clientX - box.left) / box.width * audio.duration));
  seek(time, "Playhead at " + seconds(time));
});
new ResizeObserver(drawWave).observe(wave);

function render() {
  const duration = result?.duration_seconds ?? (source === "upload" ? audio.duration : 5);
  tracks.replaceChildren();
  ticks.replaceChildren();
  const example = cases[source];
  const focus = result && example?.focus;
  finding.hidden = !focus;
  if (focus) {
    const [start, end] = example.focus_interval;
    const name = focus[0].toUpperCase() + focus.slice(1);
    const predicted = result.intervals.filter(interval => interval.instrument === focus);
    const match = predicted.some(interval => interval.start_seconds === start && interval.end_seconds === end);
    const overlap = predicted.some(interval => interval.start_seconds < end && interval.end_seconds > start);
    const description = match ? "prediction matches the MIDI reference" :
      overlap ? "prediction partly overlaps the MIDI reference" : "no prediction during MIDI activity";
    finding.textContent = name + " · " + seconds(start) + " to " + seconds(end) + " · " + description;
  }
  if (!(duration > 0)) return;
  for (let index = 0; index <= 5; index++) {
    const label = document.createElement("span");
    label.textContent = seconds(duration * index / 5);
    ticks.append(label);
  }
  for (const instrument of instruments) {
    const row = document.createElement("div");
    row.className = focus === instrument ? "row focus" : "row";
    const name = document.createElement("strong");
    name.textContent = instrument[0].toUpperCase() + instrument.slice(1);
    const lane = document.createElement("div");
    lane.className = "lane";
    lane.addEventListener("click", event => {
      if (event.target !== lane) return;
      const box = lane.getBoundingClientRect();
      const time = Math.max(0, Math.min(duration, (event.clientX - box.left) / box.width * duration));
      seek(time, "Playhead at " + seconds(time));
    });
    const predicted = (result?.intervals ?? []).filter(interval => interval.instrument === instrument)
      .map(interval => [interval.start_seconds, interval.end_seconds]);
    for (const [kind, intervals] of [
      ["Prediction", predicted], ["MIDI reference", example?.reference[instrument] ?? []],
    ]) {
      for (const [start, end] of intervals) {
        const bar = document.createElement("button");
        bar.type = "button";
        bar.className = kind === "Prediction" ? "bar" : "bar reference";
        bar.style.left = start / duration * 100 + "%";
        bar.style.width = (end - start) / duration * 100 + "%";
        const description = kind + " · " + name.textContent + " · " + seconds(start) + " to " + seconds(end);
        bar.title = description;
        bar.setAttribute("aria-label", description);
        bar.addEventListener("click", () => seek(start, description));
        lane.append(bar);
      }
    }
    const playhead = document.createElement("span");
    playhead.className = "playhead";
    lane.append(playhead);
    row.append(name, lane);
    tracks.append(row);
  }
  updatePlayhead();
}

function select(next, selectedFile) {
  revision++;
  result = undefined;
  audio.pause();
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = selectedFile ? URL.createObjectURL(selectedFile) : undefined;
  source = next;
  file = selectedFile;
  audio.src = objectURL ?? "/audio/" + source;
  for (const name of ["success", "failure"]) {
    document.getElementById(name).setAttribute("aria-pressed", String(source === name));
  }
  document.querySelector("#ref-key").hidden = Boolean(file);
  status.textContent = file ? file.name : "5-second validation clip";
  detail.textContent = "Click a lane or interval to seek in the audio.";
  render();
  loadWave();
}

for (const name of ["success", "failure"]) {
  document.getElementById(name).addEventListener("click", () => select(name));
}
upload.addEventListener("change", () => {
  if (upload.files.length) select("upload", upload.files[0]);
  upload.value = "";
});
audio.addEventListener("loadedmetadata", render);
audio.addEventListener("timeupdate", updatePlayhead);
run.addEventListener("click", async () => {
  if (pending) return;
  const requestRevision = revision;
  pending = run.disabled = true;
  status.textContent = "Running the attention model…";
  try {
    const response = await fetch("/predict/" + source, {method: "POST", body: file});
    const data = await response.json();
    if (requestRevision !== revision) return;
    if (!response.ok) throw new Error(data.error ?? "Prediction failed.");
    result = data;
    const count = data.intervals.length;
    status.textContent = "Prediction complete · " + count + (count === 1 ? " active interval" : " active intervals");
    render();
  } catch (error) {
    if (requestRevision === revision) status.textContent = error.message;
  } finally {
    pending = run.disabled = false;
  }
});
render();
loadWave();
fetch("/cases.json").then(response => {
  if (!response.ok) throw new Error("Reference data could not be loaded.");
  return response.json();
}).then(data => {
  cases = data;
  render();
}).catch(error => {
  detail.textContent = error.message;
});
