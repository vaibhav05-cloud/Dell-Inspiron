/* webapp/static/js/upload.js
   Upload a PDF, kick off ingestion, and poll /api/status/<job_id> every
   2s to drive a live stepper + progress bar + log panel -- mirrors the
   original Streamlit page's auto-rerun polling loop. */

const STAGE_ORDER = [
  "parsing",
  "multimodal_enrichment",
  "chunking",
  "entity_extraction",
  "relationship_extraction",
  "graph_ingestion",
  "indexing",
  "done",
];

let _pollTimer = null;
let _selectedFile = null;

function initUploadPage() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("pdf-file");
  const ingestBtn = document.getElementById("ingest-btn");
  const form = document.getElementById("upload-form");

  // NOTE: no manual dropzone.click() -> fileInput.click() handler here.
  // The dropzone is a <label for="pdf-file">, and clicking a <label>
  // associated with a hidden file <input> already opens the file picker
  // natively per HTML spec -- adding a JS .click() handler on top of
  // that double-triggers the dialog (open, close, reopen), which is
  // exactly the "prompted to select the PDF again" symptom.

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(fileInput.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFileSelected(fileInput.files[0]);
  });

  function handleFileSelected(file) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      document.getElementById("upload-alert").innerHTML = alertHtml("Only PDF files are supported.", "error");
      return;
    }
    _selectedFile = file;
    document.getElementById("dropzone-filename").textContent = file.name;
    ingestBtn.disabled = false;
    document.getElementById("upload-alert").innerHTML = "";
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!_selectedFile) return;

    ingestBtn.disabled = true;
    ingestBtn.innerHTML = '<span class="spinner"></span> Queuing…';

    const formData = new FormData();
    formData.append("file", _selectedFile);

    try {
      const result = await API.postForm("/api/upload", formData);
      document.getElementById("upload-alert").innerHTML = alertHtml(`Queued — job ${result.job_id}`, "success");
      startPolling(result.job_id, _selectedFile.name);
    } catch (err) {
      document.getElementById("upload-alert").innerHTML = alertHtml(`Upload failed: ${err.message}`, "error");
    } finally {
      ingestBtn.disabled = false;
      ingestBtn.textContent = "Ingest";
    }
  });
}

function startPolling(jobId, filename) {
  document.getElementById("no-job-msg").hidden = true;
  const panel = document.getElementById("job-panel");
  panel.hidden = false;
  document.getElementById("job-title").textContent = `Job ${jobId} — ${filename}`;

  if (_pollTimer) clearInterval(_pollTimer);
  pollOnce(jobId);
  _pollTimer = setInterval(() => pollOnce(jobId), 2000);
}

async function pollOnce(jobId) {
  let job;
  try {
    job = await API.get(`/api/status/${jobId}`);
  } catch (err) {
    document.getElementById("job-status-alert").innerHTML = alertHtml(`Couldn't fetch job status: ${err.message}`, "error");
    return;
  }

  renderStepper(job);
  renderProgress(job);
  renderStatus(job);
  renderLog(job);

  if (job.status === "complete" || job.status === "error") {
    if (_pollTimer) {
      clearInterval(_pollTimer);
      _pollTimer = null;
    }
  }
}

function renderStepper(job) {
  const currentIdx = STAGE_ORDER.indexOf(job.stage);
  document.querySelectorAll("#stepper .step").forEach((stepEl) => {
    const stageKey = stepEl.dataset.stage;
    const i = STAGE_ORDER.indexOf(stageKey);
    const marker = stepEl.querySelector(".marker");

    let variant = "muted";
    let symbol = "○";

    if (job.status === "error" && i === currentIdx) {
      variant = "coral";
      symbol = "✕";
    } else if (i < currentIdx || job.status === "complete") {
      variant = "sage";
      symbol = "✓";
    } else if (i === currentIdx) {
      variant = "amber";
      symbol = "●";
    }

    marker.className = `marker pill pill-${variant}`;
    marker.textContent = symbol;
  });
}

function renderProgress(job) {
  const pct = Math.min(job.progress || 0, 100);
  document.getElementById("progress-fill").style.width = `${pct}%`;
}

function renderStatus(job) {
  const el = document.getElementById("job-status-alert");
  if (job.status === "complete") {
    el.innerHTML = alertHtml("Ingestion complete — head to the Chat page to ask questions.", "success");
  } else if (job.status === "error") {
    el.innerHTML = alertHtml(`Ingestion failed: ${job.error || "unknown error"}`, "error");
  } else {
    const label = job.stage_label || job.stage || "queued";
    el.innerHTML = alertHtml(`Stage: ${label}`, "info");
  }
}

function renderLog(job) {
  const linesEl = document.getElementById("log-lines");
  const entries = job.log || [];
  linesEl.innerHTML = entries
    .map((entry) => `<div class="log-line">${escapeHtml(entry.message)}</div>`)
    .join("");
  // keep panel open while running, default-collapsible once finished is
  // left to the user (we don't force-close it).
}