const API_BASE = "http://localhost:8000";
const CHAT_STORAGE_KEY = "dellgraphrag_chat_history";

// ===================== Tab switching =====================
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    tabPanels.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`${btn.dataset.tab}-tab`).classList.add("active");
  });
});

// ===================== Chat (Query tab) =====================
const chatWindow = document.getElementById("chat-window");
const chatEmpty = document.getElementById("chat-empty");
const queryForm = document.getElementById("query-form");
const queryInput = document.getElementById("query-input");
const querySubmit = document.getElementById("query-submit");
const querySpinner = document.getElementById("query-spinner");
const clearChatBtn = document.getElementById("clear-chat");

function loadHistory() {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(history) {
  localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(history));
}

function renderBubble(entry) {
  chatEmpty.hidden = true;

  const row = document.createElement("div");
  row.className = `bubble-row ${entry.role}${entry.error ? " bubble-error" : ""}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = entry.text;

  row.appendChild(bubble);

  // For bot responses with metadata, render confidence + evidence + reasoning path
  if (entry.role === "bot" && entry.meta) {
    const meta = document.createElement("div");
    meta.className = "bubble-meta";

    const confSpan = document.createElement("div");
    confSpan.innerHTML = `Confidence: <span class="confidence ${entry.meta.confidence}">${entry.meta.confidence}</span>`;
    meta.appendChild(confSpan);

    if (entry.meta.cached) {
      const cachedNote = document.createElement("div");
      cachedNote.textContent = `Served from cache (${entry.meta.server_total_ms.toFixed(3)} ms)`;
      meta.appendChild(cachedNote);
    } else if (entry.meta.server_total_ms != null) {
      const timeNote = document.createElement("div");
      timeNote.textContent = `Response time: ${(entry.meta.server_total_ms / 1000).toFixed(2)}s`;
      meta.appendChild(timeNote);
    }

    if (entry.meta.evidence && entry.meta.evidence.length) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `Sources (${entry.meta.evidence.length})`;
      details.appendChild(summary);

      const list = document.createElement("ul");
      list.className = "evidence-list";
      entry.meta.evidence.forEach((ev) => {
        const li = document.createElement("li");
        li.textContent = `${ev.source_document} — page ${ev.page_number}`;
        list.appendChild(li);
      });
      details.appendChild(list);
      meta.appendChild(details);
    }

    if (entry.meta.reasoning_path && entry.meta.reasoning_path.length) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Reasoning path";
      details.appendChild(summary);

      const list = document.createElement("ul");
      list.className = "evidence-list";
      entry.meta.reasoning_path.forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        list.appendChild(li);
      });
      details.appendChild(list);
      meta.appendChild(details);
    }

    bubble.appendChild(meta);
  }

  chatWindow.appendChild(row);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function renderAllHistory() {
  const history = loadHistory();
  chatWindow.innerHTML = "";
  if (history.length === 0) {
    chatWindow.appendChild(chatEmpty);
    chatEmpty.hidden = false;
    return;
  }
  history.forEach(renderBubble);
}

function appendToHistory(entry) {
  const history = loadHistory();
  history.push(entry);
  saveHistory(history);
  renderBubble(entry);
}

queryForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = queryInput.value.trim();
  if (!text) return;

  appendToHistory({ role: "user", text });
  queryInput.value = "";
  queryInput.disabled = true;
  querySubmit.disabled = true;
  querySpinner.hidden = false;

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();

    appendToHistory({
      role: "bot",
      text: data.answer,
      meta: {
        confidence: data.confidence,
        evidence: data.evidence,
        reasoning_path: data.reasoning_path,
        server_total_ms: data.server_total_ms,
        cached: data.cached,
      },
    });
  } catch (err) {
    appendToHistory({
      role: "bot",
      text: `Something went wrong: ${err.message}`,
      error: true,
    });
  } finally {
    queryInput.disabled = false;
    querySubmit.disabled = false;
    querySpinner.hidden = true;
    queryInput.focus();
  }
});

clearChatBtn.addEventListener("click", () => {
  if (!confirm("Clear all chat history? This cannot be undone.")) return;
  localStorage.removeItem(CHAT_STORAGE_KEY);
  renderAllHistory();
});

// Render whatever was saved from a previous session
renderAllHistory();

// ===================== Upload tab =====================
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("pdf-file");
const fileChip = document.getElementById("file-chip");
const fileChipName = document.getElementById("file-chip-name");
const fileChipRemove = document.getElementById("file-chip-remove");
const dropzoneContent = document.getElementById("dropzone-content");
const forceCheckbox = document.getElementById("force-checkbox");
const uploadForm = document.getElementById("upload-form");
const uploadSubmit = document.getElementById("upload-submit");
const uploadSpinner = document.getElementById("upload-spinner");
const uploadStatus = document.getElementById("upload-status");

function showSelectedFile(file) {
  fileChipName.textContent = file.name;
  fileChip.hidden = false;
  dropzoneContent.hidden = true;
}

function clearSelectedFile() {
  fileInput.value = "";
  fileChip.hidden = true;
  dropzoneContent.hidden = false;
}

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) showSelectedFile(fileInput.files[0]);
});

fileChipRemove.addEventListener("click", clearSelectedFile);

["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);

["dragleave", "dragend"].forEach((evt) =>
  dropzone.addEventListener(evt, () => dropzone.classList.remove("dragover"))
);

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (file.type !== "application/pdf") {
    uploadStatus.innerHTML = `<span class="status-error">Only PDF files are accepted.</span>`;
    return;
  }
  fileInput.files = e.dataTransfer.files;
  showSelectedFile(file);
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    uploadStatus.innerHTML = `<span class="status-error">Please select a PDF file first.</span>`;
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("force", forceCheckbox.checked ? "true" : "false");

  uploadSubmit.disabled = true;
  uploadSpinner.hidden = false;
  uploadStatus.innerHTML = `<span class="status-pending">Uploading and starting ingestion…</span>`;

  try {
    const res = await fetch(`${API_BASE}/ingest`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Upload failed (${res.status})`);
    }

    const data = await res.json();
    uploadStatus.innerHTML = `<span class="status-ok">Ingestion started for "${data.pdf_path}". Check server logs for progress.</span>`;
    clearSelectedFile();
    forceCheckbox.checked = false;
  } catch (err) {
    uploadStatus.innerHTML = `<span class="status-error">${err.message}</span>`;
  } finally {
    uploadSubmit.disabled = false;
    uploadSpinner.hidden = true;
  }
});