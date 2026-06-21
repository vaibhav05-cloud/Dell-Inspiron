/* webapp/static/js/chat.js
   Chat interface: send a question to /api/query, render the answer plus
   the full agentic retrieval output (confidence, match-signal meter,
   evidence chunks, reasoning path, graph relationships) so it's clear
   *why* an answer was given. */

let _messages = [];

function initChatPage() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const submitBtn = document.getElementById("chat-submit");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    appendMessage({ role: "user", content: question });
    input.value = "";
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Thinking…';

    const thinkingId = appendMessage({ role: "assistant", content: "Retrieving and synthesizing…", meta: null, thinking: true });

    try {
      const result = await API.postJSON("/api/query", { query: question });
      replaceMessage(thinkingId, { role: "assistant", content: result.answer, meta: result });
    } catch (err) {
      replaceMessage(thinkingId, { role: "assistant", content: `Query failed: ${err.message}`, meta: null, error: true });
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Ask";
      input.focus();
    }
  });
}

function appendMessage(msg) {
  msg._id = msg._id || `m${Date.now()}${Math.random().toString(36).slice(2, 7)}`;
  _messages.push(msg);
  renderChat();
  return msg._id;
}

function replaceMessage(id, newMsg) {
  const idx = _messages.findIndex((m) => m._id === id);
  if (idx === -1) return;
  newMsg._id = id;
  _messages[idx] = newMsg;
  renderChat();
}

function renderChat() {
  const windowEl = document.getElementById("chat-window");
  const emptyEl = document.getElementById("chat-empty");

  if (_messages.length === 0) {
    emptyEl.hidden = false;
    return;
  }
  emptyEl.hidden = true;

  windowEl.innerHTML = _messages
    .map((msg) => {
      const roleLabel = msg.role === "user" ? "You" : "Assistant";
      const bodyHtml = msg.thinking
        ? `<span class="spinner"></span> ${escapeHtml(msg.content)}`
        : escapeHtml(msg.content).replace(/\n/g, "<br>");
      const metaHtml = msg.role === "assistant" && msg.meta ? renderAssistantMeta(msg.meta) : "";
      return `
        <div class="chat-msg ${msg.role}">
          <div class="chat-role">${roleLabel}</div>
          <div>${bodyHtml}</div>
          <div class="chat-meta">${metaHtml}</div>
        </div>
      `;
    })
    .join("");

  windowEl.scrollTop = windowEl.scrollHeight;
}

function renderAssistantMeta(meta) {
  const outOfScope = !!meta.out_of_scope;
  const confidence = meta.confidence || "Low";

  let html = "";

  if (outOfScope) {
    html += pillHtml("OUT OF SCOPE", "coral");
  } else {
    html += pillHtml(`Confidence: ${confidence}`, confidenceVariant(confidence));
  }

  const evidenceChunks = meta.evidence_chunks || [];
  const primary = evidenceChunks.filter((c) => !c.is_neighbor);
  if (primary.length) {
    const topRaw = Math.max(...primary.map((c) => c.score));
    const topNorm = 1 / (1 + Math.exp(-topRaw));
    const meterVariant = outOfScope ? "coral" : "sage";
    html += `
      <div class="match-signal">
        <span class="card-mono-id">match signal · ${topNorm.toFixed(2)}</span>
        ${meterHtml(topNorm, meterVariant)}
      </div>
    `;
  }

  if (evidenceChunks.length) {
    html += `
      <details class="log-panel" style="margin-top:10px;">
        <summary>Evidence (${evidenceChunks.length} chunks)</summary>
        ${evidenceChunks
          .map((c) => {
            const tag = c.is_neighbor ? pillHtml("neighbor", "muted") : pillHtml(c.source || "?", "cyan");
            return `
              <div class="evidence-card">
                <div class="card-mono-id">${escapeHtml(c.chunk_id || "")} · p.${escapeHtml(c.page_number ?? "?")} ·
                  ${escapeHtml(c.section_name || "")} · score ${escapeHtml(c.score ?? "?")}</div>
                ${tag}
                <div style="margin-top:6px;">${escapeHtml(c.content || "")}</div>
              </div>
            `;
          })
          .join("")}
      </details>
    `;
  }

  const reasoningPath = meta.reasoning_path || [];
  if (reasoningPath.length) {
    html += `
      <details class="log-panel" style="margin-top:10px;">
        <summary>Reasoning path</summary>
        <ul style="margin:8px 0 0 18px; padding:0;">
          ${reasoningPath.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}
        </ul>
      </details>
    `;
  }

  const graphPaths = meta.graph_paths || [];
  if (graphPaths.length) {
    html += `
      <details class="log-panel" style="margin-top:10px;">
        <summary>Graph relationships (${graphPaths.length})</summary>
        ${graphPaths
          .map(
            (gp) => `
            <div class="card-mono-id" style="padding:4px 0;">
              ${escapeHtml(gp.source)} → ${escapeHtml(gp.relationship)} → ${escapeHtml(gp.target)}
              (conf ${escapeHtml(gp.confidence)})
            </div>`
          )
          .join("")}
      </details>
    `;
  }

  return html;
}
