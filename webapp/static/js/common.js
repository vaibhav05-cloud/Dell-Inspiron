/* webapp/static/js/common.js
   Shared helpers used by every page: a small fetch wrapper, the
   pill/meter HTML builders that mirror the original theme.py helpers,
   and the backend connection-status badge. */

const API = {
  async get(path) {
    const resp = await fetch(path, { headers: { Accept: "application/json" } });
    return API._handle(resp);
  },
  async postJSON(path, body) {
    const resp = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return API._handle(resp);
  },
  async postForm(path, formData) {
    const resp = await fetch(path, { method: "POST", body: formData });
    return API._handle(resp);
  },
  async _handle(resp) {
    let data = null;
    try {
      data = await resp.json();
    } catch (e) {
      /* no body */
    }
    if (!resp.ok) {
      const detail = (data && data.detail) || resp.statusText || "Request failed";
      const err = new Error(detail);
      err.status = resp.status;
      throw err;
    }
    return data;
  },
};

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function pillHtml(label, variant) {
  variant = variant || "muted";
  return `<span class="pill pill-${variant}">${escapeHtml(label)}</span>`;
}

function confidenceVariant(confidence) {
  return { High: "sage", Medium: "amber", Low: "coral" }[confidence] || "muted";
}

function meterHtml(score0to1, variant) {
  variant = variant || "amber";
  const colorVar = { amber: "var(--amber)", cyan: "var(--cyan)", sage: "var(--sage)", coral: "var(--coral)" }[variant] || "var(--amber)";
  const pct = Math.max(0, Math.min(1, score0to1)) * 100;
  return `<div class="meter-track"><div class="meter-fill" style="width:${pct.toFixed(1)}%; background:${colorVar};"></div></div>`;
}

function alertHtml(message, variant) {
  variant = variant || "info";
  return `<div class="alert alert-${variant}">${escapeHtml(message)}</div>`;
}

async function initConnStatus() {
  const pillEl = document.getElementById("conn-status-pill");
  if (!pillEl) return;
  try {
    await API.get("/api/health");
    pillEl.className = "pill pill-sage";
    pillEl.textContent = "Backend connected";
  } catch (e) {
    pillEl.className = "pill pill-coral";
    pillEl.textContent = "Backend unreachable";
  }
}
