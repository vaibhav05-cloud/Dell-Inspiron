/* webapp/static/js/dashboard.js
   Overview of the currently-ingested document: counts and entity/
   relationship type breakdowns, rendered as lightweight CSS bars (same
   "signal meter" visual language as the rest of the app) instead of
   pulling in a charting library -- keeps the page fast and dependency-
   free. */

async function initDashboardPage() {
  let data;
  try {
    data = await API.get("/api/documents");
  } catch (err) {
    document.getElementById("dash-empty").hidden = false;
    document.getElementById("dash-empty").textContent = `Couldn't fetch document info: ${err.message}`;
    return;
  }

  const documents = data.documents || [];
  if (!documents.length) {
    document.getElementById("dash-empty").hidden = false;
    return;
  }

  const doc = documents[0];
  document.getElementById("dash-content").hidden = false;

  document.getElementById("dash-filename").textContent = doc.filename || "Unknown document";
  document.getElementById("dash-ingested-at").textContent = doc.ingested_at
    ? `Last ingested: ${new Date(doc.ingested_at * 1000).toLocaleString()}`
    : "Last ingested: —";

  document.getElementById("metric-chunks").textContent = doc.chunk_count || 0;
  document.getElementById("metric-entities").textContent = doc.entity_count || 0;
  document.getElementById("metric-relationships").textContent = doc.relationship_count || 0;

  renderBarList("entities-by-type", doc.entities_by_type || {}, "amber");
  renderBarList("relationships-by-type", doc.relationships_by_type || {}, "cyan");
}

function renderBarList(containerId, breakdown, variant) {
  const el = document.getElementById(containerId);
  const entries = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);

  if (!entries.length) {
    el.innerHTML = '<span class="muted">No data.</span>';
    return;
  }

  const max = Math.max(...entries.map(([, count]) => count));

  el.innerHTML = entries
    .map(([label, count]) => {
      const pct = max ? count / max : 0;
      return `
        <div class="bar-row">
          <div class="bar-row-top"><span>${escapeHtml(label)}</span><span class="count">${count}</span></div>
          ${meterHtml(pct, variant)}
        </div>
      `;
    })
    .join("");
}
