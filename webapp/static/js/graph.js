/* webapp/static/js/graph.js
   Fetches /api/graph (entities + relationships for the currently-
   ingested document) and renders an interactive vis-network graph into
   #graph-canvas, with a live name filter and a type-color legend.

   Color choices mirror run_graph_view.py's ENTITY_TYPE_COLORS exactly,
   so a node's color means the same thing whether you're looking at this
   in-app explorer or the standalone HTML the CLI script generates. */

const ENTITY_TYPE_COLORS = {
  ORGANIZATION: "#4C9AFF",
  PERSON: "#FF8B00",
  PRODUCT: "#36B37E",
  TECHNOLOGY: "#6554C0",
  PROJECT: "#FF5630",
  LOCATION: "#00B8D9",
  METRIC_KPI: "#FFC400",
  DATE: "#998DD9",
  CONCEPT: "#79E2F2",
  OTHER: "#A5ADBA",
};
const DEFAULT_ENTITY_COLOR = "#A5ADBA";

let _network = null;
let _allNodes = null; // vis.DataSet
let _allEdges = null; // vis.DataSet
let _entityNameById = new Map();

async function initGraphPage() {
  let data;
  try {
    data = await API.get("/api/graph");
  } catch (err) {
    document.getElementById("graph-empty").hidden = false;
    document.getElementById("graph-empty").textContent = `Couldn't load graph: ${err.message}`;
    return;
  }

  const entities = data.entities || [];
  const relationships = data.relationships || [];

  if (!entities.length) {
    document.getElementById("graph-empty").hidden = false;
    return;
  }

  document.getElementById("graph-content").hidden = false;

  renderMetrics(entities, relationships);
  renderLegend(entities);
  buildNetwork(entities, relationships);

  const filterInput = document.getElementById("graph-filter");
  filterInput.addEventListener("input", () => applyFilter(filterInput.value));
}

function renderMetrics(entities, relationships) {
  document.getElementById("metric-entities").textContent = entities.length;
  document.getElementById("metric-relationships").textContent = relationships.length;
  const types = new Set(entities.map((e) => e.entity_type));
  document.getElementById("metric-types").textContent = types.size;
}

function renderLegend(entities) {
  const types = Array.from(new Set(entities.map((e) => e.entity_type))).sort();
  const legendEl = document.getElementById("graph-legend");

  legendEl.innerHTML = types
    .map((type) => {
      const color = ENTITY_TYPE_COLORS[type] || DEFAULT_ENTITY_COLOR;
      return `
        <span class="legend-item" style="display:inline-flex; align-items:center; gap:6px; font-size:0.82rem; margin-right:4px;">
          <span style="width:10px; height:10px; border-radius:50%; background:${color}; display:inline-block;"></span>
          ${escapeHtml(type)}
        </span>
      `;
    })
    .join("");
}

function buildNetwork(entities, relationships) {
  _entityNameById = new Map(entities.map((e) => [e.entity_id, e.entity_name]));

  const nodeData = entities.map((e) => ({
    id: e.entity_id,
    label: e.entity_name,
    title: `${e.entity_name} (${e.entity_type})`,
    color: ENTITY_TYPE_COLORS[e.entity_type] || DEFAULT_ENTITY_COLOR,
    shape: "dot",
    size: 12,
  }));

  const edgeData = relationships
    // Defensive: only draw edges whose endpoints actually exist as
    // entity nodes -- guards against any stale/partial relationships.json
    // referencing an entity_id that didn't make it into entities.json.
    .filter((r) => _entityNameById.has(r.source_entity_id) && _entityNameById.has(r.target_entity_id))
    .map((r, i) => ({
      id: r.relationship_id || `rel_${i}`,
      from: r.source_entity_id,
      to: r.target_entity_id,
      label: r.relationship_type,
      title: `${r.relationship_type} (confidence=${r.confidence ?? "?"})`,
      arrows: "to",
      font: { size: 10, color: "#8b93a3", strokeWidth: 0 },
      color: { color: "#3a4150", highlight: "#5ec8d8" },
    }));

  _allNodes = new vis.DataSet(nodeData);
  _allEdges = new vis.DataSet(edgeData);

  const container = document.getElementById("graph-canvas");
  const options = {
    nodes: {
      font: { color: "#e6e9ef", size: 13 },
      borderWidth: 1,
    },
    edges: {
      smooth: { type: "continuous" },
    },
    physics: {
      barnesHut: { gravitationalConstant: -6000, springLength: 110, springConstant: 0.02 },
      minVelocity: 0.75,
      stabilization: { iterations: 150 },
    },
    interaction: { hover: true, tooltipDelay: 120 },
  };

  _network = new vis.Network(container, { nodes: _allNodes, edges: _allEdges }, options);
}

function applyFilter(rawQuery) {
  const query = rawQuery.trim().toLowerCase();

  if (!query) {
    _allNodes.forEach((n) => _allNodes.update({ id: n.id, hidden: false }));
    _allEdges.forEach((e) => _allEdges.update({ id: e.id, hidden: false }));
    return;
  }

  const matchingIds = new Set();
  _allNodes.forEach((n) => {
    if (n.label.toLowerCase().includes(query)) matchingIds.add(n.id);
  });

  // Also keep any node that's directly connected to a matching node, so
  // the filtered view shows immediate context rather than isolated dots.
  const visibleIds = new Set(matchingIds);
  _allEdges.forEach((e) => {
    if (matchingIds.has(e.from) || matchingIds.has(e.to)) {
      visibleIds.add(e.from);
      visibleIds.add(e.to);
    }
  });

  _allNodes.forEach((n) => _allNodes.update({ id: n.id, hidden: !visibleIds.has(n.id) }));
  _allEdges.forEach((e) =>
    _allEdges.update({ id: e.id, hidden: !(visibleIds.has(e.from) && visibleIds.has(e.to)) })
  );
}
