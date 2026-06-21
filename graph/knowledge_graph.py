"""
Step 7 — Knowledge Graph Creation
Entities  = Nodes
Relations = Edges

Builds a NetworkX directed graph, saves it as:
  - graph_data.json       (for Steps 8-13 to consume)
  - knowledge_graph.html  (open in browser to see interactive graph)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx
from pyvis.network import Network

logger = logging.getLogger(__name__)

# ── Colour per entity type ────────────────────────────────────────────────────
# These colours appear in the interactive HTML graph

ENTITY_COLORS = {
    "PERSON":       "#4A90D9",   # blue
    "ORGANIZATION": "#E67E22",   # orange
    "PRODUCT":      "#2ECC71",   # green
    "TECHNOLOGY":   "#9B59B6",   # purple
    "CONCEPT":      "#E74C3C",   # red
    "METRIC":       "#1ABC9C",   # teal
    "LOCATION":     "#F39C12",   # yellow
    "EVENT":        "#95A5A6",   # grey
}
DEFAULT_COLOR = "#BDC3C7"


# ── Main Class ────────────────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    Builds a directed Knowledge Graph from entities and relationships.

    How it works
    ------------
    - Nodes  : every unique entity (PERSON, PRODUCT, CONCEPT…)
    - Edges  : every relationship triple (source → relation → target)
    - Size   : nodes that appear in more chunks are drawn bigger
    - Colour : each entity type gets its own colour

    Output
    ------
    graph_data.json       — machine-readable graph for retrieval steps
    knowledge_graph.html  — open in browser for interactive exploration
    """

    def __init__(self):
        # DiGraph = Directed Graph (edges have a direction: A → B)
        self.graph = nx.DiGraph()

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, entities: list, relationships: list) -> nx.DiGraph:
        """
        Populate the graph with nodes and edges.

        Parameters
        ----------
        entities      : list from EntityExtractor.extract_all()
        relationships : list from RelationshipExtractor.extract_all()
        """

        # ── Nodes ─────────────────────────────────────────────────────────────
        # Same entity can appear in multiple chunks → deduplicate by name
        entity_registry: dict = {}

        for ent in entities:
            name = ent.get("name", "").strip()
            if not name:
                continue

            if name not in entity_registry:
                entity_registry[name] = {
                    "type":        ent.get("type", "CONCEPT"),
                    "chunk_ids":   [],
                    "pages":       set(),
                }

            entity_registry[name]["chunk_ids"].append(ent.get("chunk_id", ""))
            page = ent.get("page_number")
            if page:
                entity_registry[name]["pages"].add(page)

        for name, data in entity_registry.items():
            entity_type = data["type"]
            self.graph.add_node(
                name,
                entity_type      = entity_type,
                occurrence_count = len(data["chunk_ids"]),
                pages            = sorted(data["pages"]),
                color            = ENTITY_COLORS.get(entity_type, DEFAULT_COLOR),
            )

        logger.info(f"  Nodes added: {self.graph.number_of_nodes()}")

        # ── Edges ─────────────────────────────────────────────────────────────
        for rel in relationships:
            source   = rel.get("source", "").strip()
            target   = rel.get("target", "").strip()
            relation = rel.get("relation", "RELATED_TO").strip()

            # Only add edge if both nodes are already in the graph
            if source in self.graph and target in self.graph:
                self.graph.add_edge(
                    source,
                    target,
                    relation = relation,
                    chunk_id = rel.get("chunk_id", ""),
                )

        logger.info(f"  Edges added: {self.graph.number_of_edges()}")
        return self.graph

    # ── Visualise ─────────────────────────────────────────────────────────────

    def save_visualization(self, output_path: str) -> str:
        """
        Generate an interactive HTML graph using pyvis.
        Open the saved .html file in any browser to explore the graph.
        - Hover over a node to see its type, pages, and occurrence count
        - Drag nodes around
        - Scroll to zoom
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        net = Network(
            height    = "750px",
            width     = "100%",
            directed  = True,
            bgcolor   = "#1a1a2e",      # dark background
            font_color= "#ffffff",
        )

        # Add nodes
        for node, attrs in self.graph.nodes(data=True):
            # Nodes that appear more often are drawn larger
            size = 15 + (attrs.get("occurrence_count", 1) * 5)

            net.add_node(
                node,
                label = node,
                color = attrs.get("color", DEFAULT_COLOR),
                size  = size,
                title = (
                    f"<b>{node}</b><br>"
                    f"Type: {attrs.get('entity_type', 'UNKNOWN')}<br>"
                    f"Appears in: {attrs.get('occurrence_count', 1)} chunk(s)<br>"
                    f"Pages: {attrs.get('pages', [])}"
                ),
            )

        # Add edges
        for source, target, attrs in self.graph.edges(data=True):
            net.add_edge(
                source,
                target,
                label  = attrs.get("relation", ""),
                color  = "#888888",
                arrows = "to",
                title  = attrs.get("relation", ""),
            )

        # Physics layout settings — makes the graph spread out nicely
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": { "iterations": 150 },
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 220,
              "springConstant": 0.04
            }
          },
          "edges": {
            "font": { "size": 11, "color": "#cccccc" },
            "smooth": { "type": "curvedCW", "roundness": 0.2 }
          },
          "nodes": {
            "font": { "size": 14, "bold": true }
          }
        }
        """)

        net.save_graph(str(output_path))
        logger.info(f"  HTML visualization saved → {output_path}")
        return str(output_path)

    # ── Save JSON ─────────────────────────────────────────────────────────────

    def save_to_json(self, output_path: str) -> str:
        """
        Save graph as machine-readable JSON.
        This file is consumed by Steps 10 and 11 (graph traversal / retrieval).

        Structure
        ---------
        {
          "summary": { "total_nodes": N, "total_edges": M },
          "nodes":   [ { "id", "entity_type", "occurrence_count", "pages" }, ... ],
          "edges":   [ { "source", "relation", "target", "chunk_id"       }, ... ]
        }
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nodes = [
            {
                "id":               node,
                "entity_type":      attrs.get("entity_type", "CONCEPT"),
                "occurrence_count": attrs.get("occurrence_count", 1),
                "pages":            attrs.get("pages", []),
            }
            for node, attrs in self.graph.nodes(data=True)
        ]

        edges = [
            {
                "source":   source,
                "relation": attrs.get("relation", "RELATED_TO"),
                "target":   target,
                "chunk_id": attrs.get("chunk_id", ""),
            }
            for source, target, attrs in self.graph.edges(data=True)
        ]

        graph_data = {
            "summary": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
            "nodes": nodes,
            "edges": edges,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        logger.info(f"  graph_data.json saved → {output_path}")
        return str(output_path)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a summary of the graph."""
        n = self.graph.number_of_nodes()
        return {
            "nodes":        n,
            "edges":        self.graph.number_of_edges(),
            "is_connected": nx.is_weakly_connected(self.graph) if n > 0 else False,
        }