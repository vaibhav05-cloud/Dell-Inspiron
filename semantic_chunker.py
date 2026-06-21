"""
semantic_chunker.py  —  Step 4: Semantic Chunking

Optimizations vs original:
  1. Section-context prefix on EVERY chunk
        "[Section: Power and Battery]\n\n{text}"
     Entity extractor now knows which section each chunk belongs to.

  2. Rich image chunk content
        [Section: X — Figure]
        Caption: <caption>
        <Mistral entity-focused description>
     Entity extractor can now find "Dell Inspiron 1150" from image descriptions.

  3. Rich table chunk content
        [Section: X — Table]
        Caption: <caption>
        Summary: <natural-language prose from multimodal processor>
        <markdown table>
     Entity extractor works on the prose summary first (no | noise),
     then the markdown is available for structural reading.

  4. Chunk deduplication
     MD5 hash of content removes exact duplicates before saving.

  5. Minimum chunk size raised to 80 chars
     Tiny chunks (partial sentences, stray headings) add noise to
     the entity extractor and are removed.
"""

import hashlib
import json
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


class SemanticTextChunker:

    def __init__(self):
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chunker = SemanticChunker(self.embeddings)
        print("Embedding model loaded!")

    # ─────────────────────────────────────────────────────────────────────────
    #  LOAD
    # ─────────────────────────────────────────────────────────────────────────

    def load_processed_json(self, file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ─────────────────────────────────────────────────────────────────────────
    #  TEXT CHUNKS
    # ─────────────────────────────────────────────────────────────────────────

    def build_text_chunks(self, data: dict) -> tuple[list, int]:
        """
        Merge same-section text blocks, then semantically chunk each section.
        Each chunk is prefixed with its section name for entity extraction context.
        """
        chunks      = []
        sections: dict = {}
        source_file = data.get("file_name", "unknown.pdf")

        # ── Merge all text blocks belonging to the same section
        for item in data["texts"]:
            section_name = item.get("section_name", "Unknown Section").strip()
            content      = item.get("content", "").strip()
            if not content:
                continue
            if section_name not in sections:
                sections[section_name] = {
                    "page_number": item["page_number"],
                    "content":     [],
                }
            sections[section_name]["content"].append(content)

        chunk_counter = 1

        for section_name, section_data in sections.items():
            merged_text = "\n\n".join(section_data["content"]).strip()

            if len(merged_text) < 80:
                continue

            # Use SemanticChunker to split the merged section text
            docs = self.chunker.create_documents([merged_text])

            for doc in docs:
                chunk_text = doc.page_content.strip()

                if len(chunk_text) < 80:
                    continue

                # ── KEY CHANGE: prefix with section name for entity context
                content_with_context = (
                    f"[Section: {section_name}]\n\n"
                    f"{chunk_text}"
                )

                chunks.append({
                    "chunk_id":    f"chunk_{chunk_counter}",
                    "chunk_type":  "text",
                    "source_file": source_file,
                    "page_number": section_data["page_number"],
                    "section_name": section_name,
                    "content":     content_with_context,
                })
                chunk_counter += 1

        return chunks, chunk_counter

    # ─────────────────────────────────────────────────────────────────────────
    #  TABLE CHUNKS
    # ─────────────────────────────────────────────────────────────────────────

    def build_table_chunks(
        self,
        data:          dict,
        start_counter: int,
    ) -> tuple[list, int]:
        """
        Build one chunk per table.

        Chunk content = section header + caption + prose summary + markdown.
        The prose summary (added by multimodal_processor) makes entities visible
        to NER; the markdown preserves structure for the LLM in GraphRAG.
        """
        chunks      = []
        source_file = data.get("file_name", "unknown.pdf")
        counter     = start_counter

        for table in data["tables"]:
            markdown     = table.get("markdown",     "").strip()
            text_summary = table.get("text_summary", "").strip()
            caption      = table.get("caption",      "No caption found").strip()
            section_name = table.get("section_name", "Unknown Section").strip()

            if not markdown:
                continue

            # ── Build rich chunk content
            content_parts = [f"[Section: {section_name} — Table]"]

            if caption and caption != "No caption found":
                content_parts.append(f"Caption: {caption}")

            # Prose summary first — entity extractor prefers clean prose over | tables
            if text_summary:
                content_parts.append(f"\nSummary: {text_summary}")

            # Markdown table (for structured reading by the GraphRAG LLM)
            content_parts.append(f"\n{markdown}")

            content = "\n".join(content_parts)

            chunks.append({
                "chunk_id":    f"chunk_{counter}",
                "chunk_type":  "table",
                "source_file": source_file,
                "page_number": table["page_number"],
                "section_name": section_name,
                "content":     content,
            })
            counter += 1

        return chunks, counter

    # ─────────────────────────────────────────────────────────────────────────
    #  IMAGE CHUNKS
    # ─────────────────────────────────────────────────────────────────────────

    def build_image_chunks(
        self,
        data:          dict,
        start_counter: int,
    ) -> tuple[list, int]:
        """
        Build one chunk per described image.

        Chunk content = section header + caption + Mistral's entity-rich description.
        Images with empty descriptions (tiny/decorative, or failed vision calls)
        are skipped entirely.
        """
        chunks      = []
        source_file = data.get("file_name", "unknown.pdf")
        counter     = start_counter

        for image in data["images"]:
            description  = image.get("description", "").strip()
            caption      = image.get("caption",     "No caption found").strip()
            section_name = image.get("section_name", "Unknown Section").strip()

            # Skip images that were not described (decorative / too small)
            if not description:
                continue

            # ── Build rich chunk content
            content_parts = [f"[Section: {section_name} — Figure]"]

            if caption and caption != "No caption found":
                content_parts.append(f"Caption: {caption}")

            content_parts.append(f"\n{description}")

            content = "\n".join(content_parts)

            chunks.append({
                "chunk_id":    f"chunk_{counter}",
                "chunk_type":  "image",
                "source_file": source_file,
                "page_number": image["page_number"],
                "section_name": section_name,
                "content":     content,
            })
            counter += 1

        return chunks, counter

    # ─────────────────────────────────────────────────────────────────────────
    #  DEDUPLICATION
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def deduplicate_chunks(chunks: list) -> list:
        """
        Remove exact-duplicate chunks using MD5 hash of content.
        Duplicates can occur when the same text block is extracted from
        overlapping regions, or the same image xref appears twice.
        """
        seen:   set  = set()
        unique: list = []

        for chunk in chunks:
            content_hash = hashlib.md5(
                chunk["content"].strip().lower().encode("utf-8")
            ).hexdigest()

            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(chunk)

        removed = len(chunks) - len(unique)
        if removed:
            print(f"Deduplication: removed {removed} duplicate chunk(s)")

        return unique

    # ─────────────────────────────────────────────────────────────────────────
    #  SAVE
    # ─────────────────────────────────────────────────────────────────────────

    def save_chunks(self, chunks: list, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        print(f"Chunks saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE ENTRY POINT
#  Run with: uv run python semantic_chunker.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    chunker = SemanticTextChunker()

    print("\nLoading processed JSON...")

    json_files = list(Path("output").glob("*_processed.json"))
    if not json_files:
        raise FileNotFoundError("No *_processed.json found in output/")

    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Using: {latest_json}")

    data = chunker.load_processed_json(str(latest_json))

    # ── Step A: Text chunks
    text_chunks, next_id = chunker.build_text_chunks(data)
    print(f"Text chunks  : {len(text_chunks)}")

    # ── Step B: Table chunks
    table_chunks, next_id = chunker.build_table_chunks(data, next_id)
    print(f"Table chunks : {len(table_chunks)}")

    # ── Step C: Image chunks
    image_chunks, next_id = chunker.build_image_chunks(data, next_id)
    print(f"Image chunks : {len(image_chunks)}")

    # ── Combine
    all_chunks = text_chunks + table_chunks + image_chunks
    
    print(f"\nTotal before dedup : {len(all_chunks)}")

    # ── Deduplicate
    all_chunks = chunker.deduplicate_chunks(all_chunks)
    print(f"Total after  dedup : {len(all_chunks)}")

    # ── Save
    chunker.save_chunks(all_chunks, "output/chunks.json")

    # ── Summary
    print("\n" + "─" * 50)
    print("  CHUNKING COMPLETE")
    print("─" * 50)
    print(f"  Text   chunks : {len(text_chunks)}")
    print(f"  Table  chunks : {len(table_chunks)}")
    print(f"  Image  chunks : {len(image_chunks)}")
    print(f"  Total  chunks : {len(all_chunks)}")
    print(f"  Saved         : output/chunks.json")
    print("─" * 50)
    print("\n  Next step: uv run python run_graph.py")