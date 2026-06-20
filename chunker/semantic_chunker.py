import json

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


class SemanticTextChunker:

    def __init__(self):

        print("Loading embedding model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.chunker = SemanticChunker(
            self.embeddings
        )

        print("Embedding model loaded successfully!")

    def load_processed_json(self, file_path):

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_text_chunks(self, data):

        chunks = []

        sections = {}

        source_file = data.get(
            "file_name",
            "unknown.pdf"
        )

        # -----------------------
        # Merge same sections
        # -----------------------

        for item in data["texts"]:

            section_name = item.get(
                "section_name",
                "Unknown Section"
            ).strip()

            content = item.get(
                "content",
                ""
            ).strip()

            if not content:
                continue

            if section_name not in sections:

                sections[section_name] = {
                    "page_number": item["page_number"],
                    "content": []
                }

            sections[section_name]["content"].append(
                content
            )

        chunk_counter = 1

        for section_name, section_data in sections.items():

            merged_text = "\n\n".join(
                section_data["content"]
            )

            merged_text = merged_text.strip()

            if len(merged_text) < 50:
                continue

            docs = self.chunker.create_documents(
                [merged_text]
            )

            for doc in docs:

                chunk_text = doc.page_content.strip()

                if len(chunk_text) < 50:
                    continue

                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_counter}",
                        "chunk_type": "text",
                        "source_file": source_file,
                        "page_number": section_data["page_number"],
                        "section_name": section_name,
                        "content": chunk_text
                    }
                )

                chunk_counter += 1

        return chunks, chunk_counter

    def build_table_chunks(
        self,
        data,
        start_counter
    ):

        chunks = []

        source_file = data.get(
            "file_name",
            "unknown.pdf"
        )

        counter = start_counter

        for table in data["tables"]:

            markdown = table.get(
                "markdown",
                ""
            ).strip()

            if not markdown:
                continue

            chunks.append(
                {
                    "chunk_id": f"chunk_{counter}",
                    "chunk_type": "table",
                    "source_file": source_file,
                    "page_number": table["page_number"],
                    "section_name": table["section_name"],
                    "content": markdown
                }
            )

            counter += 1

        return chunks, counter

    def build_image_chunks(
        self,
        data,
        start_counter
    ):

        chunks = []

        source_file = data.get(
            "file_name",
            "unknown.pdf"
        )

        counter = start_counter

        for image in data["images"]:

            description = image.get(
                "description",
                ""
            ).strip()

            if not description:
                continue

            chunks.append(
                {
                    "chunk_id": f"chunk_{counter}",
                    "chunk_type": "image",
                    "source_file": source_file,
                    "page_number": image["page_number"],
                    "section_name": image["section_name"],
                    "content": description
                }
            )

            counter += 1

        return chunks, counter

    def save_chunks(
        self,
        chunks,
        output_path
    ):

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                chunks,
                f,
                indent=4,
                ensure_ascii=False
            )

        print(
            f"Chunks saved to: {output_path}"
        )


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Chunk processed JSON semantically."
    )
    parser.add_argument(
        "--input",
        default="output/sample_processed.json",
        help="Path to the processed JSON file (default: output/sample_processed.json)"
    )
    parser.add_argument(
        "--output",
        default="output/chunks.json",
        help="Path to write the chunks.json (default: output/chunks.json)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Please specify the correct processed file using --input")
        sys.exit(1)

    chunker = SemanticTextChunker()

    print(f"\nLoading processed JSON from {input_path}...")
    data = chunker.load_processed_json(str(input_path))

    # -----------------------
    # Text Chunks
    # -----------------------
    text_chunks, next_id = chunker.build_text_chunks(data)
    print(f"Text Chunks: {len(text_chunks)}")

    # -----------------------
    # Table Chunks
    # -----------------------
    table_chunks, next_id = chunker.build_table_chunks(data, next_id)
    print(f"Table Chunks: {len(table_chunks)}")

    # -----------------------
    # Image Chunks
    # -----------------------
    image_chunks, next_id = chunker.build_image_chunks(data, next_id)
    print(f"Image Chunks: {len(image_chunks)}")

    all_chunks = text_chunks + table_chunks + image_chunks
    print(f"\nTotal Chunks: {len(all_chunks)}")

    chunker.save_chunks(all_chunks, args.output)
    print("\nSemantic chunking pipeline completed successfully!")