"""
Entry point for the Dell FutureMinds project.
Run with:  python main.py
"""

from parser.pdf_parser import PDFParser
from processor.multimodal_processor import MultimodalProcessor


def main():
    # ── Config ────────────────────────────────────────────────────────────────
    PDF_PATH   = "data/pdfs/sample.pdf"   # ← point this at your new PDF's filename
    OUTPUT_DIR = "output"

    # ── Step 2: Parse ─────────────────────────────────────────────────────────
    parser     = PDFParser(output_dir=OUTPUT_DIR)
    parsed_doc = parser.parse(PDF_PATH)   # clean=True by default: wipes old output first

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = parser.save_to_json(parsed_doc)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  PARSING COMPLETE")
    print("─" * 50)
    print(f"  Pages        : {parsed_doc.total_pages}")
    print(f"  Text blocks  : {len(parsed_doc.texts)}")
    print(f"  Tables       : {len(parsed_doc.tables)}")
    print(f"  Images       : {len(parsed_doc.images)}")
    print(f"  JSON saved   : {output_path}")
    print("─" * 50)

    # ── Quick preview ─────────────────────────────────────────────────────────
    if parsed_doc.texts:
        print("\nFirst text block:")
        first = parsed_doc.texts[0]
        print(f"  Page    : {first.page_number}")
        print(f"  Section : {first.section_name}")
        print(f"  Content : {first.content[:120]}...")

    if parsed_doc.tables:
        print("\nFirst table:")
        t = parsed_doc.tables[0]
        print(f"  Page    : {t.page_number}")
        print(f"  Headers : {t.headers}")
        print(f"  Rows    : {len(t.rows)} data rows")

    if parsed_doc.images:
        print("\nFirst image:")
        img = parsed_doc.images[0]
        print(f"  Page    : {img.page_number}")
        print(f"  Size    : {img.width}×{img.height} px")
        print(f"  Saved   : {img.image_path}")
        print(f"  Caption : {img.caption}")

    # ── Step 3: Multimodal Processing ────────────────────────────────────────
    mm_processor   = MultimodalProcessor()
    processed_path = mm_processor.process_file(output_path)

    print("\n" + "─" * 50)
    print("  MULTIMODAL PROCESSING COMPLETE")
    print("─" * 50)
    print(f"  Processed JSON : {processed_path}")
    print("─" * 50)


if __name__ == "__main__":
    main()