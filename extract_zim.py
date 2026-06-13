#!/usr/bin/env python3
"""
Extract and chunk text from a Kiwix .zim file.

Handles HTML pages and embedded PDFs. Scanned-image PDFs (no extractable text)
are detected and either skipped or OCR'd depending on flags.

Usage:
    python extract_zim.py <file.zim> [options]
    python extract_zim.py <file.zim> --ocr                        # tesseract (default)
    python extract_zim.py <file.zim> --ocr --ocr-engine easyocr  # no binary needed

Output is a JSON Lines file (.jsonl) — one chunk per line — ready for embedding.
"""

import argparse
import io
import json
import sys
from pathlib import Path

import libzim
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

# Tags that add noise without useful text.
# Note: "table" is intentionally excluded — tables often contain primary reference
# content (dosage tables, spec matrices, CLI options). get_text() extracts cell text.
_NOISE_TAGS = ["script", "style", "nav", "header", "footer", "figure"]

# PDFs with fewer characters than this per page are likely scanned images
_PDF_MIN_CHARS_PER_PAGE = 100


def extract_html_blocks(html_bytes: bytes) -> list[dict]:
    """
    Parse HTML and return a list of text blocks, each with an is_accepted flag.
    For Stack Exchange pages, the accepted answer is extracted as a separate
    block so its chunks can be boosted during retrieval. All other pages return
    a single block with is_accepted=False.
    """
    soup = BeautifulSoup(html_bytes.decode("utf-8", errors="ignore"), "html.parser")
    blocks = []

    accepted_el = soup.find(class_="accepted-answer")
    if accepted_el:
        # Extract the accepted answer as its own block
        for tag in accepted_el(_NOISE_TAGS):
            tag.decompose()
        accepted_text = sanitize(accepted_el.get_text(separator=" ", strip=True))
        accepted_el.decompose()  # remove from main soup before extracting rest
        if accepted_text:
            blocks.append({"text": accepted_text, "is_accepted": True})

    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    main_text = sanitize(soup.get_text(separator=" ", strip=True))
    if main_text:
        blocks.append({"text": main_text, "is_accepted": False})

    return blocks


def sanitize(text: str) -> str:
    """Replace control characters (except whitespace) with a space."""
    return "".join(
        ch if ch >= " " or ch in "\n\r\t" else " "
        for ch in text
    )


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, bool]:
    """Return (text, is_scanned). is_scanned=True when no selectable text found."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    full_text = sanitize(" ".join(pages)).strip()
    total_pages = len(reader.pages)
    avg_chars = len(full_text) / total_pages if total_pages else 0
    is_scanned = avg_chars < _PDF_MIN_CHARS_PER_PAGE
    return full_text, is_scanned


def iter_chunks(archive: libzim.Archive, splitter: RecursiveCharacterTextSplitter,
                ocr_engine=None, entry_offset: int = 0, entry_limit: int = 0,
                quality_filter: bool = False):
    total = archive.all_entry_count
    start = entry_offset
    end = min(entry_offset + entry_limit, total) if entry_limit > 0 else total
    counts = {"skipped": 0, "html": 0, "pdf": 0, "pdf_scanned": 0, "pdf_error": 0,
              "filtered": 0}

    if quality_filter:
        from chunk_filter import is_clean

    for i in range(start, end):
        if (i - start) % 500 == 0:
            print(f"\r  {i - start:,} / {end - start:,} entries scanned ...", end="", flush=True)

        try:
            entry = archive._get_entry_by_id(i)  # private API; no public equivalent exists
        except Exception:
            counts["skipped"] += 1
            continue

        if entry.is_redirect:
            counts["skipped"] += 1
            continue

        try:
            item = entry.get_item()
        except Exception:
            counts["skipped"] += 1
            continue

        mime = item.mimetype
        content = bytes(item.content)
        title = entry.title or entry.path

        if "text/html" in mime:
            blocks = extract_html_blocks(content)
            has_content = any(len(b["text"]) >= 150 for b in blocks)
            if not has_content:
                counts["skipped"] += 1
                continue
            counts["html"] += 1
            for block in blocks:
                if len(block["text"]) < 150:
                    continue
                for chunk in splitter.split_text(block["text"]):
                    if quality_filter and not is_clean(chunk):
                        counts["filtered"] += 1
                        continue
                    yield {
                        "text": chunk,
                        "source": entry.path,
                        "title": title,
                        "is_accepted": block["is_accepted"],
                    }
            continue  # chunks already yielded above

        elif "application/json" in mime and entry.path.startswith("videos/") and entry.path.endswith(".json"):
            # Video channel ZIM format (e.g. s2underground): videos/<slug>.json
            # Contains title + description; no transcripts in this format.
            try:
                data = json.loads(content)
            except Exception:
                counts["skipped"] += 1
                continue
            desc = data.get("description", "").strip()
            vtitle = data.get("title", title).strip()
            text = f"{vtitle}\n\n{desc}" if desc else vtitle
            if len(text) < 150:
                counts["skipped"] += 1
                continue
            counts["video_desc"] = counts.get("video_desc", 0) + 1
            for chunk in splitter.split_text(text):
                if quality_filter and not is_clean(chunk):
                    counts["filtered"] += 1
                    continue
                yield {
                    "text": chunk,
                    "source": entry.path,
                    "title": vtitle,
                }
            continue

        elif "application/json" in mime and "page_content_" in entry.path:
            # LibreTexts ZIM format: content lives in page_content_*.json { htmlBody: "..." }
            try:
                html_body = json.loads(content).get("htmlBody", "")
            except Exception:
                counts["skipped"] += 1
                continue
            if not html_body:
                counts["skipped"] += 1
                continue
            blocks = extract_html_blocks(html_body.encode("utf-8"))
            has_content = any(len(b["text"]) >= 150 for b in blocks)
            if not has_content:
                counts["skipped"] += 1
                continue
            counts["json_html"] = counts.get("json_html", 0) + 1
            for block in blocks:
                if len(block["text"]) < 150:
                    continue
                for chunk in splitter.split_text(block["text"]):
                    if quality_filter and not is_clean(chunk):
                        counts["filtered"] += 1
                        continue
                    yield {
                        "text": chunk,
                        "source": entry.path,
                        "title": title,
                        "is_accepted": block["is_accepted"],
                    }
            continue

        elif "application/pdf" in mime:
            if not _PYPDF_AVAILABLE:
                counts["skipped"] += 1
                continue
            try:
                text, is_scanned = extract_pdf_text(content)
            except Exception:
                counts["pdf_error"] += 1
                continue
            if is_scanned or len(text) < 150:
                if ocr_engine is not None:
                    try:
                        from ocr import ocr_pdf
                        text = sanitize(ocr_pdf(content, ocr_engine))
                        if len(text) < 150:
                            counts["pdf_scanned"] += 1
                            continue
                        counts["pdf_ocr"] = counts.get("pdf_ocr", 0) + 1
                    except Exception:
                        counts["pdf_scanned"] += 1
                        continue
                else:
                    counts["pdf_scanned"] += 1
                    continue
            else:
                counts["pdf"] += 1

        else:
            counts["skipped"] += 1
            continue

        for chunk in splitter.split_text(text):
            if quality_filter and not is_clean(chunk):
                counts["filtered"] += 1
                continue
            yield {
                "text": chunk,
                "source": entry.path,
                "title": title,
            }

    print()  # end the progress line
    print(f"  HTML pages:        {counts['html']:,}")
    if counts.get("video_desc"):
        print(f"  Video desc (JSON): {counts['video_desc']:,}")
    if counts.get("json_html"):
        print(f"  JSON pages:        {counts['json_html']:,}")
    print(f"  PDFs (text):       {counts['pdf']:,}")
    if counts.get("pdf_ocr"):
        print(f"  PDFs (OCR'd):      {counts['pdf_ocr']:,}")
    if counts["pdf_scanned"]:
        print(f"  PDFs (scanned/empty, skipped): {counts['pdf_scanned']:,}")
    if counts["pdf_error"]:
        print(f"  PDFs (error, skipped):         {counts['pdf_error']:,}")
    print(f"  Other/redirects:   {counts['skipped']:,}")
    if counts["filtered"]:
        print(f"  Quality-filtered:  {counts['filtered']:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract text chunks from a Kiwix .zim file for RAG ingestion."
    )
    parser.add_argument("zim_file", help="Path to the .zim file")
    parser.add_argument(
        "--output", "-o",
        help="Output .jsonl file (default: <stem>_chunks.jsonl in the same directory)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512,
        help="Max characters per chunk (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=64,
        help="Overlap between consecutive chunks in characters (default: 64)",
    )
    parser.add_argument(
        "--ocr", action="store_true",
        help="OCR scanned/image PDFs instead of skipping them",
    )
    parser.add_argument(
        "--ocr-engine", default="tesseract", choices=["tesseract", "easyocr"],
        help="OCR engine to use (default: tesseract)",
    )
    parser.add_argument(
        "--entry-offset", type=int, default=0,
        help="First ZIM entry index to process (default: 0)",
    )
    parser.add_argument(
        "--entry-limit", type=int, default=0,
        help="Max number of entries to process (default: 0 = all remaining)",
    )
    parser.add_argument(
        "--filter", action="store_true",
        help="Drop chunks that score as ads or conspiracy content (uses chunk_filter.py)",
    )
    args = parser.parse_args()

    zim_path = Path(args.zim_file).expanduser().resolve()
    if not zim_path.exists():
        print(f"Error: file not found: {zim_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    elif args.entry_offset > 0 or args.entry_limit > 0:
        output_path = zim_path.parent / f"{zim_path.stem}_e{args.entry_offset:08d}_chunks.jsonl"
    else:
        output_path = zim_path.parent / f"{zim_path.stem}_chunks.jsonl"

    print(f"Input:        {zim_path}")
    print(f"Output:       {output_path}")
    print(f"Chunk size:   {args.chunk_size} chars  (overlap: {args.chunk_overlap})")
    print(f"PDF support:  {'yes (pypdf)' if _PYPDF_AVAILABLE else 'no — install pypdf'}")
    ocr_label = f"yes ({args.ocr_engine})" if args.ocr else "no (pass --ocr to enable)"
    print(f"OCR support:  {ocr_label}")
    print(f"Filter:       {'yes (chunk_filter.py)' if args.filter else 'no (pass --filter to enable)'}")
    print()

    archive = libzim.Archive(zim_path)
    total_entries = archive.all_entry_count
    entry_end = min(args.entry_offset + args.entry_limit, total_entries) if args.entry_limit > 0 else total_entries
    if args.entry_offset > 0 or args.entry_limit > 0:
        print(f"Entries in archive: {total_entries:,}  (processing {args.entry_offset:,} – {entry_end:,})")
    else:
        print(f"Entries in archive: {total_entries:,}")
    print()

    ocr_engine = None
    if args.ocr:
        from ocr import load_engine
        try:
            ocr_engine = load_engine(args.ocr_engine)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for chunk in iter_chunks(archive, splitter, ocr_engine=ocr_engine,
                                 entry_offset=args.entry_offset, entry_limit=args.entry_limit,
                                 quality_filter=args.filter):
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1

    print(f"\nDone — {count:,} chunks written to {output_path}")


if __name__ == "__main__":
    main()
