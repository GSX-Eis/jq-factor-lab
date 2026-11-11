#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pypdf import PdfReader

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # noqa: BLE001
    RapidOCR = None


PAGE_BREAK = "\n\n---\n\n"


def extract_with_pdftotext(pdf_path: Path) -> str:
    proc = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore").strip() or "pdftotext failed")
    return proc.stdout.decode("utf-8", errors="ignore")


def extract_with_pypdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\f".join(pages)


def sort_ocr_lines(result: list[tuple]) -> list[str]:
    def key(item: tuple) -> tuple[float, float]:
        box = item[0]
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        return (min(ys), min(xs))

    ordered = sorted(result, key=key)
    return [str(item[1]).strip() for item in ordered if str(item[1]).strip()]


def extract_with_ocr(pdf_path: Path) -> str:
    if RapidOCR is None:
        raise RuntimeError("rapidocr_onnxruntime is not available")
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm is not available")

    with tempfile.TemporaryDirectory(prefix="pdf_ocr_", dir=str(pdf_path.parent)) as tmp_dir:
        prefix = Path(tmp_dir) / "page"
        proc = subprocess.run(
            [pdftoppm, "-r", "150", "-png", str(pdf_path), str(prefix)],
            check=False,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore").strip() or "pdftoppm failed")

        image_paths = sorted(Path(tmp_dir).glob("page-*.png"))
        if not image_paths:
            raise RuntimeError("pdftoppm produced no page images")

        ocr = RapidOCR()
        pages: list[str] = []
        for image_path in image_paths:
            result, _ = ocr(str(image_path))
            if not result:
                pages.append("")
                continue
            pages.append("\n".join(sort_ocr_lines(result)))
        return "\f".join(pages)


def extract_with_tesseract(pdf_path: Path) -> str:
    pdftoppm = shutil.which("pdftoppm")
    tesseract = shutil.which("tesseract")
    if not pdftoppm:
        raise RuntimeError("pdftoppm is not available")
    if not tesseract:
        raise RuntimeError("tesseract is not available")

    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", pdf_path.stem))
    languages = "chi_sim+eng" if has_cjk else "eng"

    with tempfile.TemporaryDirectory(prefix="pdf_ocr_", dir=str(pdf_path.parent)) as tmp_dir:
        prefix = Path(tmp_dir) / "page"
        proc = subprocess.run(
            [pdftoppm, "-r", "150", "-png", str(pdf_path), str(prefix)],
            check=False,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore").strip() or "pdftoppm failed")

        image_paths = sorted(Path(tmp_dir).glob("page-*.png"))
        if not image_paths:
            raise RuntimeError("pdftoppm produced no page images")

        pages: list[str] = []
        for image_path in image_paths:
            proc = subprocess.run(
                [tesseract, str(image_path), "stdout", "-l", languages, "--psm", "1"],
                check=False,
                capture_output=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    proc.stderr.decode("utf-8", errors="ignore").strip() or f"tesseract failed on {image_path.name}"
                )
            pages.append(proc.stdout.decode("utf-8", errors="ignore").strip())
        return "\f".join(pages)


def clean_text(text: str, title: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = text.replace("\f", PAGE_BREAK)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return f"# {title}\n"
    return f"# {title}\n\n{text}\n"


def build_extractors(prefer_tesseract: bool) -> list[tuple[str, object]]:
    extractors: list[tuple[str, object]] = [
        ("pdftotext", extract_with_pdftotext),
        ("pypdf", extract_with_pypdf),
        ("tesseract", extract_with_tesseract),
        ("ocr", extract_with_ocr),
    ]
    if prefer_tesseract:
        return [
            ("tesseract", extract_with_tesseract),
            ("ocr", extract_with_ocr),
            ("pdftotext", extract_with_pdftotext),
            ("pypdf", extract_with_pypdf),
        ]
    return extractors


def convert_pdf(pdf_path: Path, overwrite: bool, prefer_tesseract: bool) -> tuple[str, Path, str]:
    md_path = pdf_path.with_suffix(".md")
    if md_path.exists() and not overwrite:
        return ("skipped", md_path, "exists")

    errors: list[str] = []
    raw_text = ""
    for extractor_name, extractor in build_extractors(prefer_tesseract):
        try:
            raw_text = extractor(pdf_path)
            if raw_text.strip():
                break
            errors.append(f"{extractor_name}: empty output")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{extractor_name}: {exc}")
    else:
        md_path.write_text(f"# {pdf_path.stem}\n", encoding="utf-8")
        return ("empty", md_path, "; ".join(errors))

    md_path.write_text(clean_text(raw_text, pdf_path.stem), encoding="utf-8")
    return ("ok", md_path, "")


def iter_pdfs(target: Path) -> list[Path]:
    if target.is_file() and target.suffix.lower() == ".pdf":
        return [target]
    return sorted(p for p in target.rglob("*.pdf") if p.is_file())


def is_effectively_empty_markdown(md_path: Path) -> bool:
    if not md_path.exists():
        return False
    text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
    return text in {f"# {md_path.stem}", f"# {md_path.stem}\n"} or len(text) <= len(md_path.stem) + 4


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PDFs to same-name Markdown files.")
    parser.add_argument("target", type=Path, help="PDF file or directory containing PDFs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Markdown files")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel workers")
    parser.add_argument(
        "--retry-empty-only",
        action="store_true",
        help="Only process PDFs whose same-name Markdown exists but is effectively empty",
    )
    parser.add_argument(
        "--prefer-tesseract",
        action="store_true",
        help="Try Tesseract OCR before text-layer extraction",
    )
    args = parser.parse_args()

    pdfs = iter_pdfs(args.target)
    if args.retry_empty_only:
        pdfs = [pdf_path for pdf_path in pdfs if is_effectively_empty_markdown(pdf_path.with_suffix(".md"))]
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 1

    counts = {"ok": 0, "skipped": 0, "empty": 0}
    empties: list[tuple[Path, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(convert_pdf, pdf_path, args.overwrite, args.prefer_tesseract): pdf_path
            for pdf_path in pdfs
        }
        for future in as_completed(futures):
            status, path, detail = future.result()
            counts[status] += 1
            if status == "empty":
                empties.append((path, detail))

    print(
        f"converted={counts['ok']} skipped={counts['skipped']} empty={counts['empty']} total={len(pdfs)}"
    )
    for path, detail in empties:
        print(f"EMPTY\t{path}\t{detail}")
    return 0 if counts["empty"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
