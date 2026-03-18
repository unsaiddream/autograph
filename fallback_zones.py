"""
Fallback zone detector: uses PyMuPDF text extraction to find signature-related
keywords in the PDF and derive zone coordinates without Gemini.

Keywords searched (case-insensitive, parentheses optional):
  signature : подпись, signature, підпис, қолы
  fullname  : фамилия, ф.и.о, фио, имя, отчество, fullname, аты-жөні, аты
  date      : дата, date, күні
  stamp     : печать, м.п., stamp, l.s., мөр
"""

import logging
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

_log = logging.getLogger("autograph.fallback")

# DPI used when rendering page images in document_processor.py
RENDER_DPI = 150
_SCALE = RENDER_DPI / 72.0   # points → pixels

_SIG_RE   = re.compile(r"подпись|signature|підпис|қолы", re.IGNORECASE)
_NAME_RE  = re.compile(r"фамилия|ф\.и\.о|фио|имя.*отчество|отчество|fullname|аты-жөні|аты", re.IGNORECASE)
_DATE_RE  = re.compile(r"\bдата\b|\bdate\b|күні", re.IGNORECASE)
_STAMP_RE = re.compile(r"печать|м\.п\.|stamp|l\.s\.|мөр", re.IGNORECASE)


def _pt(v: float) -> float:
    return v * _SCALE


def _zone(zone_id: str, zone_type: str, label: str,
          x: float, y: float, w: float, h: float) -> dict:
    return {
        "id": zone_id,
        "type": zone_type,
        "label": label,
        "x": round(x, 1),
        "y": round(y, 1),
        "width": round(w, 1),
        "height": round(h, 1),
        "confidence": 0.85,
        "source": "fallback",
    }


def detect_zones_from_pdf(pdf_path: str) -> list[dict]:
    """
    Open the PDF, scan every page for signature-related text spans,
    and return a list of zone dicts (same structure as Gemini output)
    grouped by page: [{"page": 1, "zones": [...]}, ...].
    """
    doc = fitz.open(pdf_path)
    results: list[dict] = []

    for page_num, page in enumerate(doc):
        pw = page.rect.width   # points
        ph = page.rect.height  # points
        zones: list[dict] = []
        sig_count = name_count = date_count = stamp_count = 0

        # Log all text on the page so we can diagnose keyword misses
        all_text = page.get_text("text")
        _log.info("FALLBACK page %d text (first 400 chars): %r", page_num + 1, all_text[:400])

        # get_text("rawdict") gives per-span info with bbox
        raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    bbox = span["bbox"]           # (x0, y0, x1, y1) in points
                    sx, sy, ex, ey = bbox
                    sw = ex - sx                  # span width  (points)
                    sh = ey - sy                  # span height (points)
                    if sh <= 0:
                        sh = span.get("size", 10)

                    # ---- Signature ---------------------------------------------------
                    if _SIG_RE.search(text):
                        sig_count += 1
                        # Place the zone above the label text
                        zone_h = max(sh * 2.5, 12)     # points
                        zone_w = max(sw * 2.0, pw * 0.25)
                        zone_x = sx
                        zone_y = sy - zone_h - sh * 0.5
                        # Make sure we don't go above the page
                        zone_y = max(zone_y, 0)
                        zones.append(_zone(
                            f"sign_{sig_count}", "signature", text,
                            _pt(zone_x), _pt(zone_y), _pt(zone_w), _pt(zone_h),
                        ))

                    # ---- Full name ---------------------------------------------------
                    elif _NAME_RE.search(text):
                        name_count += 1
                        zone_h = max(sh * 1.8, 10)
                        zone_w = max(sw * 1.8, pw * 0.30)
                        zone_x = sx
                        zone_y = sy - zone_h - sh * 0.3
                        zone_y = max(zone_y, 0)
                        zones.append(_zone(
                            f"name_{name_count}", "fullname", text,
                            _pt(zone_x), _pt(zone_y), _pt(zone_w), _pt(zone_h),
                        ))

                    # ---- Date --------------------------------------------------------
                    elif _DATE_RE.search(text):
                        date_count += 1
                        zone_h = max(sh * 1.5, 10)
                        zone_w = max(sw * 2.0, pw * 0.20)
                        zone_x = sx
                        zone_y = sy - zone_h * 0.5
                        zone_y = max(zone_y, 0)
                        zones.append(_zone(
                            f"date_{date_count}", "date", text,
                            _pt(zone_x), _pt(zone_y), _pt(zone_w), _pt(zone_h),
                        ))

                    # ---- Stamp -------------------------------------------------------
                    elif _STAMP_RE.search(text):
                        stamp_count += 1
                        side = max(sh * 4, pw * 0.15)
                        zone_x = sx - side * 0.2
                        zone_y = sy - side * 0.5
                        zone_x = max(zone_x, 0)
                        zone_y = max(zone_y, 0)
                        zones.append(_zone(
                            f"stamp_{stamp_count}", "stamp", text,
                            _pt(zone_x), _pt(zone_y), _pt(side), _pt(side),
                        ))

        results.append({"page": page_num + 1, "zones": zones})

    doc.close()
    return results


def ensure_zones(
    zones_per_page: list[dict],
    pdf_path: Optional[str],
) -> list[list[dict]]:
    """
    Given the stored zones list, return a normalised list[list[dict]] ready
    for overlay_all_pages().  If every page has zero zones AND a pdf_path is
    provided, run the fallback detector and return its zones instead.
    """
    # Normalise whatever format is stored
    if zones_per_page and isinstance(zones_per_page[0], dict) and "zones" in zones_per_page[0]:
        normalized = [p.get("zones", []) for p in zones_per_page]
    elif zones_per_page and isinstance(zones_per_page[0], list):
        normalized = zones_per_page  # type: ignore[assignment]
    else:
        normalized = []

    total_zones = sum(len(z) for z in normalized)
    if total_zones > 0:
        return normalized

    # No zones at all — try fallback
    if not pdf_path:
        _log.warning("ensure_zones: no pdf_path provided, cannot run fallback")
        return normalized
    if not Path(pdf_path).exists():
        _log.warning("ensure_zones: pdf_path %r does not exist (already deleted?)", pdf_path)
        return normalized

    _log.info("ensure_zones: running PDF fallback detector on %r", pdf_path)
    try:
        fallback = detect_zones_from_pdf(pdf_path)
        fb_normalized = [p.get("zones", []) for p in fallback]
        _log.info("ensure_zones: fallback found zones_per_page=%s", [len(z) for z in fb_normalized])
        return fb_normalized
    except Exception as exc:
        _log.error("ensure_zones: fallback failed: %s", exc, exc_info=True)
        return normalized
