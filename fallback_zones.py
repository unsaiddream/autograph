"""
Fallback zone detector — three strategies, tried in order:

1. PyMuPDF text extraction: finds signature keywords in a text-layer PDF.
2. OpenCV horizontal line detection: detects blank underlines in the page image.
3. Hardcoded defaults: places zones at fixed fractions of the page size.

The first strategy that yields at least one zone wins.
"""

import base64
import logging
import re
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

_log = logging.getLogger("autograph.fallback")

# DPI used when rendering page images in document_processor.py
RENDER_DPI = 150
_SCALE = RENDER_DPI / 72.0          # PDF points → image pixels

# ── keyword regexes ──────────────────────────────────────────────────────────
# Applied after Unicode NFKC normalisation on each text span.

# Matches the APPLICANT's signature label only.
# Excluded: "(фио, подпись)" or "(дата, подпись)" — those are manager labels.
_SIG_ONLY_RE   = re.compile(r"^[\(\s]*подпись[\)\s]*$", re.IGNORECASE)
_SIG_RE        = re.compile(r"подпись|signature|підпис|қолы", re.IGNORECASE)

# "(фамилия)", "(имя, отчество полностью)", "ФИО" etc.
_NAME_RE  = re.compile(
    r"фамилия|имя.*отчество|отчество.*имя|fullname|аты.жөні|^фио$",
    re.IGNORECASE,
)

# Date labels (standalone "дата" / "date" words)
_DATE_RE  = re.compile(r"^\W*дата\W*$|^\W*date\W*$|күні", re.IGNORECASE)

# Stamp labels
_STAMP_RE = re.compile(r"печать|м\.п\.|stamp|l\.s\.|мөр", re.IGNORECASE)

# Combined manager labels that should NOT get applicant zones
_COMBINED_RE = re.compile(
    r"(фио|имя|фамилия).{0,10}(подпись)|(подпись).{0,10}(фио|имя|фамилия)",
    re.IGNORECASE,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _px(pts: float) -> float:
    return pts * _SCALE


def _zone(zid: str, ztype: str, label: str,
          x: float, y: float, w: float, h: float,
          source: str = "text") -> dict:
    return {
        "id": zid, "type": ztype, "label": label,
        "x": round(x, 1), "y": round(y, 1),
        "width": round(w, 1), "height": round(h, 1),
        "confidence": 0.85, "source": source,
    }


# ── Strategy 1: PyMuPDF text extraction ──────────────────────────────────────

def _detect_from_pdf_text(pdf_path: str) -> list[list[dict]]:
    """Return [[zones_page0], ...] using PDF text spans."""
    doc = fitz.open(pdf_path)
    results: list[list[dict]] = []

    for page_num, page in enumerate(doc):
        pw = page.rect.width
        zones: list[dict] = []
        sig_n = name_n = date_n = stamp_n = 0

        plain = page.get_text("text")
        _log.info("PDF text page %d (first 600 chars): %r", page_num + 1, plain[:600])

        raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    raw_text = span.get("text", "").strip()
                    if not raw_text:
                        continue
                    text = unicodedata.normalize("NFKC", raw_text)

                    x0, y0, x1, y1 = span["bbox"]
                    sw, sh = x1 - x0, y1 - y0
                    if sh <= 0:
                        sh = span.get("size", 10)

                    # Skip manager combined labels like "(фио, подпись)"
                    if _COMBINED_RE.search(text):
                        _log.debug("Skipping combined manager label: %r", raw_text)
                        continue

                    # ---- Signature (applicant only) --------------------------
                    # Accept "(подпись)" but reject if it also contains name words
                    if _SIG_RE.search(text) and not _NAME_RE.search(text):
                        sig_n += 1
                        zh = max(sh * 2.5, 12)
                        zw = max(sw * 2.0, pw * 0.25)
                        zy = max(y0 - zh - sh * 0.5, 0)
                        zones.append(_zone(f"sign_{sig_n}", "signature", raw_text,
                                           _px(x0), _px(zy), _px(zw), _px(zh)))
                        _log.info("  → signature zone at y=%.0f from %r", _px(zy), raw_text)

                    # ---- Full name -------------------------------------------
                    elif _NAME_RE.search(text):
                        name_n += 1
                        zh = max(sh * 1.8, 10)
                        zw = max(sw * 2.2, pw * 0.35)
                        zy = max(y0 - zh - sh * 0.3, 0)
                        zones.append(_zone(f"name_{name_n}", "fullname", raw_text,
                                           _px(x0), _px(zy), _px(zw), _px(zh)))
                        _log.info("  → fullname zone at y=%.0f from %r", _px(zy), raw_text)

                    # ---- Date ------------------------------------------------
                    elif _DATE_RE.search(text):
                        date_n += 1
                        zh = max(sh * 1.5, 10)
                        zw = max(sw * 2.5, pw * 0.22)
                        zy = max(y0 - zh * 0.5, 0)
                        zones.append(_zone(f"date_{date_n}", "date", raw_text,
                                           _px(x0), _px(zy), _px(zw), _px(zh)))

                    # ---- Stamp -----------------------------------------------
                    elif _STAMP_RE.search(text):
                        stamp_n += 1
                        side = max(sh * 4, pw * 0.15)
                        zx = max(x0 - side * 0.2, 0)
                        zy = max(y0 - side * 0.5, 0)
                        zones.append(_zone(f"stamp_{stamp_n}", "stamp", raw_text,
                                           _px(zx), _px(zy), _px(side), _px(side)))

        _log.info("PDF text page %d: found %d zones", page_num + 1, len(zones))
        results.append(zones)

    doc.close()
    return results


# ── Strategy 2: OpenCV horizontal line detection ──────────────────────────────

def _detect_from_image_lines(pages_b64: list[str]) -> list[list[dict]]:
    """
    Detect long blank underlines in each page image.

    Heuristics for Russian official documents (A4 portrait):
      - Top-right area   (y < 28%, x_center > 38%) → fullname
      - Bottom-right area (55% < y < 80%, x_center > 42%) → signature
    Lines not matching either region are ignored.
    Minimum line length = 20% of page width to skip underlined text formatting.
    """
    all_zones: list[list[dict]] = []

    for page_idx, page_b64 in enumerate(pages_b64):
        zones: list[dict] = []

        img_data = base64.b64decode(page_b64)
        pil_img = Image.open(BytesIO(img_data)).convert("L")
        img = np.array(pil_img)
        h, w = img.shape

        # Binarise (dark pixels → 255)
        _, binary = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY_INV)

        # Morphological opening to isolate long horizontal lines
        # Use 20% of page width as minimum — filters out short underlined text
        min_len = max(int(w * 0.20), 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Dilate slightly to merge fragmented segments
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
        h_lines = cv2.dilate(h_lines, dilate_k, iterations=1)

        contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            # Only thin horizontal lines (height ≤ 0.7% of page)
            if cw >= min_len and ch <= max(8, int(h * 0.007)):
                boxes.append((cx, cy, cw, ch))

        # Sort top → bottom, merge rows within 8 px
        boxes.sort(key=lambda b: b[1])
        merged: list[list[int]] = []
        for cx, cy, cw, ch in boxes:
            if merged and abs(cy - merged[-1][1]) < 8:
                px, py, pw2, ph2 = merged[-1]
                nx = min(px, cx)
                merged[-1] = [nx, min(py, cy),
                               max(px + pw2, cx + cw) - nx, max(ph2, ch)]
            else:
                merged.append([cx, cy, cw, ch])

        _log.info("Image detector page %d: %d candidate lines", page_idx + 1, len(merged))

        zone_idx = 0
        zone_h = max(int(h * 0.030), 28)

        for lx, ly, lw, lh in merged:
            y_frac = ly / h
            x_center_frac = (lx + lw / 2) / w

            _log.debug("  line: y_frac=%.2f x_center=%.2f width=%d", y_frac, x_center_frac, lw)

            if y_frac < 0.28 and x_center_frac > 0.38:
                # Top-right → fullname field
                zone_idx += 1
                zones.append(_zone(
                    f"name_{zone_idx}", "fullname", "detected line",
                    float(lx), float(max(0, ly - zone_h)),
                    float(lw), float(zone_h), source="image",
                ))
                _log.info("  → fullname at y=%.0f (%.0f%%)", ly, y_frac * 100)

            elif 0.50 < y_frac < 0.80 and x_center_frac > 0.42:
                # Bottom-right → applicant signature field
                zone_idx += 1
                zones.append(_zone(
                    f"sign_{zone_idx}", "signature", "detected line",
                    float(lx), float(max(0, ly - zone_h)),
                    float(lw), float(zone_h), source="image",
                ))
                _log.info("  → signature at y=%.0f (%.0f%%)", ly, y_frac * 100)

        # Add date zone just below the first signature zone
        for z in zones:
            if z["type"] == "signature":
                date_y = z["y"] + z["height"] + int(h * 0.015)
                if date_y + zone_h < h:
                    zone_idx += 1
                    zones.append(_zone(
                        f"date_{zone_idx}", "date", "detected date area",
                        z["x"], float(date_y),
                        z["width"] * 0.70, float(zone_h), source="image",
                    ))
                break

        _log.info("Image detector page %d: %d zones total", page_idx + 1, len(zones))
        all_zones.append(zones)

    return all_zones


# ── Strategy 3: Hardcoded defaults ───────────────────────────────────────────

def _default_zones(pages_b64: list[str]) -> list[list[dict]]:
    """Fixed-fraction zones for typical A4 Russian application forms."""
    all_zones: list[list[dict]] = []
    for page_b64 in pages_b64:
        img_data = base64.b64decode(page_b64)
        pil_img = Image.open(BytesIO(img_data))
        pw, ph = pil_img.size
        zones = [
            _zone("name_1", "fullname", "default surname",
                  pw * 0.42, ph * 0.112, pw * 0.50, ph * 0.032, source="default"),
            _zone("name_2", "fullname", "default firstname",
                  pw * 0.42, ph * 0.162, pw * 0.50, ph * 0.032, source="default"),
            _zone("sign_1", "signature", "default signature",
                  pw * 0.53, ph * 0.625, pw * 0.40, ph * 0.038, source="default"),
            _zone("date_1", "date", "default date",
                  pw * 0.53, ph * 0.678, pw * 0.28, ph * 0.028, source="default"),
        ]
        _log.info("Using hardcoded default zones (page %dx%d)", pw, ph)
        all_zones.append(zones)
    return all_zones


# ── Public entry point ────────────────────────────────────────────────────────

def ensure_zones(
    zones_per_page: list,
    pdf_path: Optional[str],
    pages_b64: Optional[list[str]] = None,
) -> list[list[dict]]:
    """
    Return normalised list[list[dict]] for overlay_all_pages().
    Tries three fallback strategies when stored zones are empty.
    """
    # Normalise stored zones
    if zones_per_page and isinstance(zones_per_page[0], dict) and "zones" in zones_per_page[0]:
        normalized: list[list[dict]] = [p.get("zones", []) for p in zones_per_page]
    elif zones_per_page and isinstance(zones_per_page[0], list):
        normalized = zones_per_page  # type: ignore[assignment]
    else:
        normalized = []

    if sum(len(z) for z in normalized) > 0:
        _log.info("ensure_zones: using stored Gemini zones %s", [len(z) for z in normalized])
        return normalized

    _log.info("ensure_zones: no stored zones — running fallbacks")

    # Strategy 1: PDF text extraction
    if pdf_path:
        candidate = Path(pdf_path)
        if candidate.suffix.lower() in (".doc", ".docx"):
            candidate = candidate.with_suffix(".pdf")
        if candidate.exists():
            try:
                result = _detect_from_pdf_text(str(candidate))
                if any(result):
                    _log.info("ensure_zones: PDF text → %s zones", [len(z) for z in result])
                    return result
                _log.info("ensure_zones: PDF text found no zones (image-only PDF?)")
            except Exception as exc:
                _log.error("ensure_zones: PDF text failed: %s", exc)
        else:
            _log.warning("ensure_zones: PDF not available (%s)", candidate)

    # Strategy 2: image line detection
    if pages_b64:
        try:
            result = _detect_from_image_lines(pages_b64)
            if any(result):
                _log.info("ensure_zones: image lines → %s zones", [len(z) for z in result])
                return result
            _log.info("ensure_zones: image line detection found no zones")
        except Exception as exc:
            _log.error("ensure_zones: image line detection failed: %s", exc)

    # Strategy 3: hardcoded defaults
    if pages_b64:
        _log.info("ensure_zones: using hardcoded defaults")
        return _default_zones(pages_b64)

    return normalized
