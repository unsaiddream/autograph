import base64
import io
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("fonts/Caveat-Regular.ttf")
NAME_COLOR = (15, 25, 100)  # Dark blue ballpoint pen
NAME_PADDING = 8  # px from left edge

_MONTHS_RU = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


def _format_date_ru(date_str: str) -> str:
    """Convert YYYY-MM-DD to Russian format: '2 марта 2026 г.'"""
    from datetime import datetime
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return f"{dt.day} {_MONTHS_RU[dt.month - 1]} {dt.year} г."
    except Exception:
        return date_str


def _name_for_zone(fullname: str, zone_type: str) -> str:
    """Return the portion of fullname appropriate for the zone type."""
    parts = fullname.split()
    if zone_type == "surname":
        return parts[0] if parts else fullname
    if zone_type == "firstname":
        return " ".join(parts[1:]) if len(parts) > 1 else fullname
    return fullname  # "fullname" or anything else


def _load_image_from_b64(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGBA")


def _image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _rotate_image(img: Image.Image, angle: float) -> Image.Image:
    """Rotate image with transparent background."""
    return img.rotate(angle, expand=True, resample=Image.BICUBIC)


def _place_signature(page: Image.Image, sig_b64: str, zone: dict) -> Image.Image:
    """Overlay signature PNG onto the page at the given zone coordinates."""
    if not sig_b64:
        return page

    sig = _load_image_from_b64(sig_b64)

    # Resize signature to fit zone
    zone_w = int(zone["width"])
    zone_h = int(zone["height"])
    sig_aspect = sig.width / sig.height if sig.height > 0 else 1
    zone_aspect = zone_w / zone_h if zone_h > 0 else 1

    if sig_aspect > zone_aspect:
        new_w = zone_w
        new_h = max(1, int(zone_w / sig_aspect))
    else:
        new_h = zone_h
        new_w = max(1, int(zone_h * sig_aspect))

    sig = sig.resize((new_w, new_h), Image.LANCZOS)

    x = int(zone["x"])
    y = int(zone["y"]) + (zone_h - new_h) // 2

    page = page.convert("RGBA")
    page.paste(sig, (x, y), sig)
    return page


def _place_stamp(page: Image.Image, stamp_b64: str, zone: dict) -> Image.Image:
    """Overlay stamp PNG onto the page at the given zone coordinates."""
    if not stamp_b64:
        return page

    stamp = _load_image_from_b64(stamp_b64)

    # Resize to fit zone
    zone_w = int(zone["width"])
    zone_h = int(zone["height"])
    stamp = stamp.resize((zone_w, zone_h), Image.LANCZOS)

    # Apply slight opacity
    r, g, b, a = stamp.split()
    a = a.point(lambda p: int(p * 0.85))
    stamp = Image.merge("RGBA", (r, g, b, a))

    page = page.convert("RGBA")
    page.paste(stamp, (int(zone["x"]), int(zone["y"])), stamp)
    return page


def _render_name(fullname: str, zone: dict, sign_date: str | None = None) -> Image.Image:
    """Render handwritten-style name/date onto a transparent image."""
    zone_type = zone.get("type", "fullname")
    zone_w = int(zone["width"])
    zone_h = int(zone["height"])

    if zone_type == "date":
        text = _format_date_ru(sign_date) if sign_date else ""
    else:
        text = _name_for_zone(fullname, zone_type)

    if not text:
        return Image.new("RGBA", (zone_w, zone_h), (0, 0, 0, 0))

    font_size = max(10, int(zone_h * 0.65))
    try:
        font = ImageFont.truetype(str(FONT_PATH), font_size)
    except Exception:
        font = ImageFont.load_default()

    canvas = Image.new("RGBA", (zone_w, zone_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((NAME_PADDING, (zone_h - font_size) // 2), text, font=font, fill=(*NAME_COLOR, 220))

    tilt = random.uniform(-2, 2)
    canvas = _rotate_image(canvas, tilt)
    canvas = canvas.crop((0, 0, zone_w, zone_h))
    return canvas


def _place_name(page: Image.Image, fullname: str, zone: dict, sign_date: str | None = None) -> Image.Image:
    """Render and overlay handwritten name/date on the page."""
    if not fullname:
        return page

    text_img = _render_name(fullname, zone, sign_date)
    page = page.convert("RGBA")
    page.paste(text_img, (int(zone["x"]), int(zone["y"])), text_img)
    return page


def overlay_page(
    page_b64: str,
    zones: list[dict],
    signature_b64: str | None,
    stamp_b64: str | None,
    fullname: str | None,
    sign_date: str | None,
) -> str:
    """
    Overlay all signatures/stamps/names onto a single page.
    Returns the modified page as base64 PNG.
    """
    page = _load_image_from_b64(page_b64).convert("RGBA")

    for zone in zones:
        zone_type = zone.get("type", "")

        if zone_type == "signature" and signature_b64:
            page = _place_signature(page, signature_b64, zone)

        elif zone_type == "stamp" and stamp_b64:
            page = _place_stamp(page, stamp_b64, zone)

        elif zone_type in ("fullname", "surname", "firstname") and fullname:
            page = _place_name(page, fullname, zone)

        elif zone_type == "date" and sign_date:
            page = _place_name(page, sign_date, zone, sign_date)

    # Convert to RGB for final output
    result = Image.new("RGB", page.size, (255, 255, 255))
    result.paste(page, mask=page.split()[3])
    return _image_to_b64(result, fmt="PNG")


def overlay_all_pages(
    pages_b64: list[str],
    all_zones: list[list[dict]],
    signature_b64: str | None,
    stamp_b64: str | None,
    fullname: str | None,
    sign_date: str | None,
) -> list[str]:
    """
    Overlay signatures/stamps/names on all pages.
    all_zones[i] is the list of zones for page i.
    """
    result_pages = []
    for i, (page_b64, zones) in enumerate(zip(pages_b64, all_zones)):
        modified = overlay_page(
            page_b64, zones, signature_b64, stamp_b64, fullname, sign_date
        )
        result_pages.append(modified)
    return result_pages
