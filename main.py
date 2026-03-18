import json
import logging
import os
import shutil
import traceback
from pathlib import Path

import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autograph")

load_dotenv()

from database import (
    cleanup_old_sessions,
    create_session,
    get_session,
    update_session,
)
from document_processor import get_page_sizes_from_b64, process_upload
from fallback_zones import ensure_zones
from gemini_analyzer import analyze_all_pages
from scan_effect import apply_scan_effect_all, pages_to_pdf
from signature_overlay import overlay_all_pages

app = FastAPI(title="Autograph", version="1.0.0")

from fastapi import Request
from fastapi.responses import JSONResponse as _JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s:\n%s", request.url, traceback.format_exc())
    return _JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Serve the main frontend."""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload PDF or DOCX document.
    Returns session_id and page images as base64.
    """
    cleanup_old_sessions()

    # Validate file type
    filename = file.filename or "document"
    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(400, "Only PDF and DOCX files are supported")

    # Save uploaded file
    save_path = UPLOAD_DIR / f"{filename}"
    # Use unique name to avoid collisions
    import uuid
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / unique_name

    async with aiofiles.open(save_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Create session
    session_id = create_session(
        file_path=str(save_path),
        file_name=filename,
        file_type=file.content_type or "",
    )

    try:
        # Convert document to page images
        pages_b64, sizes = process_upload(
            str(save_path), file.content_type or "", session_id
        )

        # Store in session
        update_session(
            session_id,
            page_count=len(pages_b64),
            pages_b64=json.dumps(pages_b64),
        )

        return {
            "session_id": session_id,
            "page_count": len(pages_b64),
            "pages": pages_b64,
            "sizes": sizes,
            "file_name": filename,
        }
    except Exception as e:
        logger.error("Upload error:\n%s", traceback.format_exc())
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to process document: {type(e).__name__}: {e}")


@app.post("/analyze")
async def analyze_document(
    session_id: str = Form(...),
):
    """
    Run Gemini Vision analysis on all pages.
    Returns detected zones per page.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    pages_b64 = json.loads(session["pages_b64"])
    if not pages_b64:
        raise HTTPException(400, "No pages found in session")

    page_sizes = get_page_sizes_from_b64(pages_b64)

    try:
        zones_per_page = analyze_all_pages(pages_b64, page_sizes)
        update_session(session_id, zones=json.dumps(zones_per_page))

        return {
            "session_id": session_id,
            "zones_per_page": zones_per_page,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Gemini analysis failed: {str(e)}")


@app.post("/preview")
async def preview_document(
    session_id: str = Form(...),
    signature_b64: str = Form(default=""),
    stamp_b64: str = Form(default=""),
    fullname: str = Form(default=""),
    sign_date: str = Form(default=""),
    zones_override: str = Form(default=""),  # JSON override for zone positions
):
    """
    Overlay signature/stamp/name on all pages and return preview images.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    pages_b64 = json.loads(session["pages_b64"])
    stored_zones = json.loads(session["zones"] or "[]")

    # Use override zones if provided
    if zones_override:
        try:
            all_zones = json.loads(zones_override)
        except Exception:
            all_zones = stored_zones
    else:
        all_zones = stored_zones

    normalized = ensure_zones(all_zones, session.get("file_path"), pages_b64)

    # Pad if needed
    while len(normalized) < len(pages_b64):
        normalized.append([])

    # Save signature/stamp/name to session
    update_session(
        session_id,
        signature_b64=signature_b64 or None,
        stamp_b64=stamp_b64 or None,
        fullname=fullname or None,
        sign_date=sign_date or None,
    )

    try:
        preview_pages = overlay_all_pages(
            pages_b64=pages_b64,
            all_zones=normalized,
            signature_b64=signature_b64 or None,
            stamp_b64=stamp_b64 or None,
            fullname=fullname or None,
            sign_date=sign_date or None,
        )

        return {
            "session_id": session_id,
            "pages": preview_pages,
            "page_count": len(preview_pages),
        }
    except Exception as e:
        raise HTTPException(500, f"Preview generation failed: {str(e)}")


@app.post("/export")
async def export_document(
    session_id: str = Form(...),
    zones_override: str = Form(default=""),
    signature_b64: str = Form(default=""),
    stamp_b64: str = Form(default=""),
    fullname: str = Form(default=""),
    sign_date: str = Form(default=""),
):
    """
    Apply scan effect and export final PDF for download.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    pages_b64 = json.loads(session["pages_b64"])
    stored_zones = json.loads(session["zones"] or "[]")

    # Prefer form-submitted values; fall back to session
    signature_b64 = signature_b64 or session.get("signature_b64") or None
    stamp_b64 = stamp_b64 or session.get("stamp_b64") or None
    fullname = fullname or session.get("fullname") or None
    sign_date = sign_date or session.get("sign_date") or None

    # Use override zones if provided
    if zones_override:
        try:
            all_zones = json.loads(zones_override)
        except Exception:
            all_zones = stored_zones
    else:
        all_zones = stored_zones

    normalized = ensure_zones(all_zones, session.get("file_path"), pages_b64)

    while len(normalized) < len(pages_b64):
        normalized.append([])

    logger.info(
        "EXPORT: fullname=%r sign_date=%r has_sig=%s has_stamp=%s "
        "pages=%d zones_per_page=%s",
        fullname, sign_date,
        bool(signature_b64), bool(stamp_b64),
        len(pages_b64),
        [len(z) for z in normalized],
    )

    try:
        # Step 1: Overlay signatures
        overlaid = overlay_all_pages(
            pages_b64=pages_b64,
            all_zones=normalized,
            signature_b64=signature_b64,
            stamp_b64=stamp_b64,
            fullname=fullname,
            sign_date=sign_date,
        )

        # Step 2: Apply scan effect
        scanned = apply_scan_effect_all(overlaid)

        # Step 3: Build PDF
        pdf_bytes = pages_to_pdf(scanned)

        # Clean up uploaded file
        file_path = session.get("file_path")
        if file_path:
            Path(file_path).unlink(missing_ok=True)
            # Also try PDF version if DOCX was converted
            pdf_path = str(Path(file_path).with_suffix(".pdf"))
            Path(pdf_path).unlink(missing_ok=True)

        original_name = session.get("file_name", "document")
        stem = Path(original_name).stem
        download_name = f"{stem}_signed_scanned.pdf"

        # RFC 5987 encoding for non-ASCII filenames (e.g. Cyrillic)
        from urllib.parse import quote
        encoded_name = quote(download_name, safe="")
        content_disposition = (
            f"attachment; filename=\"{download_name.encode('ascii', 'replace').decode()}\"; "
            f"filename*=UTF-8''{encoded_name}"
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": content_disposition},
        )
    except Exception as e:
        raise HTTPException(500, f"Export failed: {str(e)}")


@app.get("/session/{session_id}")
async def get_session_state(session_id: str):
    """Get current session state."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Don't return heavy base64 data, just metadata
    return {
        "session_id": session_id,
        "file_name": session.get("file_name"),
        "file_type": session.get("file_type"),
        "page_count": session.get("page_count", 0),
        "has_signature": bool(session.get("signature_b64")),
        "has_stamp": bool(session.get("stamp_b64")),
        "fullname": session.get("fullname"),
        "sign_date": session.get("sign_date"),
        "has_zones": bool(session.get("zones") and session["zones"] != "[]"),
        "created_at": session.get("created_at"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
