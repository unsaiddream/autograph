"""
Run this script once to download the Caveat font from Google Fonts.
Usage: python download_fonts.py
"""
import urllib.request
from pathlib import Path

FONT_URL = "https://fonts.gstatic.com/s/caveat/v18/WnznHAc5bAfYB2QRah7pcpNvOx-pjcB9eIWpZQ.woff2"
# TTF direct link from GitHub mirror
FONT_TTF_URL = "https://github.com/googlefonts/caveat/raw/main/fonts/ttf/Caveat-Regular.ttf"
FONT_PATH = Path("fonts/Caveat-Regular.ttf")


def download_caveat():
    FONT_PATH.parent.mkdir(exist_ok=True)
    if FONT_PATH.exists():
        print(f"Font already exists: {FONT_PATH}")
        return

    print(f"Downloading Caveat-Regular.ttf...")
    try:
        req = urllib.request.Request(
            FONT_TTF_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
        FONT_PATH.write_bytes(data)
        print(f"✓ Font saved to {FONT_PATH}")
    except Exception as e:
        print(f"✗ Failed to download font: {e}")
        print("  Please manually download Caveat-Regular.ttf from Google Fonts")
        print("  and place it in the fonts/ directory.")


if __name__ == "__main__":
    download_caveat()
