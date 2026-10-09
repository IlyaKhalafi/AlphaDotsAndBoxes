"""Exercise the live UI and record a real self-play game for the README."""

import argparse
import io
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as browser_api:
        browser = browser_api.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 820}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_selector(".edge-control[role=button]")
        page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert page.locator(".game-panel").bounding_box()["y"] < 150
        assert "Connect the dots. Close a box." not in page.locator("body").inner_text()
        assert page.locator(".rules-help").get_attribute("open") is None
        page.locator(".rules-help summary").click()
        assert page.locator(".rules-help p").is_visible()
        page.locator(".rules-help summary").click()
        # Verify hint, a real human move, the agent response, and undo.
        page.locator("#budget").select_option("32")
        page.locator("#hint").click()
        page.wait_for_selector(".hinted")
        page.locator(".hinted").click()
        page.wait_for_function("!document.getElementById('undo').disabled")
        page.locator("#undo").click()
        page.wait_for_function("document.getElementById('human-score').textContent === '0'")
        page.locator(".edge-control[role=button]").first.focus()
        page.locator(".edge-control[role=button]").first.press("Enter")
        page.wait_for_function("!document.getElementById('undo').disabled")
        page.locator("#undo").click()
        page.wait_for_function("document.querySelectorAll('.drawn-edge').length === 0")
        # Start over and capture the actual graph agent playing both seats.
        page.locator("#watch").click()
        page.wait_for_function(
            "document.getElementById('board-label').textContent.includes('SELF-PLAY')"
        )
        page.locator("#watch").click()
        page.wait_for_function("!document.getElementById('watch').disabled")
        assert page.locator("#turn-title").inner_text() == "Paused."
        assert page.locator(".edge-control[role=button]").count() == 0
        paused_edges = page.locator(".drawn-edge").count()
        page.wait_for_timeout(800)
        assert page.locator(".drawn-edge").count() == paused_edges
        page.locator("#watch").click()
        frames, started = [], time.monotonic()
        while time.monotonic() - started < 90:
            frames.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
            if page.locator("#live-pill").inner_text().strip() == "BOARD COMPLETE":
                break
            page.wait_for_timeout(500)
        else:
            raise RuntimeError("Self-play did not finish within 90 seconds.")
        assert "Watch game" in page.locator("#watch").inner_text()
        page.screenshot(path=str(args.output / "desktop-played.png"), full_page=True)
        frames.extend([frames[-1]] * 4)
        # One shared palette avoids flickering colors between frames.
        palette = frames[-1].resize((960, 615)).quantize(colors=96)
        indexed = [frame.resize((960, 615)).quantize(palette=palette) for frame in frames]
        indexed[0].save(
            args.output / "self-play.gif",
            save_all=True,
            append_images=indexed[1:],
            duration=500,
            loop=0,
            optimize=True,
        )
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        mobile.goto(args.url)
        mobile.wait_for_selector(".edge-control[role=button]")
        assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert mobile.locator("#new-game").bounding_box()["y"] < 650
        mobile.screenshot(path=str(args.output / "mobile.png"), full_page=True)
        # A rectangular custom board must remain playable.
        mobile.locator(".custom-size summary").click()
        mobile.locator("#rows").fill("2")
        mobile.locator("#cols").fill("5")
        mobile.locator("#new-game").click()
        mobile.wait_for_function(
            "document.getElementById('board-label').textContent.startsWith('2 × 5')"
        )
        assert mobile.locator(".edge-control[role=button]").count() == 27
        mobile.screenshot(path=str(args.output / "mobile-rectangular.png"), full_page=True)
        browser.close()
        if errors:
            raise RuntimeError("Browser errors: " + "; ".join(errors))
        print(f"Browser checks passed. Saved {len(indexed)} real UI frames to {args.output}.")


if __name__ == "__main__":
    main()
