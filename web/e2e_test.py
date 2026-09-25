"""End-to-end check of the built site in headless Chromium (Playwright).

    python web/e2e_test.py _site [--upload clip.webm --roi "x1,y1,..."] [--screenshot shot.png]

Serves the site, runs the synthetic demo clip and checks the detected state
changes against its ground truth. With --upload it also analyses a local file
with the YOLO detector (needs a build with --model).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

EXPECTED = ["WORKING", "STOPPED / FAULT", "WORKING", "IDLE"]


def serve(root: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/"


def wait_for_run(page, timeout_ms: int = 300_000) -> dict:
    page.wait_for_function("() => window.escalatorMonitor && window.escalatorMonitor.run && !window.escalatorMonitor.busy",
                           timeout=timeout_ms)  # fmt: skip
    return page.evaluate(
        """() => {
          const r = window.escalatorMonitor.run;
          return {events: r.events.map(e => e.event), summary: r.summary, frames: r.frames.length,
                  gt: document.getElementById('gt-note').textContent,
                  error: document.getElementById('error').hidden ? '' : document.getElementById('error').textContent};
        }"""
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site")
    parser.add_argument("--upload", help="video file to analyse with the YOLO detector")
    parser.add_argument("--roi", help="corners for --upload, in video pixels")
    parser.add_argument("--screenshot")
    parser.add_argument("--chromium", help="path to a Chromium executable")
    args = parser.parse_args()

    server, url = serve(Path(args.site))
    errors: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.chromium) if args.chromium else p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(url)
            page.get_by_text("Synthetic clip: running, fault, running, idle").click()
            page.wait_for_function("() => document.getElementById('roi-text').value.length > 0", timeout=60_000)
            page.click("#analyse-btn")
            result = wait_for_run(page)
            print("demo events:", result["events"])
            print("demo:", result["gt"])
            transitions = [e.split(" -> ")[1] for e in result["events"] if " -> " in e]
            ok = (
                transitions[1:] == EXPECTED[1:]
                and transitions[0] == "WORKING"
                and result["summary"]["fault_count"] == 1
            )
            if args.screenshot:
                page.screenshot(path=args.screenshot, full_page=True)
            if not ok:
                print("FAIL: unexpected state changes", transitions, result["error"])
                return 1

            if args.upload:
                page.set_input_files("#file-input", args.upload)
                page.wait_for_function("() => !document.getElementById('step-roi').hidden", timeout=60_000)
                page.wait_for_timeout(500)
                if args.roi:
                    page.fill("#roi-text", args.roi)
                    page.dispatch_event("#roi-text", "change")
                page.evaluate("() => { window.escalatorMonitor.run = null; }")
                page.click("#analyse-btn")
                page.wait_for_function("() => window.escalatorMonitor.busy", timeout=10_000)
                upload = wait_for_run(page)
                print("upload:", upload["frames"], "frames, detector:", upload["summary"]["detector"],
                      "events:", upload["events"], upload["error"])  # fmt: skip
                if upload["error"] or not upload["frames"]:
                    return 1
            browser.close()
    finally:
        server.shutdown()
    if errors:
        print("page errors:", errors)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
