import sys

from playwright.sync_api import sync_playwright

url, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1400, "height": 1350}, device_scale_factor=1)
    pg.goto(url)
    pg.wait_for_selector("[data-testid='stImage'] img", timeout=180000)
    pg.wait_for_selector("[data-testid='stDataFrame']", timeout=60000)
    pg.wait_for_timeout(3000)
    pg.screenshot(path=out, full_page=False)
    b.close()
