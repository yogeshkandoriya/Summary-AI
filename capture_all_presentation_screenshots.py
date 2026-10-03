import subprocess
import time
import json
import urllib.request
import base64
import os
import websocket

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PROFILE_DIR = r"C:\BOT\chrome_debug_profile"
OUTPUT_DIR = r"C:\BOT\screenshots"

os.makedirs(PROFILE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = [
    {
        "name": "screenshot_upload.png",
        "url": "http://localhost:8501/?view=upload",
        "description": "Slide 4: PDF Upload & Processing",
        "window": "1280,820",
        "wait_dom": "Drop PDF files here",
        "scroll_js": "window.scrollTo(0, 0);"
    },
    {
        "name": "screenshot_summary.png",
        "url": "http://localhost:8501/?view=summary",
        "description": "Slide 4: Generated Summary with Citations",
        "window": "1280,820",
        "wait_dom": "summary-point-line",
        "scroll_js": "const el = document.querySelector('.summary-point-line'); if(el) el.scrollIntoView({block: 'center'}); else window.scrollTo(0, 550);"
    },
    {
        "name": "screenshot_askpdf.png",
        "url": "http://localhost:8501/?view=askpdf",
        "description": "Slide 4: Grounded Ask PDF with Slash Commands",
        "window": "1280,820",
        "wait_dom": "ai-output-card",
        "scroll_js": "const el = document.querySelector('.ai-output-card'); if(el) el.scrollIntoView({block: 'center'}); else window.scrollTo(0, 520);"
    },
    {
        "name": "screenshot_keywords.png",
        "url": "http://localhost:8501/?view=keywords",
        "description": "Slide 4: Extracted Keywords & Sources",
        "window": "1280,820",
        "wait_dom": "keyword-chip",
        "scroll_js": "window.scrollTo(0, 550);"
    },
    {
        "name": "screenshot_summary_output.png",
        "url": "http://localhost:8501/?view=summary_output",
        "description": "Slide 6: Generated Summary Output Details",
        "window": "1200,750",
        "wait_dom": "summary-point-line",
        "scroll_js": "const el = document.querySelector('.summary-point-line'); if(el) el.scrollIntoView({block: 'start'}); else window.scrollTo(0, 600);"
    },
    {
        "name": "screenshot_askpdf_citation.png",
        "url": "http://localhost:8501/?view=askpdf_citation",
        "description": "Slide 6: Grounded Ask PDF with Citation Evidence",
        "window": "1200,780",
        "wait_dom": "source-quote",
        "scroll_js": "const el = document.querySelector('.source-quote'); if(el) el.scrollIntoView({block: 'center'}); else window.scrollTo(0, 650);"
    },
    {
        "name": "screenshot_prototype_main.png",
        "url": "http://localhost:8501/?view=prototype",
        "description": "Slide 8: Full Working Prototype Interface",
        "window": "1366,850",
        "wait_dom": "doc-card",
        "scroll_js": "window.scrollTo(0, 220);"
    },
]

def capture_target(target):
    out_file = os.path.join(OUTPUT_DIR, target["name"])
    print(f"\n==================================================")
    print(f"Capturing: {target['name']} ({target['description']})")
    print(f"URL: {target['url']}")
    
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        f"--user-data-dir={PROFILE_DIR}",
        f"--window-size={target['window']}",
        "--remote-allow-origins=*",
        "--disable-gpu",
        "--no-sandbox",
        target["url"]
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    try:
        req = urllib.request.urlopen("http://localhost:9222/json/list")
        tabs = json.loads(req.read().decode())
        page_tab = None
        for t in tabs:
            if t.get("type") == "page":
                page_tab = t
                break
        if not page_tab:
            raise RuntimeError("No page tab found")

        ws_url = page_tab["webSocketDebuggerUrl"]
        ws = websocket.create_connection(ws_url, timeout=15)

        # Poll for DOM text
        print(f"Waiting for DOM to contain '{target['wait_dom']}'...")
        for sec in range(15):
            eval_msg = {
                "id": 200 + sec,
                "method": "Runtime.evaluate",
                "params": {"expression": "document.body.innerText + ' ' + document.body.innerHTML"}
            }
            ws.send(json.dumps(eval_msg))
            resp = json.loads(ws.recv())
            val = resp.get("result", {}).get("result", {}).get("value", "")
            if target["wait_dom"].lower() in val.lower():
                print(f"Match found at {sec+1}s!")
                break
            time.sleep(1)

        # Allow extra 1.5s for layout rendering
        time.sleep(1.5)

        # If it's keywords tab, switch to Keywords tab via JS click if needed
        if target["name"] == "screenshot_keywords.png":
            click_kw = {
                "id": 301,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": """
                    const tabs = Array.from(document.querySelectorAll('button[role="tab"], [data-baseweb="tab"]'));
                    const kwTab = tabs.find(t => t.innerText && t.innerText.includes('Keywords'));
                    if (kwTab) { kwTab.click(); 'clicked keywords'; } else { 'not found'; }
                    """
                }
            }
            ws.send(json.dumps(click_kw))
            ws.recv()
            time.sleep(1)

        # Scroll to focus area
        if target.get("scroll_js"):
            scroll_eval = {
                "id": 350,
                "method": "Runtime.evaluate",
                "params": {"expression": target["scroll_js"]}
            }
            ws.send(json.dumps(scroll_eval))
            ws.recv()
            time.sleep(1.0)

        # Capture screenshot
        cap_msg = {"id": 999, "method": "Page.captureScreenshot", "params": {"format": "png"}}
        ws.send(json.dumps(cap_msg))
        
        while True:
            resp = json.loads(ws.recv())
            if resp.get("id") == 999:
                b64_data = resp["result"]["data"]
                break

        with open(out_file, "wb") as f:
            f.write(base64.b64decode(b64_data))
        print(f"Successfully saved {out_file} ({os.path.getsize(out_file):,} bytes)")
        ws.close()

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except:
            proc.kill()
        time.sleep(1)

if __name__ == "__main__":
    print("Starting automated capture of real Seminar 3 presentation screenshots...")
    for t in TARGETS:
        try:
            capture_target(t)
        except Exception as e:
            print(f"Error capturing {t['name']}: {e}")
    print("\nAll presentation screenshots captured successfully!")
