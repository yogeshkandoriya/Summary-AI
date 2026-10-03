import subprocess
import time
import json
import urllib.request
import base64
import os
import websocket

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PROFILE_DIR = r"C:\BOT\chrome_debug_profile"

os.makedirs(PROFILE_DIR, exist_ok=True)

cmd = [
    CHROME_PATH,
    "--headless=new",
    "--remote-debugging-port=9222",
    f"--user-data-dir={PROFILE_DIR}",
    "--window-size=1280,850",
    "--remote-allow-origins=*",
    "--disable-gpu",
    "--no-sandbox",
    "http://localhost:8501"
]

print("Launching Chrome navigating directly to http://localhost:8501...")
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

try:
    print("Connecting to CDP...")
    req = urllib.request.urlopen("http://localhost:9222/json/list")
    tabs = json.loads(req.read().decode())
    print("Found tabs:", len(tabs))
    target_tab = None
    for tab in tabs:
        print("Tab:", tab.get("url"), tab.get("type"))
        if tab.get("type") == "page":
            target_tab = tab
            break
            
    if not target_tab:
        raise RuntimeError("No page tab found")

    ws_url = target_tab["webSocketDebuggerUrl"]
    print("Connecting websocket to:", ws_url)
    ws = websocket.create_connection(ws_url)

    print("Polling for Streamlit app to render...")
    for i in range(25):
        eval_msg = {
            "id": 100 + i,
            "method": "Runtime.evaluate",
            "params": {"expression": "document.body.innerText"}
        }
        ws.send(json.dumps(eval_msg))
        resp = json.loads(ws.recv())
        text = resp.get("result", {}).get("result", {}).get("value", "")
        print(f"[{i+1}s] Text length: {len(text)}, preview: {repr(text[:50])}")
        if "SUMMARY AI" in text or "Document Assistant" in text or "Upload" in text:
            print("App successfully rendered!")
            break
        time.sleep(1)

    time.sleep(2)
    # Capture screenshot
    msg = json.dumps({"id": 1, "method": "Page.captureScreenshot", "params": {"format": "png"}})
    ws.send(msg)
    
    # Handle potentially multiple incoming frames until we get our response id: 1
    while True:
        resp = json.loads(ws.recv())
        if resp.get("id") == 1:
            b64_img = resp["result"]["data"]
            break
            
    out_path = r"C:\BOT\screenshot_prototype_main.png"
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64_img))
    print(f"SUCCESS! Screenshot saved to {out_path} ({os.path.getsize(out_path)} bytes)")

    ws.close()

finally:
    print("Terminating Chrome...")
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except:
        proc.kill()
    print("Done!")
