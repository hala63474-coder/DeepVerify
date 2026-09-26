"""Benchmark all dataset samples against the running server."""
import os, json, time, sys, urllib.request, urllib.error

API = "http://127.0.0.1:5000/predict"
ROOT = r"d:\Hala-Pro\datasets"

def post_file(path):
    boundary = "----boundary_" + str(int(time.time() * 1000))
    with open(path, "rb") as f:
        data = f.read()
    name = os.path.basename(path)
    body  = (f"--{boundary}\r\n"
             f"Content-Disposition: form-data; name=\"video\"; filename=\"{name}\"\r\n"
             f"Content-Type: video/mp4\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(API, data=body, method="POST",
                                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())

def run(label, folder, expected):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))])
    rows = []
    correct = 0
    for p in files:
        try:
            r = post_file(p)
            verdict = r.get("result")
            conf    = r.get("confidence")
            unc     = r.get("uncertain")
            risk    = r.get("risk_level")
            veto    = r.get("forensic_veto")
            ms      = r.get("models", [])
            sig = {m["name"][:4]: m["score"] for m in ms}
            ok = (verdict == expected) and not unc
            correct += 1 if ok else 0
            rows.append((os.path.basename(p), verdict, conf, risk, unc, veto, sig))
            tag = "OK " if ok else ("UNC" if unc else "WRG")
            print(f"  [{tag}] {os.path.basename(p):40s} -> {verdict:5s} conf={conf} risk={risk} unc={unc} veto={veto} {sig}")
        except Exception as e:
            print(f"  [ERR] {os.path.basename(p)}: {e}")
            rows.append((os.path.basename(p), "ERR", None, None, None, None, {}))
    print(f"  >> {label}: {correct}/{len(files)} correct & confident")
    return rows, correct, len(files)

print("== REAL ==")
real_rows, rc, rt = run("REAL", os.path.join(ROOT, "real"), "Real")
print("\n== FAKE ==")
fake_rows, fc, ft = run("FAKE", os.path.join(ROOT, "fake"), "Fake")

print("\n== SUMMARY ==")
print(f"Real correct: {rc}/{rt}")
print(f"Fake correct: {fc}/{ft}")
print(f"Overall accuracy (correct & not-uncertain): {(rc+fc)}/{rt+ft} = {100*(rc+fc)/(rt+ft):.1f}%")
