"""AgroSaarthi local crop recommendation demo; uses only Python's standard library."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from train_model import MODEL, FEATURES, predict

ROOT = Path(__file__).resolve().parent
model = json.loads(MODEL.read_text(encoding="utf-8"))

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgroSaarthi · Crop advisor</title>
<style>
:root{--forest:#164d36;--leaf:#2f7955;--lime:#dceccf;--ink:#1e3026;--muted:#68766d;--paper:#f6f8f3;--line:#e0e7dd;--white:#fff;--gold:#a87831}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}.shell{max-width:1110px;margin:auto;padding:0 24px}.top{background:var(--forest);color:white}.topin{height:78px;display:flex;align-items:center;justify-content:space-between}.brand{display:flex;align-items:center;gap:11px;font-weight:750;font-size:18px;letter-spacing:-.3px}.logo{width:39px;height:39px;border-radius:13px;background:#e5f1db;color:var(--forest);display:grid;place-items:center;font-size:21px}.tag{font-size:12px;color:#c7d9cb}.hero{padding:48px 0 34px}.eyebrow{color:var(--leaf);text-transform:uppercase;letter-spacing:1.7px;font-size:11px;font-weight:800}.hero h1{font-size:clamp(30px,4vw,43px);line-height:1.12;letter-spacing:-1.2px;margin:10px 0 12px;color:#183a2a}.hero p{color:var(--muted);max-width:650px;margin:0}.layout{display:grid;grid-template-columns:minmax(0,1.08fr) minmax(320px,.92fr);gap:22px;align-items:start;padding-bottom:50px}.card{background:white;border:1px solid var(--line);border-radius:17px;box-shadow:0 5px 20px #243e2910}.formcard{padding:25px}.cardhead{display:flex;justify-content:space-between;gap:15px;align-items:flex-start;margin-bottom:20px}.card h2{margin:0;font-size:18px;letter-spacing:-.3px}.sub{font-size:13px;color:var(--muted);margin:4px 0 0}.badge{font-size:11px;font-weight:700;color:#39704f;background:#eef5e9;border-radius:100px;padding:6px 10px;white-space:nowrap}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}.field label{display:flex;justify-content:space-between;font-size:12px;font-weight:700;color:#344b3a;margin-bottom:6px}.unit{color:#87928a;font-weight:500}.field input{width:100%;height:45px;border:1px solid #dce4db;border-radius:9px;padding:0 12px;color:var(--ink);font:inherit;background:#fff;outline:none;transition:.15s}.field input:focus{border-color:#4e9468;box-shadow:0 0 0 3px #4e94681c}.hint{font-size:11px;color:#89948b;margin-top:5px}.actions{display:flex;gap:11px;align-items:center;margin-top:22px}.primary,.secondary{height:45px;border:0;border-radius:9px;padding:0 17px;font:700 13px inherit;cursor:pointer}.primary{background:var(--forest);color:#fff;flex:1}.primary:hover{background:#226344}.secondary{background:#f0f4ed;color:#435c49}.primary:disabled{opacity:.65;cursor:wait}.result{padding:25px;min-height:360px}.empty{min-height:290px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:var(--muted)}.emptyicon{width:64px;height:64px;border-radius:20px;background:#eff5e9;display:grid;place-items:center;font-size:29px;margin-bottom:13px}.empty strong{color:#3d5143;font-size:15px}.empty span{max-width:260px;font-size:13px;margin-top:6px}.resulttop{display:flex;justify-content:space-between;align-items:center}.resulttop h2{font-size:17px}.cropbox{margin-top:17px;padding:19px;background:linear-gradient(130deg,#164d36,#2b704c);color:#fff;border-radius:13px;position:relative;overflow:hidden}.cropbox:after{content:'🌱';font-size:70px;position:absolute;right:18px;top:10px;opacity:.19}.cropbox .tiny{text-transform:uppercase;letter-spacing:1.3px;font-size:10px;color:#d2e2d5;font-weight:700}.cropbox h3{font-size:27px;margin:5px 0 2px;text-transform:capitalize}.cropbox p{font-size:12px;color:#d6e7da;margin:0}.confidence{margin-top:18px}.linehead{display:flex;justify-content:space-between;font-size:12px;color:#506256}.track{height:7px;background:#edf1eb;border-radius:8px;margin-top:7px;overflow:hidden}.bar{height:100%;border-radius:8px;background:#71a978}.altlabel{font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:750;color:#738077;margin:20px 0 8px}.alt{display:flex;justify-content:space-between;padding:9px 11px;border:1px solid var(--line);border-radius:8px;margin-top:7px;font-size:13px}.alt b{text-transform:capitalize}.note{font-size:11px;color:#7d8980;line-height:1.45;margin:16px 0 0}.error{display:none;background:#fff2ef;color:#a33d2e;padding:10px 12px;border-radius:8px;font-size:12px;margin-top:14px}.foot{padding:15px 0 25px;color:#818d83;font-size:11px;border-top:1px solid var(--line)}.footin{display:flex;justify-content:space-between;gap:12px}.hidden{display:none!important}@media(max-width:760px){.layout{grid-template-columns:1fr}.result{min-height:0}.empty{min-height:235px}.hero{padding-top:36px}.tag{display:none}}@media(max-width:420px){.shell{padding:0 15px}.formcard,.result{padding:19px}.grid{gap:12px}.field label{font-size:11px}.hero h1{font-size:31px}}
</style></head><body>
<header class="top"><div class="shell topin"><div class="brand"><span class="logo">🌾</span>AgroSaarthi</div><div class="tag">A simple crop recommendation assistant</div></div></header>
<main class="shell"><section class="hero"><div class="eyebrow">Your field, your next step</div><h1>Find a crop suited to your soil.</h1><p>Enter your soil nutrients and local growing conditions to get a crop suggestion from our trained recommendation model.</p></section>
<section class="layout"><div class="card formcard"><div class="cardhead"><div><h2>Field conditions</h2><p class="sub">Use your soil test and local climate readings.</p></div><span class="badge">7 values</span></div><form id="form"><div class="grid">
<div class="field"><label for="N">Nitrogen <span class="unit">N · kg/ha</span></label><input id="N" name="N" type="number" min="0" max="200" step="any" placeholder="e.g. 90" required><div class="hint">Soil nitrogen content</div></div>
<div class="field"><label for="P">Phosphorus <span class="unit">P · kg/ha</span></label><input id="P" name="P" type="number" min="0" max="200" step="any" placeholder="e.g. 42" required><div class="hint">Soil phosphorus content</div></div>
<div class="field"><label for="K">Potassium <span class="unit">K · kg/ha</span></label><input id="K" name="K" type="number" min="0" max="250" step="any" placeholder="e.g. 43" required><div class="hint">Soil potassium content</div></div>
<div class="field"><label for="temperature">Temperature <span class="unit">°C</span></label><input id="temperature" name="temperature" type="number" min="0" max="55" step="any" placeholder="e.g. 20.8" required><div class="hint">Average growing temperature</div></div>
<div class="field"><label for="humidity">Humidity <span class="unit">%</span></label><input id="humidity" name="humidity" type="number" min="0" max="100" step="any" placeholder="e.g. 82" required><div class="hint">Relative humidity</div></div>
<div class="field"><label for="ph">Soil pH <span class="unit">0–14</span></label><input id="ph" name="ph" type="number" min="0" max="14" step="any" placeholder="e.g. 6.5" required><div class="hint">Soil acidity or alkalinity</div></div>
<div class="field"><label for="rainfall">Rainfall <span class="unit">mm</span></label><input id="rainfall" name="rainfall" type="number" min="0" max="1000" step="any" placeholder="e.g. 203" required><div class="hint">Expected seasonal rainfall</div></div></div>
<div id="error" class="error"></div><div class="actions"><button class="primary" id="submit" type="submit">Get crop recommendation <span aria-hidden="true">→</span></button><button class="secondary" id="example" type="button">Try example</button></div></form></div>
<div class="card result"><div id="empty" class="empty"><div class="emptyicon">🌿</div><strong>Your recommendation will appear here</strong><span>Fill in the seven field values to see the best matching crop and other model suggestions.</span></div><div id="answer" class="hidden"><div class="resulttop"><h2>Crop recommendation</h2><span class="badge">Model result</span></div><div class="cropbox"><div class="tiny">Best match for these conditions</div><h3 id="crop">—</h3><p>Suggested from soil and climate inputs</p></div><div class="confidence"><div class="linehead"><span>Model vote share</span><b id="conf">—</b></div><div class="track"><div class="bar" id="bar" style="width:0%"></div></div></div><div class="altlabel">Other matching crops</div><div id="alternatives"></div><p class="note">This is a pattern-based suggestion from the training dataset, not a guarantee of yield, profitability, or local suitability. Verify with a local agriculture expert before planting.</p></div></div></section>
<footer class="foot"><div class="footin"><span>AgroSaarthi · Crop advisory prototype</span><span>Based on the Crop Recommendation Dataset</span></div></footer></main>
<script>
const form=document.querySelector('#form'), errorBox=document.querySelector('#error'), submit=document.querySelector('#submit');
document.querySelector('#example').addEventListener('click',()=>{Object.entries({N:90,P:42,K:43,temperature:20.8,humidity:82,ph:6.5,rainfall:202.9}).forEach(([k,v])=>document.querySelector(`[name=${k}]`).value=v); form.requestSubmit()});
form.addEventListener('submit',async e=>{e.preventDefault();errorBox.style.display='none';submit.disabled=true;submit.textContent='Finding a match…';try{const values=Object.fromEntries(new FormData(form));const res=await fetch('/api/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});const data=await res.json();if(!res.ok)throw Error(data.error||'Could not create recommendation.');document.querySelector('#empty').classList.add('hidden');document.querySelector('#answer').classList.remove('hidden');document.querySelector('#crop').textContent=data.recommendations[0].crop;const pct=Math.round(data.recommendations[0].vote_share*100);document.querySelector('#conf').textContent=pct+'%';document.querySelector('#bar').style.width=pct+'%';document.querySelector('#alternatives').innerHTML=data.recommendations.slice(1).map(x=>`<div class="alt"><b>${x.crop}</b><span>${Math.round(x.vote_share*100)}% vote share</span></div>`).join('')}catch(err){errorBox.textContent=err.message;errorBox.style.display='block'}finally{submit.disabled=false;submit.innerHTML='Get crop recommendation <span aria-hidden="true">→</span>'}});
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if urlparse(self.path).path == "/api/health":
            return self.send_json(200, {"status": "ok", "model": model["model"],
                "dataset_rows": model["dataset_rows"], "holdout_accuracy": model["holdout_accuracy"]})
        page = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
        if urlparse(self.path).path != "/api/recommend":
            return self.send_json(404, {"error": "Not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            values = [float(body[key]) for key in FEATURES]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Please enter valid numeric values.")
            # Use dataset feature bounds to avoid giving confident extrapolations.
            mins = [min(point[i] * model["stds"][i] + model["means"][i] for point in model["points"]) for i in range(7)]
            maxs = [max(point[i] * model["stds"][i] + model["means"][i] for point in model["points"]) for i in range(7)]
            outside = [FEATURES[i] for i, v in enumerate(values) if v < mins[i] or v > maxs[i]]
            recommendations = predict(model, values)
            return self.send_json(200, {"recommendations": [{"crop": name, "vote_share": score}
                for name, score in recommendations], "outside_training_range": outside})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return self.send_json(400, {"error": f"Please check all seven values. {exc}"})

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    host, port = "127.0.0.1", 8000
    print(f"AgroSaarthi is running at http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
