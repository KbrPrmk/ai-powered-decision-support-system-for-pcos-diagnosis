from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import joblib
import json
from huggingface_hub import hf_hub_download
import uvicorn

REPO_ID = "KubraParmak/pcos-xgboost-model"

model  = joblib.load(hf_hub_download(REPO_ID, "xgboost_model.pkl"))
scaler = joblib.load(hf_hub_download(REPO_ID, "scaler.pkl"))
with open(hf_hub_download(REPO_ID, "feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)

FEATURE_META = {
    " Age (yrs)":              ("Yaş (yıl)",             10,  60,  25),
    "Weight (Kg)":             ("Kilo (kg)",              30, 150,  65),
    "Height(Cm) ":             ("Boy (cm)",              140, 195, 162),
    "Pulse rate(bpm) ":        ("Nabız (bpm)",            50, 120,  75),
    "RR (breaths/min)":        ("Solunum hızı (/dk)",     10,  25,  16),
    "Hb(g/dl)":                ("Hemoglobin (g/dL)",       8,  18,  13),
    "Cycle(R/I)":              ("Adet düzeni R=1 I=2",     1,   2,   1),
    "Cycle length(days)":      ("Adet süresi (gün)",       2,  10,   5),
    "Marraige Status (Yrs)":   ("Evlilik süresi (yıl)",    0,  20,   3),
    "Pregnant(Y/N)":           ("Hamile (0/1)",            0,   1,   0),
    "No. of aborptions":       ("Düşük sayısı",            0,   5,   0),
    "I   beta-HCG(mIU/mL)":   ("I. beta-HCG (mIU/mL)",   0, 100,   1),
    "II    beta-HCG(mIU/mL)":  ("II. beta-HCG (mIU/mL)",  0, 100,   1),
    "FSH(mIU/mL)":             ("FSH (mIU/mL)",            1,  25,   7),
    "LH(mIU/mL)":              ("LH (mIU/mL)",             1,  60,   6),
    "FSH/LH":                  ("FSH/LH oranı",            0,  10, 1.2),
    "Hip(inch)":               ("Kalça (inch)",            28,  55,  38),
    "Waist(inch)":             ("Bel (inch)",              22,  50,  30),
    "Waist:Hip Ratio":         ("Bel/Kalça oranı",        0.5, 1.2, 0.8),
    "TSH (mIU/L)":             ("TSH (mIU/L)",             0,  10, 2.5),
    "AMH(ng/mL)":              ("AMH (ng/mL)",             0,  15, 3.5),
    "PRL(ng/mL)":              ("Prolaktin (ng/mL)",       2,  40,  15),
    "Vit D3 (ng/mL)":          ("Vitamin D3 (ng/mL)",      5,  80,  25),
    "PRG(ng/mL)":              ("Progesteron (ng/mL)",     0,  25,   2),
    "RBS(mg/dl)":              ("Açlık kan şekeri (mg/dL)",60, 300,  90),
    "Weight gain(Y/N)":        ("Kilo artışı (0/1)",       0,   1,   0),
    "hair growth(Y/N)":        ("Kıllanma artışı (0/1)",   0,   1,   0),
    "Skin darkening (Y/N)":    ("Cilt koyulaşması (0/1)",  0,   1,   0),
    "Hair loss(Y/N)":          ("Saç dökülmesi (0/1)",     0,   1,   0),
    "Pimples(Y/N)":            ("Sivilce (0/1)",           0,   1,   0),
    "Fast food (Y/N)":         ("Fast food (0/1)",         0,   1,   0),
    "Reg.Exercise(Y/N)":       ("Düzenli egzersiz (0/1)",  0,   1,   0),
    "BP _Systolic (mmHg)":     ("Sistolik KB (mmHg)",     70, 180, 110),
    "BP _Diastolic (mmHg)":    ("Diastolik KB (mmHg)",    40, 120,  70),
    "Follicle No. (L)":        ("Folikül sayısı Sol",      0,  30,   8),
    "Follicle No. (R)":        ("Folikül sayısı Sağ",      0,  30,   8),
    "Avg. F size (L) (mm)":    ("Ort. folikül boyutu Sol (mm)", 0, 20, 9),
    "Endometrium (mm)":        ("Endometrium kalınlığı (mm)", 2, 18,  8),
}

app = FastAPI()

# HTML arayüzü
def build_html():
    fields_html = ""
    for feat in FEATURE_NAMES:
        meta = FEATURE_META.get(feat, (feat, 0, 999, 0))
        label, mn, mx, default = meta
        fields_html += f"""
        <div class="field">
          <label>{label}</label>
          <input type="number" name="{feat}" value="{default}" min="{mn}" max="{mx}" step="any">
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PCOS Teşhis Destek Sistemi</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #333; }}
  header {{ background: linear-gradient(135deg, #c850c0, #4158d0); color: white; padding: 24px; text-align: center; }}
  header h1 {{ font-size: 1.6rem; }}
  header p {{ margin-top: 6px; opacity: 0.85; font-size: 0.9rem; }}
  .container {{ max-width: 900px; margin: 30px auto; padding: 0 16px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }}
  .field {{ background: white; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .field label {{ display: block; font-size: 0.78rem; color: #666; margin-bottom: 6px; }}
  .field input {{ width: 100%; border: 1px solid #ddd; border-radius: 6px; padding: 8px 10px; font-size: 1rem; }}
  .field input:focus {{ outline: none; border-color: #c850c0; }}
  .btn {{ display: block; width: 100%; margin: 24px 0; padding: 14px; background: linear-gradient(135deg, #c850c0, #4158d0); color: white; border: none; border-radius: 10px; font-size: 1.1rem; cursor: pointer; }}
  .btn:hover {{ opacity: 0.9; }}
  #result {{ display: none; background: white; border-radius: 12px; padding: 24px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 30px; }}
  #result .label {{ font-size: 1.5rem; font-weight: bold; margin-bottom: 10px; }}
  #result .prob {{ font-size: 0.95rem; color: #555; }}
  .warn {{ background: #fff8e1; border-left: 4px solid #f9a825; padding: 12px 16px; border-radius: 6px; font-size: 0.82rem; color: #555; margin-bottom: 20px; }}
</style>
</head>
<body>
<header>
  <h1>🩺 PCOS Teşhis Destek Sistemi</h1>
  <p>XGBoost tabanlı karar destek aracı</p>
</header>
<div class="container">
  <br>
  <div class="warn">⚠️ Bu araç yalnızca araştırma amaçlıdır. Tıbbi tanı yerine geçmez.</div>
  <div class="grid" id="form">{fields_html}</div>
  <button class="btn" onclick="predict()">Tahmin Et</button>
  <div id="result">
    <div class="label" id="res-label"></div>
    <div class="prob" id="res-prob"></div>
  </div>
</div>
<script>
async function predict() {{
  const inputs = document.querySelectorAll('#form input');
  const data = {{}};
  inputs.forEach(i => data[i.name] = parseFloat(i.value) || 0);
  const res = await fetch('/predict', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(data)}});
  const json = await res.json();
  document.getElementById('result').style.display = 'block';
  document.getElementById('res-label').innerHTML = json.label;
  document.getElementById('res-label').style.color = json.pcos ? '#e53935' : '#43a047';
  document.getElementById('res-prob').innerHTML = json.detail;
  document.getElementById('result').scrollIntoView({{behavior:'smooth'}});
}}
</script>
</body>
</html>"""

class InputData(BaseModel):
    __annotations__ = {f: float for f in [
        " Age (yrs)", "Weight (Kg)", "Height(Cm) ", "Pulse rate(bpm) ",
        "RR (breaths/min)", "Hb(g/dl)", "Cycle(R/I)", "Cycle length(days)",
        "Marraige Status (Yrs)", "Pregnant(Y/N)", "No. of aborptions",
        "I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)", "FSH(mIU/mL)",
        "LH(mIU/mL)", "FSH/LH", "Hip(inch)", "Waist(inch)", "Waist:Hip Ratio",
        "TSH (mIU/L)", "AMH(ng/mL)", "PRL(ng/mL)", "Vit D3 (ng/mL)",
        "PRG(ng/mL)", "RBS(mg/dl)", "Weight gain(Y/N)", "hair growth(Y/N)",
        "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)",
        "Fast food (Y/N)", "Reg.Exercise(Y/N)", "BP _Systolic (mmHg)",
        "BP _Diastolic (mmHg)", "Follicle No. (L)", "Follicle No. (R)",
        "Avg. F size (L) (mm)", "Endometrium (mm)"
    ]}

@app.get("/", response_class=HTMLResponse)
def index():
    return build_html()

@app.post("/predict")
def predict(data: dict):
    values = [data.get(f, 0) for f in FEATURE_NAMES]
    arr = np.array(values, dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prob = model.predict_proba(arr_scaled)[0]
    pcos = bool(prob[1] >= 0.5)
    return {
        "pcos": pcos,
        "label": "🔴 PCOS Tespit Edildi" if pcos else "🟢 PCOS Tespit Edilmedi",
        "detail": f"PCOS olasılığı: {prob[1]*100:.1f}% | Sağlıklı olasılığı: {prob[0]*100:.1f}%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)