<h1 align="center">🩺 PCOS Teşhis Destek Sistemi</h1>
<p align="center"><b>XGBoost tabanlı Polikistik Over Sendromu tespiti için klinik karar destek aracı</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-XGBoost-FF6600?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Dağıtım-HuggingFace%20Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Konteyner-Docker-2496ED?style=flat&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Lisans-MIT-green?style=flat" />
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/KubraParmak/pcos-diagnosis"><b>▶️ Canlı demoyu HuggingFace Spaces'te dene</b></a>
</p>

---

## 📌 Genel Bakış

**PCOS Teşhis Destek Sistemi**, **Polikistik Over Sendromu (PCOS)** tespitine yönelik makine öğrenmesi tabanlı bir klinik karar destek aracıdır. 38 hormonal kan değeri ve klinik ölçümü girdi olarak alıp PCOS risk tahmini ve sınıf olasılıkları döndürür.

Sistem, HuggingFace Spaces üzerinde **Docker** konteyneri içinde çalışan bir **FastAPI** uygulaması olarak dağıtılmıştır. Eğitilmiş model ve ölçekleyici, HuggingFace Model Hub'da ayrı bir repoda barındırılır ve çalışma zamanında yüklenir.

> ⚠️ Bu araç yalnızca **araştırma ve eğitim amaçlıdır**. **Tıbbi tanı yerine geçmez.**

---

## ✨ Özellikler

- 🔬 **38 özellikli klinik girdi** — hormonal, antropometrik ve yaşam tarzı verileri
- ⚡ **XGBoost sınıflandırıcı** — IQR tabanlı aykırı değer temizliği ve MinMaxScaler ön işleme
- 🌐 **FastAPI backend** — temiz, duyarlı HTML/CSS/JS frontend (ağır UI çerçevesi kullanılmadan)
- 🐳 **Docker dağıtımı** — HuggingFace Spaces'te tam ortam kontrolü
- 🗂️ **Ayrı model reposu** — model dosyaları HuggingFace Model Hub'da, çalışma zamanında yüklenir
- 📊 **Olasılık çıktısı** — hem PCOS hem Sağlıklı sınıfı olasılıklarını döndürür

---

## 🧠 Model

| Özellik | Detay |
|---|---|
| Algoritma | XGBoost (`XGBClassifier`) |
| Girdi özellikleri | 38 klinik ve hormonal ölçüm |
| Aykırı değer temizliği | IQR yöntemi — LH, FSH, FSH/LH sütunlarında |
| Çıkarılan sütunlar | `Sl. No`, `Patient File No.`, `Hip(inch)`, `BMI`, `Avg. F size (R)` |
| Eksik değerler | Eğitim seti medyanıyla dolduruldu (veri sızıntısı olmadan) |
| Ölçekleme | MinMaxScaler (yalnızca eğitim setine fit edildi) |
| Eğitim / Test ayrımı | %80 / %20, stratified |
| n_estimators | 300 |
| learning_rate | 0.05 |
| max_depth | 5 |
| scale_pos_weight | 2 (sınıf dengesizliği düzeltmesi) |
| random_state | 42 |

### Girdi Özellikleri (38 adet)

| Kategori | Özellikler |
|---|---|
| Antropometrik | Yaş, Kilo, Boy, Bel, Bel/Kalça oranı |
| Vital bulgular | Nabız, Solunum hızı, Sistolik KB, Diastolik KB |
| Kan değerleri | Hemoglobin, FSH, LH, FSH/LH, beta-HCG (I. ve II.), TSH, AMH, Prolaktin, D3 Vitamini, Progesteron, Açlık kan şekeri |
| Üreme sağlığı | Adet düzeni ve süresi, Hamilelik durumu, Düşük sayısı, Folikül sayısı (sol/sağ), Ort. folikül boyutu (sol), Endometrium kalınlığı |
| Yaşam tarzı | Kilo artışı, Kıllanma, Cilt koyulaşması, Saç dökülmesi, Sivilce, Fast food tüketimi, Düzenli egzersiz |

---

## 🏗️ Mimari

```
HuggingFace Model Hub                HuggingFace Spaces (Docker)
─────────────────────                ──────────────────────────────
xgboost_model.pkl   ──┐              ┌─────────────────────────────┐
scaler.pkl          ──┼─── yükle ──▶ │  FastAPI app (app.py)       │
feature_names.json  ──┘              │  • GET  /   → HTML arayüzü  │
                                     │  • POST /predict → JSON      │
                                     └─────────────────────────────┘
                                                  ▲
                                           Kullanıcı tarayıcısı
```

---

## 📁 Depo Yapısı

```
pcos-diagnosis/
├── app.py                # FastAPI uygulaması + HTML frontend
├── requirements.txt      # Python bağımlılıkları
├── Dockerfile            # Docker derleme yapılandırması
├── README.md
└── notebooks/
    └── pcos_diagnosis_table_data.ipynb   # Veri analizi ve model eğitimi
```

### HuggingFace Model Reposu — `KubraParmak/pcos-xgboost-model`

```
pcos-xgboost-model/
├── xgboost_model.pkl     # Eğitilmiş XGBoost sınıflandırıcı (joblib)
├── scaler.pkl            # Fit edilmiş MinMaxScaler (joblib)
├── feature_names.json    # Girdi özellik isimlerinin sıralı listesi
└── README.md             # Model kartı
```

---

## 🚀 Kurulum

### Yerel Çalıştırma

**Gereksinimler:** Python 3.10+, Git

```bash
# 1. Repoyu klonla
git clone https://github.com/KbrPrmk/pcos-diagnosis.git
cd pcos-diagnosis

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Sunucuyu başlat
python app.py
# → http://localhost:7860 adresini aç
```

### Docker ile Çalıştırma

```bash
# İmajı derle
docker build -t pcos-diagnosis .

# Konteyneri çalıştır
docker run -p 7860:7860 pcos-diagnosis
# → http://localhost:7860 adresini aç
```

---

## 🌐 API Referansı

### `GET /`
HTML web arayüzünü döndürür.

### `POST /predict`
38 özellik değerini içeren JSON nesnesi alır, tahmin sonuçlarını döndürür.

**İstek gövdesi:**
```json
{
  " Age (yrs)": 25,
  "FSH(mIU/mL)": 7.2,
  "LH(mIU/mL)": 6.1,
  "AMH(ng/mL)": 3.5,
  "Follicle No. (L)": 8,
  "Follicle No. (R)": 9,
  ...
}
```

**Yanıt:**
```json
{
  "pcos": true,
  "label": "🔴 PCOS Tespit Edildi",
  "detail": "PCOS olasılığı: 78.3% | Sağlıklı olasılığı: 21.7%"
}
```

---

## 📊 Veri Seti

- **Kaynak:** [PCOS Dataset — Kaggle](https://www.kaggle.com/datasets/ayamoheddine/pcos-dataset)
- **Hasta sayısı:** 541
- **Orijinal özellik sayısı:** 42
- **Kullanılan özellik sayısı:** 38 (ID sütunları ve yüksek korelasyonlu özellikler çıkarıldı)
- **Hedef:** İkili — PCOS (1) / Sağlıklı (0)
- **Sınıf dağılımı:** ~%33 PCOS, ~%67 Sağlıklı

---

## 🗺️ Yol Haritası

### ✅ v1.0 — Tablo Verisi Modeli (mevcut)
- 38 klinik özellik üzerinde XGBoost sınıflandırıcı
- FastAPI + HTML frontend
- HuggingFace Spaces'te Docker dağıtımı

### 🔲 v2.0 — Çok Modlu Sistem
- [ ] Ultrason görüntü sınıflandırması için **EfficientNetB0** entegrasyonu (128×128, özel sınıflandırma katmanı)
- [ ] Tablo verisi + görüntü birleşik çıkarım hattı
- [ ] Her iki modaliteden birleşik güven skoru
- [ ] Ayrı görüntü yükleme UI paneli

### 🔲 v3.0 — Klinik Düzey
- [ ] LIME / SHAP açıklanabilirlik paneli (tahmin başına özellik katkısı)
- [ ] Hasta oturumu geçmiş takibi
- [ ] PDF rapor oluşturma
- [ ] Token tabanlı kimlik doğrulamalı REST API
- [ ] Çok dilli arayüz (TR / EN)

---

## 🛠️ Teknoloji Yığını

<p>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Uvicorn-4B0082?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black" />
</p>

---

## 📄 Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır — detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 🙋 Geliştirici

**Kübra Parmak**
- GitHub: [@KbrPrmk](https://github.com/KbrPrmk)
- HuggingFace: [@KubraParmak](https://huggingface.co/KubraParmak)
