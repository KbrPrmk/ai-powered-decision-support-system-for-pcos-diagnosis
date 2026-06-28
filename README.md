<h1 align="center">🩺 PCOS Diagnosis Support System</h1>
<p align="center"><b>XGBoost-powered clinical decision support for Polycystic Ovary Syndrome detection</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-XGBoost-FF6600?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Deployment-HuggingFace%20Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Container-Docker-2496ED?style=flat&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" />
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/KubraParmak/pcos-diagnosis"><b>▶️ Try the live demo on HuggingFace Spaces</b></a>
</p>

---

## 📌 Overview

**PCOS Diagnosis Support System** is a machine learning–powered clinical decision support tool for **Polycystic Ovary Syndrome (PCOS)** detection. It takes 38 hormonal blood values and clinical measurements as input and returns a PCOS risk prediction along with class probabilities.

The system is deployed as a **FastAPI** web application inside a **Docker** container on HuggingFace Spaces, with the trained model and scaler hosted separately on the HuggingFace Model Hub.

> ⚠️ This tool is intended for **research and educational purposes only**. It is **not a substitute for medical diagnosis**.

---

## ✨ Features

- 🔬 **38-feature clinical input** — covers hormonal, anthropometric, and lifestyle data
- ⚡ **XGBoost classifier** with IQR-based outlier removal and MinMaxScaler preprocessing
- 🌐 **FastAPI backend** with a clean, responsive HTML/CSS/JS frontend — no heavy UI frameworks
- 🐳 **Docker deployment** on HuggingFace Spaces for full environment control
- 🗂️ **Separate model repo** — model artifacts hosted on HuggingFace Model Hub, loaded at runtime
- 📊 **Probability output** — returns both PCOS and Healthy class probabilities

---

## 🧠 Model

| Property | Detail |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`) |
| Input features | 38 clinical & hormonal measurements |
| Outlier removal | IQR method on LH, FSH, FSH/LH columns |
| Dropped columns | `Sl. No`, `Patient File No.`, `Hip(inch)`, `BMI`, `Avg. F size (R)` |
| Missing values | Filled with train-set median (leak-free) |
| Scaling | MinMaxScaler (fit on train only) |
| Train / Test split | 80% / 20%, stratified |
| n_estimators | 300 |
| learning_rate | 0.05 |
| max_depth | 5 |
| scale_pos_weight | 2 (class imbalance correction) |
| random_state | 42 |

### Input Features (38)

| Category | Features |
|---|---|
| Anthropometric | Age, Weight, Height, Waist, Waist:Hip Ratio |
| Vital signs | Pulse rate, Respiratory rate, Systolic BP, Diastolic BP |
| Blood values | Hemoglobin, FSH, LH, FSH/LH, beta-HCG (I & II), TSH, AMH, Prolactin, Vitamin D3, Progesterone, Fasting blood glucose |
| Reproductive | Menstrual cycle regularity & length, Pregnancy status, Number of abortions, Follicle count (L/R), Average follicle size (L), Endometrium thickness |
| Lifestyle | Weight gain, Hair growth, Skin darkening, Hair loss, Acne, Fast food consumption, Regular exercise |

---

## 🏗️ Architecture

```
HuggingFace Model Hub                HuggingFace Spaces (Docker)
─────────────────────                ──────────────────────────────
xgboost_model.pkl   ──┐              ┌─────────────────────────────┐
scaler.pkl          ──┼─── load ───▶ │  FastAPI app (app.py)       │
feature_names.json  ──┘              │  • GET  /   → HTML UI       │
                                     │  • POST /predict → JSON      │
                                     └─────────────────────────────┘
                                                  ▲
                                           User browser
```

---

## 📁 Repository Structure

```
pcos-diagnosis/
├── app.py                # FastAPI application + HTML frontend
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker build configuration
├── README.md
└── notebooks/
    └── pcos_diagnosis_table_data.ipynb   # Data analysis & model training
```

### HuggingFace Model Repo — `KubraParmak/pcos-xgboost-model`

```
pcos-xgboost-model/
├── xgboost_model.pkl     # Trained XGBoost classifier (joblib)
├── scaler.pkl            # Fitted MinMaxScaler (joblib)
├── feature_names.json    # Ordered list of input feature names
└── README.md             # Model card
```

---

## 🚀 Getting Started

### Run Locally

**Prerequisites:** Python 3.10+, Git

```bash
# 1. Clone the repository
git clone https://github.com/KbrPrmk/pcos-diagnosis.git
cd pcos-diagnosis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python app.py
# → Open http://localhost:7860
```

### Run with Docker

```bash
# Build the image
docker build -t pcos-diagnosis .

# Run the container
docker run -p 7860:7860 pcos-diagnosis
# → Open http://localhost:7860
```

---

## 🌐 API Reference

### `GET /`
Returns the HTML web interface.

### `POST /predict`
Accepts a JSON object with all 38 feature values, returns prediction results.

**Request body:**
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

**Response:**
```json
{
  "pcos": true,
  "label": "🔴 PCOS Tespit Edildi",
  "detail": "PCOS olasılığı: 78.3% | Sağlıklı olasılığı: 21.7%"
}
```

---

## 📊 Dataset

- **Source:** [PCOS Dataset — Kaggle](https://www.kaggle.com/datasets/ayamoheddine/pcos-dataset)
- **Patients:** 541
- **Original features:** 42
- **Features used:** 38 (after dropping ID columns and high-correlation features)
- **Target:** Binary — PCOS (1) / Healthy (0)
- **Class distribution:** ~33% PCOS, ~67% Healthy

---

## 🗺️ Roadmap

### ✅ v1.0 — Tabular Model (current)
- XGBoost classifier on 38 clinical features
- FastAPI + HTML frontend
- Docker deployment on HuggingFace Spaces

### 🔲 v2.0 — Multimodal System
- [ ] **EfficientNetB0** integration for ultrasound image classification (128×128, custom classification head)
- [ ] Combined tabular + image inference pipeline
- [ ] Unified confidence score from both modalities
- [ ] Separate image upload UI panel

### 🔲 v3.0 — Clinical Grade
- [ ] LIME / SHAP explainability panel (feature contribution per prediction)
- [ ] Patient session history tracking
- [ ] PDF report generation
- [ ] REST API with token-based authentication
- [ ] Multilingual UI (TR / EN)

---

## 🛠️ Tech Stack

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

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Kübra Parmak**

- GitHub: [@KbrPrmk](https://github.com/KbrPrmk)
- HuggingFace: [@KubraParmak](https://huggingface.co/KubraParmak)
