
---

## 🚀 Features

### 📊 Tab 1 — Pipeline Dashboard
- Live KPIs: Total predictions, accuracy, correct vs wrong counts
- Per-class accuracy horizontal bar chart
- Normalized confusion matrix heatmap
- Top misclassification pairs
- Prediction distribution (donut chart)
- Gold table summary from Databricks
- Filterable predictions explorer
> *Requires Databricks connection — gracefully disabled when credits are unavailable*

### 🏷️ Tab 2 — Automated Product Tagger
- Upload any clothing image → instant CNN classification
- Top-3 predictions with confidence scores
- Confidence thresholds: ✅ High / ⚠️ Low / ❌ Uncertain
- One-click save to Databricks Delta table (`product_tagging_log`)

### 🔍 Tab 3 — Visual Search
- Upload a query image → retrieve top 5 visually similar catalog items
- Feature extraction from CNN flatten layer
- Cosine similarity over 10,000 catalog embeddings
- Results displayed with similarity scores

### 🏭 Tab 4 — Warehouse Scanner
- Simulates a conveyor-belt item scanner
- KPIs: items scanned, accuracy, avg scan time (ms)
- Per-category accuracy breakdown
- Scan speed distribution histogram
- Full raw scan log explorer

### 📈 Tab 5 — Trend Forecasting
- Weekly category demand tracking across 10 clothing types
- Week-over-week (WoW) change % for each category
- Identifies hottest rising and sharpest falling categories
- Line chart: weekly volume trends
- Bar chart: WoW % changes with green/red color coding

### 📦 Tab 6 — Returns Reduction
- Flags items as High / Medium / Low return risk based on model confidence
- KPIs: total items, risk tier counts and percentages
- Donut chart: risk distribution
- Error rate by risk tier (bar chart)
- Filterable browse by risk tier

---

## 🧠 Model Performance

| Metric              | Value     |
|---------------------|-----------|
| Architecture        | CNN (Conv2D × 2 + Dense) |
| Dataset             | Fashion MNIST (70,000 images) |
| Training samples    | 60,000    |
| Test samples        | 10,000    |
| Overall Accuracy    | ~91%      |
| Classes             | 10        |
| Input shape         | 28×28×1   |

**Hardest classes** (most confused):
- Shirt ↔ T-shirt/Top
- Coat ↔ Pullover
- Sneaker ↔ Ankle boot

---

## 🗂️ Project Structure

fashion-mnist-dashboard/
│
├── app.py # Main Streamlit application (6 tabs)
├── requirements.txt # Python dependencies
├── .env # Databricks credentials (not committed)
├── .env.example # Template for environment variables
│
├── models/
│ ├── fashion_cnn.keras # Trained CNN model
│ ├── catalog_embeddings.npy # Pre-computed feature embeddings (10K)
│ └── catalog_labels.npy # Corresponding class labels
│
├── data/ # (gitignored — generated locally)
│ ├── warehouse_scan_log.csv # Simulated warehouse scan results
│ ├── trend_forecast_data.csv # Weekly trend simulation data
│ └── returns_risk_log.csv # Returns risk scoring data
│
└── notebooks/
└── ML-Model.ipynb # Model training + Databricks pipeline


---

## ⚙️ Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fashion-mnist-dashboard.git
cd fashion-mnist-dashboard

---

## ⚙️ Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fashion-mnist-dashboard.git
cd fashion-mnist-dashboard

python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

pip install -r requirements.txt

DATABRICKS_HOST=<your-workspace-url>
DATABRICKS_HTTP_PATH=<your-sql-warehouse-http-path>
DATABRICKS_TOKEN=<your-personal-access-token>

 Tech Stack
Layer	Technology
Data Engineering	Databricks, Delta Lake, PySpark
Model Training	TensorFlow / Keras, NumPy
Feature Extraction	Keras functional layers, TF functions
Similarity Search	scikit-learn (cosine similarity)
Dashboard	Streamlit, Plotly Express
Image Processing	Pillow (PIL)
Database	Databricks SQL Connector, Delta tables
Environment	Python 3.11, python-dotenv

🔮 Future Improvements
 Re-enable Databricks Tab 1 with renewed credits

 Add batch image upload for the Product Tagger

 Deploy to Streamlit Cloud

 Add FAISS for faster similarity search at scale

 Integrate MLflow for experiment tracking

 Build a REST API layer with FastAPI

👤 Author
Sunera Udana
BSc in Artificial Intelligence & Data Science RGU Aberdeen UK

🔗 LinkedIn - https://www.linkedin.com/in/sunera-wanninayaka-461358242/

💻 GitHub - https://github.com/SuneraUdana/fashion-mnist-dashboard

📧 suneraudana1@gmail.com
