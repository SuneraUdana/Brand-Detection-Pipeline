# Brand Detection Pipeline 🧠👗

An automated fashion pattern detection and brand analysis pipeline built on **Databricks**, powered by a **Medallion Architecture** (Bronze → Silver → Gold) and deployed with a **CI/CD pipeline via GitHub Actions**.

---

## 🏗️ Architecture Overview
  Raw CSV Data (Databricks Volume)
              ↓
[Bronze Layer] — Raw ingestion as Delta table
              ↓
[Silver Layer] — Cleaned, confidence-tiered, null-filtered
              ↓
[Gold Layer] — Aggregated stats, ML-ready features
              ↓
[CI/CD] — Auto-refreshes on every GitHub push

---

## 📁 Project Structure
Brand-Detection-Pipeline/
├── .github/
│ └── workflows/
│ └── databricks-ci.yml # GitHub Actions CI/CD workflow
├── sql/
│ ├── bronze.sql # Raw CSV → Delta table
│ ├── silver.sql # Cleaned & enriched layer
│ └── gold.sql # Aggregated ML-ready layer
├── src/
│ └── run_pipeline.py # Python runner for Bronze→Silver→Gold
├── fashionData/ # Source fashion dataset
└── README.md


---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| **Databricks** | Cloud data platform (SQL Warehouse) |
| **Delta Lake** | Storage format for all three layers |
| **Python** | Pipeline runner and data connection |
| **databricks-sql-connector** | Python ↔ Databricks SQL Warehouse |
| **GitHub Actions** | CI/CD automation on every push |

---

## 📊 Dataset

Fashion pattern classification dataset containing:
- **16+ clothing pattern categories** (plain, floral, stripes, polka dot, tribal, ikat, geometry, etc.)
- **13,000+ labelled items** with confidence scores
- Source image URLs (S3)

---

## 🥉🥈🥇 Medallion Layers

### Bronze
- Raw CSV ingested directly from Databricks Volume
- No transformations — pure source of truth

### Silver
- Null rows removed
- Category lowercased and trimmed
- Confidence score rounded
- `confidence_tier` column added: `high (≥0.8)` / `medium (≥0.5)` / `low (<0.5)`

### Gold
- Aggregated per category: total items, avg/min/max confidence
- High / medium / low confidence counts and percentages
- Ready for ML feature engineering and dashboarding

---

## ⚙️ CI/CD Pipeline

Every push to `main` automatically:
1. Spins up a GitHub Actions runner
2. Installs `databricks-sql-connector`
3. Connects to the Databricks SQL Warehouse
4. Refreshes Bronze → Silver → Gold tables

Credentials are stored securely as **GitHub Actions Secrets** — never hardcoded.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Databricks workspace with SQL Warehouse
- GitHub repository secrets configured

### Local Setup

```bash
git clone https://github.com/SuneraUdana/Brand-Detection-Pipeline.git
cd Brand-Detection-Pipeline
python -m venv .venv
source .venv/bin/activate
pip install databricks-sql-connector
