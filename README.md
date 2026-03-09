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
