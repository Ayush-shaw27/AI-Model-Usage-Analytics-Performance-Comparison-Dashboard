# 🚀 AI Model Analytics & Decision Intelligence Platform

An end-to-end **AI Model Analytics Dashboard** that enables users to compare, evaluate, and make data-driven decisions across multiple AI models using real-world datasets.

---

## 🔗 Live Demo

👉 [https://ai-model-usage-analytics-performance-comparison-dashboard.streamlit.app/](https://ai-model-usage-analytics-performance-comparison-dashboard.streamlit.app/)

---

## 📌 Project Overview

This project is a **full-stack data analytics and machine learning platform** designed to:

- Analyze AI model performance across multiple metrics  
- Compare models based on cost, accuracy, latency, and speed  
- Provide intelligent recommendations  
- Predict model cost using machine learning  

It integrates:
- **Data Engineering Pipeline**
- **Machine Learning Models**
- **Interactive Dashboard (Streamlit + Plotly)**

---

## 🎯 Objectives

- Build a **real-world analytics system (no synthetic data)**
- Enable **decision intelligence for AI model selection**
- Create an **industry-ready dashboard**
- Combine **Data + ML + UI into one system**

---

## ⚙️ Features

### 📊 1. Interactive Dashboard
- Cost vs Accuracy visualization  
- Speed vs Latency analysis  
- Provider comparison  
- Correlation heatmaps  

---

### 🧠 2. Machine Learning Integration
- KMeans Clustering (Model Segmentation)
- Composite Score Ranking System
- Outlier Detection (Z-score)
- Linear Regression (Cost Prediction)

---

### ⚖️ 3. Model Comparison
- Compare multiple models side-by-side  
- Radar charts for performance analysis  
- Multi-metric benchmarking  

---

### 🎯 4. Recommendation Engine
- Input:
  - Budget  
  - Minimum accuracy  
- Output:
  - Top models ranked by performance  

---

### 📈 5. Cost Prediction Tool
Predict model cost using:
- Accuracy  
- Speed  
- Latency  

---

## 🏗️ System Architecture

```

Raw Dataset
↓
Data Preprocessing (data.py)
↓
Feature Engineering
↓
ML Pipeline (ml_models.py)
↓
Streamlit Dashboard (app.py)

```

---

## 📂 Project Structure

```

AI-Model-Analytics/
│
├── app.py                         # Streamlit dashboard (UI layer)
├── data.py                        # Data preprocessing & pipeline
├── ml_models.py                   # ML models & analytics logic
├── final_dataset.csv              # Generated dataset (auto-created)
│
├── data/
│   ├── ai_models_performance.csv
│   ├── open_llm_leaderboard_train.csv
│
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation

```

---

## 📊 Dataset

This project uses **real-world datasets (no synthetic data)**:

### 1️⃣ AI Model Performance Dataset
Contains:
- Model Name  
- Provider  
- Cost (USD per 1M tokens)  
- Speed (tokens/sec)  
- Latency (seconds)  

---

### 2️⃣ Open LLM Leaderboard Dataset
Contains:
- Model evaluation scores  
- Benchmark accuracy  

---

## 📥 Dataset Links

Add your dataset links here:

```


[AI Models Dataset] : <(https://www.kaggle.com/datasets/asadullahcreative/ai-models-benchmark-dataset-2026-latest)>
[LLM Leaderboard Dataset] : <(https://artificialanalysis.ai/leaderboards/providers)>
[Open LLM Leaderboard Archived] : <(https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/)>

````

---

## 🧠 Machine Learning Concepts Used

- **Clustering:** KMeans (Model Segmentation)  
- **Regression:** Linear Regression (Cost Prediction)  

### Feature Engineering:
- Cost Efficiency  
- Speed Efficiency  
- Composite Score  

- **Outlier Detection:** Z-score method  

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-link>
cd AI-Model-Analytics
````

---

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📈 Business Use Cases

* AI model selection for companies
* Cost optimization in LLM usage
* Performance benchmarking
* Decision support for AI adoption

---

## 💡 Key Highlights

* ✅ Real-world dataset integration (no dummy data)
* ✅ End-to-end pipeline (Data → ML → Dashboard)
* ✅ Industry-level UI dashboard
* ✅ Decision Intelligence system
* ✅ Production-ready structure

---

## 🔮 Future Enhancements

* API integration for real-time model data
* Advanced ML models (XGBoost, Deep Learning)
* User authentication system
* Cloud deployment (AWS/GCP)
* Automated data updates

---

## 👨‍💻 Author

**Ayush Shaw**

---

## ⭐ Conclusion

This project demonstrates a complete **Data Analytics + Machine Learning + Product Development pipeline**
