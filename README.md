# 🚀 AI Model Analytics & Decision Intelligence Platform

An end-to-end **AI Model Analytics Dashboard** that enables users to compare, evaluate, and make data-driven decisions across multiple AI models using real-world datasets.

---

## 🔗 Live Demo

👉 [https://ai-model-usage-analytics-performance-comparison-dashboard.streamlit.app/](https://ai-model-usage-analytics-performance-comparison-dashboard.streamlit.app/)

---

## 📌 Project Overview

AI Model Analytics & Decision Intelligence Platform
Author: Ayush Shaw(16014223027)

1.	Project Overview
The AI Model Analytics & Decision Intelligence Platform is a comprehensive Streamlit-based web dashboard designed to help users compare, rank, and select Large Language Models (LLMs) based on real-world performance metrics.
It integrates data from Artificial Analysis and Hugging Face Open LLM Leaderboard, processes it through a clean data pipeline (data.py), applies machine learning techniques (ml_models.py), and presents interactive visualizations in a modern dark-themed UI (app.py).
Key Features:
•	Real-time filtering by provider, cost, accuracy, latency
•	Composite scoring, KMeans clustering, outlier detection
•	Cost prediction using Linear Regression
•	Smart recommendation engine
•	Side-by-side model comparison and radar charts 

2. Overview Tab
The Overview tab gives an immediate snapshot of the current filtered dataset.
What it delivers:
•	KPI Cards: Total models shown, average accuracy, cost per 1M tokens, latency, and speed.
•	Quick Insights: Four intelligent highlight cards showing the most cost-efficient model, cheapest model, fastest model, and highest accuracy model.
•	Top 10 Ranked Models: Sorted table with composite score (higher is better).
•	Cost vs Accuracy Scatter Plot: Visual trade-off between price and performance with bubble size representing speed and color-coded clusters.
This tab helps users instantly understand the market landscape and identify top performers without deep analysis.  
 
 
3. Analytics Tab
The Analytics tab provides deeper statistical insights and visualizations.
What it delivers:
•	Speed vs Latency Scatter: Shows the classic trade-off between response time and throughput.
•	Top 10 Bar Chart: Horizontal bar chart of the best models by composite score.
•	Provider Comparison: Side-by-side bar charts comparing average accuracy, cost, and latency across providers (OpenAI, Anthropic, Google, xAI, etc.).
•	Cost Efficiency Distribution: Histogram + box plot showing how cost-efficiency is distributed across performance segments.
•	Feature Correlation Heatmap: Pearson correlation matrix revealing relationships between accuracy, cost, latency, speed, and composite score.
This section is extremely useful for understanding market trends and provider strengths/weaknesses.
  
4. ML Insights Tab
The ML Insights tab showcases the machine learning layer of the platform.
What it delivers:
•	KMeans Clustering Scatter: Models automatically grouped into three segments - High-Performance, Budget, and Balanced (based on accuracy, cost, latency, and speed).
•	Cluster Statistics Table: Detailed metrics (count, avg accuracy, cost, latency, speed, composite score) for each segment.
•	Outlier Detection: Flags models that are unusually expensive or slow using Z-score (>2.5).
•	Segment Radar Chart: Multi-axis radar plot comparing the three clusters across all key metrics.
This tab demonstrates how unsupervised ML (KMeans) and statistical techniques turn raw data into actionable intelligence.
  
5. Model Comparison Tab
The Model Comparison tab allows side-by-side evaluation of 2-6 selected models.
What it delivers:
•	Comparison Table: Clean formatted table with all key metrics.
•	Grouped Bar Charts: One chart per metric (Accuracy, Cost, Latency, Speed, Composite Score) for visual comparison.
•	Multi-Model Radar Chart: Overlaid radar plots showing strengths and weaknesses of each selected model at a glance.
This is the most practical tab for decision-making when shortlisting models for a specific project.
  
6. Tools Tab
The Tools tab contains two powerful decision-making utilities.
What it delivers:
•	Model Recommendation Engine: User inputs maximum budget and minimum accuracy → system returns the top-N best models ranked by composite score.
•	Cost Predictor: Linear Regression model (R² displayed) that predicts cost per 1M tokens based on desired accuracy, speed, and latency.
These tools make the dashboard not just analytical but truly prescriptive.   
7. Conclusion
The AI Model Analytics & Decision Intelligence Platform successfully transforms fragmented LLM benchmark data into an intuitive, interactive, and intelligent decision-support system.
Key achievements:
•	Clean data pipeline with realistic filtering (no zero-cost or hallucinated efficiency values).
•	Robust ML layer (clustering, regression, outlier detection, composite scoring).
•	Professional dark-themed UI with fully interactive Plotly visualizations.
•	Practical tools that directly help users choose the right model under budget and performance constraints.
This project demonstrates end-to-end skills in data engineering, machine learning, and full-stack dashboard development




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

[AI Models Dataset] : https://www.kaggle.com/datasets/asadullahcreative/ai-models-benchmark-dataset-2026-latest

[LLM Leaderboard Dataset] : https://artificialanalysis.ai/leaderboards/providers

[Open LLM Leaderboard Archived] : https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/

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
