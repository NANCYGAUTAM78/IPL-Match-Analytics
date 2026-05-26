# 🏏 IPL Match Analytics & Win Predictor

<div align="center">

![IPL Analytics Banner](https://img.shields.io/badge/IPL-Analytics%20Dashboard-orange?style=for-the-badge&logo=cricket&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-yellow?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-green?style=for-the-badge)

**An end-to-end Data Analytics & Machine Learning project on real IPL data (2008–2023)**  
*Exploratory Data Analysis · Win Prediction · Interactive Dashboard*

</div>

---

## 📊 EDA Visualizations

### 🏆 Total Wins by Team
![Team Wins](p1_team_wins.png)

### 📈 Season-wise Scoring Trend
![Season Trend](p2_season_trend.png)

### 🏃 Chase Success Rate
![Chase Success](p3_chase.png)

### ⚔️ Head-to-Head Win % Matrix
![H2H Heatmap](p4_h2h.png)

### 🤖 Feature Importance
![Feature Importance](p5_feature_imp.png)

### 📉 Model Accuracy Comparison
![Model Comparison](p6_model.png)

---

## 📌 Project Overview

This project performs a **complete data analytics pipeline** on the IPL dataset from Kaggle, covering:

- 🧹 **Data Cleaning** — Standardised team names, fixed season formats, handled nulls
- 📊 **Exploratory Data Analysis** — 12 visualizations covering wins, venues, toss impact, powerplay, scoring trends
- 🤖 **Machine Learning** — Logistic Regression & Random Forest to predict match outcomes
- 🌐 **Interactive Dashboard** — Browser-based dashboard with 4 tabs and a live win predictor

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10 |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Chart.js |
| **Machine Learning** | Scikit-learn (Logistic Regression, Random Forest) |
| **Dashboard** | HTML, CSS, JavaScript |
| **Data Source** | Kaggle IPL Complete Dataset |

---

## 📁 Project Structure

```
IPL-Match-Analytics/
│
├── 📄 ipl_analytics_kaggle.py      # Main EDA + ML script
├── 📄 ipl_powerbi_export.py        # Power BI data exporter
├── 🌐 IPL_Dashboard.html           # Interactive browser dashboard
│
├── 📊 matches.csv                  # Kaggle IPL match data (1090 rows)
├── 📊 deliveries.csv               # Ball-by-ball data (260K+ rows)
├── 📊 ipl_matches_clean.csv        # Cleaned match data
├── 📊 ipl_team_stats.csv           # Team-wise statistics
└── 📊 ipl_season_summary.csv       # Season-wise summary
```

---

## 🚀 How to Run

```bash
# 1. Clone repo
git clone https://github.com/NANCYGAUTAM78/IPL-Match-Analytics.git
cd IPL-Match-Analytics

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Run analytics
python ipl_analytics_kaggle.py

# 4. Open dashboard — just double-click!
# IPL_Dashboard.html → opens in browser
```

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| 📦 Total Matches Analyzed | **1,090** |
| 🗃️ Delivery Records | **260,920+** |
| 📅 Seasons Covered | **2008 – 2023** |
| 🎯 LR Model Accuracy | **56.9%** |
| 🌲 Random Forest Accuracy | **51.8%** |
| 📊 Above Random Baseline | **+6.9%** |

### 🔍 Key Findings
- 🏏 Toss impact is minimal — winner wins only **50.2%** of matches
- 🏃 **RCB** has highest chase success rate at **65.7%**
- 📍 **M. Chinnaswamy Stadium** hosted the most matches (130)
- 📈 **Historical win rate** is the strongest predictor of outcome
- 🌟 **CSK dominates MI** with **74.3%** head-to-head win rate

---

## 🌐 Interactive Dashboard

| Tab | Features |
|-----|----------|
| 📊 **Overview** | KPI cards, season trend, toss analysis, team wins |
| 🏆 **Teams** | Win leaderboard, score distribution, venue stats |
| ⚔️ **Matchups** | H2H win % matrix, rivalry highlights |
| 🤖 **Predictor** | Live win probability tool |

---

## 👩‍💻 Author

**Nancy Gautam**  
📧 nancygautam7890@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/nancy-gautam78)  
🐙 [GitHub](https://github.com/NANCYGAUTAM78)

---

<div align="center">
⭐ <b>If you found this helpful, please give it a star!</b> ⭐
</div>