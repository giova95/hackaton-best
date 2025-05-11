# InsureTrust: Client Loyalty Project for Mutual Insurance

## BEST Hackathon 2025 - Team 11 Solution

<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
<img src="https://img.shields.io/badge/Status-Completed-success.svg" alt="Status">

A data-driven insurance customer segmentation and marketing strategy solution developed for the InsureTech: Analyze Data, Innovate, Win! Hackathon.

## Team 11 Members

- [Riccardo Becciolini](https://github.com/Beccio00)
- [Marco Donatucci](https://github.com/marcodonatucci)
- [Loris Catalano](https://github.com/loris-catalano)
- [Giovanni Angeli](https://github.com/giova95)

## 📋 Project Overview

InsureTrust is a comprehensive data analytics solution designed to help mutual insurance companies segment their customer base for personalized marketing and risk management. Through advanced clustering techniques and data enrichment, our solution identifies distinct customer profiles based on risk indicators and customer lifetime value.

### Project Goals

- Identify hidden patterns in claims data
- Understand factors influencing customer risk and value
- Maximize Customer Lifetime Value (CLV) while minimizing risk
- Propose a product recommendation system based on customer profiles

## 🔧 Technical Approach

Our solution follows a structured four-phase analytical process:

### 1. Exploratory Data Analysis (EDA)

- Understanding data structure
- Assessing quality
- Detecting anomalies and missing values

### 2. Data Preprocessing

- Built a cleaning pipeline for outliers and missing data
- IQR-based detection on claim amounts and age with winsorization
- Contextual imputation using geographic and vehicle relationships

### 3. Feature Engineering & Data Enrichment

- Created temporal features (year, month, day) to capture seasonal trends
- Calculated loss ratio as a key performance metric
- Developed demographic binning (age groups, claim categories)
- Integrated simulated internal and external data sources
- Developed combined risk and customer value indices

### 4. Segmentation and Clustering

- Applied K-Means clustering algorithm
- Used PCA for dimensionality reduction and visualization
- Selected optimal number of clusters using silhouette and elbow methods

## 📈 Key Visualizations

The project includes several visualization types:

- **Exploratory Visuals**: Boxplots, histograms, correlation heatmaps
- **KPI Visuals**: Time series charts, colored bar charts, geographic maps
- **Cluster Visuals**: PCA scatter plots, radar charts, parallel coordinates
- **Interactive Dashboards**: KPI dashboard and cluster profiler

## 💡 Key Insights & Business Value

### Customer Segments

Our analysis revealed clear behavior and value patterns that justify differentiated marketing strategies. The clusters provide actionable insights for:

- **High-value, low-risk segments**: Ideal for retention and cross-selling
- **High-risk, low-value segments**: Targets for risk mitigation or premium adjustments
- **Channel optimization**: Data-driven acquisition strategy improvement
- **Seasonal and geographic targeting**: For time and location specific campaigns

### Business Impact

- **Increased Profitability**: Through segment-based pricing strategies
- **Improved Customer Retention**: Enabled by personalized offers
- **Marketing Optimization**: Better budget allocation based on expected ROI by segment
- **Risk Reduction**: Via targeted mitigation strategies for high-risk groups

## 🚀 Implementation Plan

1. **Pilot Phase**: Test strategies on a sample from each cluster
2. **Measurement**: Track KPIs such as conversion, upsell, retention, satisfaction
3. **Optimization**: Refine strategies based on pilot results
4. **Scale-Up**: Roll out to the full customer base

## 📊 Future Enhancements

- Incorporate additional behavioral data (app usage, contact frequency, feedback)
- Explore predictive modeling to anticipate churn or cross-selling potential
- Prototype real-time recommendation systems
- Test dynamic pricing models linked to combined risk-value indicators

---

_This project was developed during the BEST Hackathon in May 2025 at Reale ITES._
