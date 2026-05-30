
Telangana PDS Analytics – Clustering & Policy Impact Analysis

Project Overview

This project analyzes the Telangana Public Distribution System (PDS) using Machine Learning and Data Analytics techniques to identify shop behavior, portability trends, fraud patterns, and distribution efficiency.

The goal of this project is to cluster ration shops based on transaction behavior and identify meaningful insights for logistics optimization and policy analysis.


Problem Statement

The Telangana ration distribution system generates a large amount of transaction and card-level data. It is difficult to manually identify:

High-demand portability hubs
Fraud-prone ration shops
Shop transaction anomalies
Policy impact across regions

This project uses clustering techniques to group similar ration shops and generate actionable insights.

Objectives

Analyze Telangana PDS transaction data
Perform data cleaning and preprocessing
Detect and handle outliers
Apply feature engineering
Reduce dimensionality using PCA
Perform clustering analysis
Generate business and policy insights
Build an interactive Streamlit dashboard

Project Pipeline

Data Collection
Data Cleaning & Preprocessing
Missing Value Handling
Outlier Detection & Clipping
Encoding Categorical Features
Exploratory Data Analysis (EDA)
Skewness Analysis
Feature Engineering
Feature Scaling
PCA (Dimensionality Reduction)
Clustering Model Building
Cluster Interpretation & Profiling
Streamlit Dashboard Deployment

Technologies Used
Python
Pandas
NumPy
Scikit-Learn
Matplotlib
Seaborn
Plotly
Streamlit
Jupyter Notebook

Project Structure
Telangana PDS Analytics/
│── app/
│── data/
│── models/
│── notebooks/
│── requirements.txt
│── README.md



## PCA Results
Using Kmean

data in  4 clusters bane hain:

Cluster 0 → Most bigest group (~46%)
Cluster 1 → second biggest (~40%)
Cluster 2 → smaller segment (~7.5%)
Cluster 3 → smaller segment (~6.5%)

using DBCA
- Original Features: 69
- Reduced Features: 15
- Variance Retained: 86%

## DBSCAN Results
- Best eps value: 3.0
- Total Clusters: 7
- Noise Points: 870
##  Compare:
 Kmean is better Approch for my data

 ## Installation

```bash
pip install -r requirements.txt
```

## Run Project

```bash
streamlit run app.py
```

## Author
Priyanka Kumari



