
# CIS 9660 — Project #2, Question #3 (Streamlit App)

**Live App Goal:** Unsupervised BI for hotel bookings using **K-Means** + **GMM** for customer segmentation and **Apriori** for association rules.

## Features
- Load the **TidyTuesday Hotels** dataset from a public URL (or upload your CSV)
- Standardize numeric features and run **K-Means** with **Elbow + Silhouette**
- Alternative clustering via **Gaussian Mixture (GMM)** with **BIC**
- **PCA (2D)** visualization of clusters
- **Apriori** association rules with tunable support, confidence, and lift
- Download cluster profiles and rules as CSV

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
App will open at `http://localhost:8501`.

## Deploy (24/7) — Streamlit Community Cloud
1. Push these files to a GitHub repo:
   - `app.py`
   - `requirements.txt`
2. Go to https://share.streamlit.io/ → New app → point to your repo and main file `app.py`
3. Click Deploy. Share the public URL for grading.

## Dataset
Default: TidyTuesday Hotels CSV  
`https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv`

You can also upload your own CSV with similar columns.
