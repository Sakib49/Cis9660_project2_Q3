
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="CIS 9660 Q3 — Segmentation & Basket (Hotels)", layout="wide")
st.title("CIS 9660 — Q3: Customer Segmentation & Association Rules (Hotels)")

st.markdown(
    """
This app performs **K-Means** segmentation (with Elbow & Silhouette) and **Gaussian Mixture** as an alternative, 
then runs **Apriori** association rule mining over selected categorical attributes.

**Default dataset:** TidyTuesday Hotels CSV (hosted on GitHub).
"""
)

default_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"

with st.sidebar:
    st.header("Data Source")
    mode = st.radio("Choose data input", ["Use default URL", "Upload CSV"])
    url = st.text_input("CSV URL", value=default_url) if mode == "Use default URL" else None
    st.caption("Expected columns exist in the TidyTuesday hotels dataset.")

@st.cache_data(show_spinner=False)
def load_data(url: str = None, file=None) -> pd.DataFrame:
    if url:
        return pd.read_csv(url)
    else:
        return pd.read_csv(file)

df = None
if mode == "Use default URL":
    try:
        df = load_data(url=url)
        st.success("Loaded dataset from URL.")
    except Exception as e:
        st.error(f"Failed to load from URL: {e}")
else:
    upl = st.file_uploader("Upload a CSV", type=["csv"])
    if upl:
        try:
            df = load_data(file=upl)
            st.success("Loaded uploaded CSV.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

if df is None:
    st.stop()

st.subheader("Preview")
st.dataframe(df.head())

# ============ SEGMENTATION ============
st.header("1) Customer Segmentation")
st.markdown("We use numerical booking features for clustering; values are standardized.")

num_cols = [
    'lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
    'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'previous_cancellations',
    'booking_changes', 'days_in_waiting_list', 'adr', 'total_of_special_requests'
]

df_seg = df.copy()
if 'children' in df_seg.columns:
    df_seg['children'] = df_seg['children'].fillna(0)

# Drop rows with missing values in selected numeric columns
df_seg = df_seg.dropna(subset=[c for c in num_cols if c in df_seg.columns])
num_cols = [c for c in num_cols if c in df_seg.columns]
X = df_seg[num_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow
K_range = list(range(2, 11))
inertias, silhouettes = [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Elbow (Inertia vs K)")
    fig, ax = plt.subplots()
    ax.plot(K_range, inertias, marker='o')
    ax.set_xlabel("K"); ax.set_ylabel("Inertia (WSS)")
    st.pyplot(fig)
with col2:
    st.subheader("Silhouette vs K")
    fig, ax = plt.subplots()
    ax.plot(K_range, silhouettes, marker='o')
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score")
    st.pyplot(fig)

best_k = K_range[int(np.argmax(silhouettes))]
st.info(f"Suggested K by silhouette peak: **{best_k}**")

# Fit KMeans
km = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit(X_scaled)
df_seg['kmeans_cluster'] = km.labels_

# Profile clusters
st.subheader("K-Means Segment Profile (mean of features)")
profile_km = df_seg.groupby('kmeans_cluster')[num_cols].mean().round(2)
st.dataframe(profile_km)

# PCA 2D plot
st.subheader("K-Means Clusters in PCA(2D)")
pca = PCA(n_components=2, random_state=42)
pts = pca.fit_transform(X_scaled)
fig, ax = plt.subplots()
sns.scatterplot(x=pts[:,0], y=pts[:,1], hue=df_seg['kmeans_cluster'], palette='viridis', s=20, ax=ax, legend=True)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
st.pyplot(fig)

# Alternative: GMM + BIC
st.subheader("Alternative Clustering: Gaussian Mixture (BIC)")
bics = []
for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    bics.append(gmm.bic(X_scaled))
fig, ax = plt.subplots()
ax.plot(K_range, bics, marker='o')
ax.set_xlabel("Components"); ax.set_ylabel("BIC (lower is better)")
st.pyplot(fig)

best_gmm_k = K_range[int(np.argmin(bics))]
st.info(f"Best GMM components by BIC: **{best_gmm_k}**")

gmm = GaussianMixture(n_components=best_gmm_k, covariance_type='full', random_state=42).fit(X_scaled)
df_seg['gmm_cluster'] = gmm.predict(X_scaled)

st.subheader("GMM Segment Profile (mean of features)")
profile_gmm = df_seg.groupby('gmm_cluster')[num_cols].mean().round(2)
st.dataframe(profile_gmm)

# Downloads
st.download_button("Download KMeans Profiles CSV", profile_km.to_csv().encode("utf-8"), file_name="kmeans_segment_profile.csv")
st.download_button("Download GMM Profiles CSV", profile_gmm.to_csv().encode("utf-8"), file_name="gmm_segment_profile.csv")

# ============ ASSOCIATION RULES ============
st.header("2) Association Rule Mining (Apriori)")
st.markdown("We use categorical attributes as 'items'. Adjust thresholds to explore rules.")

cat_cols = ['hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type']
cat_cols = [c for c in cat_cols if c in df.columns]
if not cat_cols:
    st.warning("No categorical columns found for association rules. Upload a dataset with categorical attributes.")
else:
    df_rules = df.dropna(subset=cat_cols).copy()
    basket = pd.get_dummies(df_rules[cat_cols], drop_first=False).astype(int)

    min_sup = st.slider("Min Support", 0.01, 0.2, 0.05, 0.01)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.25, 0.05)
    min_lift = st.slider("Min Lift", 1.0, 5.0, 1.5, 0.1)

    try:
        itemsets = apriori(basket, min_support=min_sup, use_colnames=True)
        rules = association_rules(itemsets, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False)

        # Tidy the set types for display
        def fs_to_str(s):
            return ", ".join(sorted(list(s))) if isinstance(s, (set, frozenset)) else str(s)
        if not rules.empty:
            rules_view = rules.copy()
            rules_view["antecedents"] = rules_view["antecedents"].apply(fs_to_str)
            rules_view["consequents"] = rules_view["consequents"].apply(fs_to_str)
            st.write(f"Showing **{len(rules_view)}** rules matching filters.")
            st.dataframe(rules_view[['antecedents','consequents','support','confidence','lift','leverage','conviction']].head(200))
            st.download_button("Download Rules CSV", rules_view.to_csv(index=False).encode("utf-8"), file_name="association_rules.csv")
        else:
            st.info("No rules matched the thresholds. Lower support or confidence and try again.")
    except Exception as e:
        st.error(f"Association mining failed: {e}")

st.markdown("---")
st.caption("Built for CIS 9660 — Q3. Data: TidyTuesday Hotels.")
