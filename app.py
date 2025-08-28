import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ------------------- Streamlit Config -------------------
st.set_page_config(page_title="Clustering App", layout="wide")
st.title("🔎 Clustering Analysis App")

# ------------------- Sample CSV -------------------
st.sidebar.subheader("📂 Sample Dataset")

sample_path = os.path.join("data", "sample.csv")
if os.path.exists(sample_path):
    with open(sample_path, "rb") as f:
        st.sidebar.download_button(
            label="⬇️ Download Sample CSV",
            data=f,
            file_name="sample.csv",
            mime="text/csv"
        )
else:
    st.sidebar.warning("⚠️ Sample CSV not found in /data folder")

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    # Data Info
    st.write("**Shape:**", df.shape)
    st.write("**Null Values:**")
    st.write(df.isnull().sum())
    st.write("**Duplicates:**", df.duplicated().sum())

    # ------------------- Preprocessing -------------------
    st.subheader("⚙️ Data Preprocessing")
    df_clean = df.dropna().drop_duplicates()
    st.write("Cleaned dataset shape:", df_clean.shape)

    numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[numeric_cols])

    st.success("Data scaled and ready for clustering.")

    # ------------------- Clustering -------------------
    st.subheader("📊 Clustering")
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df_clean["Cluster"] = kmeans.fit_predict(X)

    st.write("Clustered Data")
    st.dataframe(df_clean.head())

    # ------------------- Evaluation -------------------
    st.subheader("📈 Evaluation Metrics")
    silhouette = silhouette_score(X, df_clean["Cluster"])
    st.metric(label="Silhouette Score", value=f"{silhouette:.3f}")

    # Elbow method
    wcss = []
    K = range(2, 10)
    for i in K:
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X)
        wcss.append(km.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(K, wcss, "bo-")
    ax1.set_xlabel("k")
    ax1.set_ylabel("WCSS")
    ax1.set_title("Elbow Method")
    st.pyplot(fig1)

    # ------------------- Cluster Visualization -------------------
    st.subheader("🖼️ Cluster Visualization")
    if X.shape[1] >= 2:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=df_clean["Cluster"], palette="viridis", ax=ax2)
        ax2.set_title("Clusters (first 2 features)")
        st.pyplot(fig2)
    else:
        st.warning("Not enough features to show scatter plot.")

    # ------------------- Save / Load Model -------------------
    st.subheader("💾 Save & Load Model")

    if st.button("Save Model"):
        os.makedirs("models", exist_ok=True)
        joblib.dump(kmeans, "models/kmeans_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        st.success("✅ Model and Scaler saved in models/ folder.")

    if st.button("Load Model"):
        try:
            loaded_model = joblib.load("models/kmeans_model.pkl")
            loaded_scaler = joblib.load("models/scaler.pkl")
            st.success("✅ Model and Scaler loaded successfully.")

            # Re-run prediction with loaded model
            X_loaded = loaded_scaler.transform(df_clean[numeric_cols])
            df_clean["Cluster_Loaded"] = loaded_model.predict(X_loaded)
            st.write("Predictions using loaded model:")
            st.dataframe(df_clean.head())

        except FileNotFoundError:
            st.error("❌ No saved model found. Please save first.")

    # ------------------- Download Results -------------------
    st.subheader("⬇️ Download Results")
    csv = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("Download Clustered Data", data=csv,
                       file_name="clustered_data.csv", mime="text/csv")
