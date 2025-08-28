import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.data_load import df

scaler = StandardScaler()
X = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Save model
joblib.dump(kmeans, "model/kmeans_model.pkl")

# Save scaler too (important for preprocessing new data consistently)
joblib.dump(scaler, "model/scaler.pkl")
