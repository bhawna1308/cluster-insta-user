

import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from src.clustering_model import model_3, X
from src.data_load import df

# -------- Silhouette Score --------
score= silhouette_score(X, model_3.fit_predict(X))
print(f"📌 Silhouette Score: {score:.3f}")

# -------- Analysis & Conclusion --------
X['Cluster'] = model_3.labels_ + 1
X.insert(0, 'User ID', df['User ID']) # 0 means first col position
print(X.head(3))

# -------- Clusters grouping--------
groups=X.groupby(['Cluster'])[['Instagram visit score', 'Spending_rank(0 to 100)']].agg(['mean'])*100

print(f"📌 Total clusters: {groups}")
