# src/clustering_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from packaging.version import Version
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# -------- Load Dataset --------
file_path = r"C:\Users\bhumi\Desktop\AI capstone project\my capstone 1\project\data\sample.csv"   
user_data = pd.read_csv(file_path)
print("✅ Data Loaded")

numeric_columns = ['Instagram visit score', 'Spending_rank(0 to 100)']

# -------- Scaling --------
# normalization features using min-max scaler
scaler = MinMaxScaler()  
user_data_scaled = user_data.copy()
user_data_scaled[numeric_columns] = scaler.fit_transform(user_data[numeric_columns])
print("✅ Scaling Complete")

# =============
# check outlier
# =============

for col in numeric_columns:

    #first quartile (Q1)
    Q1 = user_data[col].quantile(0.25)

    #third quartile (Q3)
    Q3 = user_data[col].quantile(0.75)
    
    #Interquartile Range (IQR), majority of the data points lie
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = user_data[(user_data[col] < lower_bound) | (user_data[col] > upper_bound)]
    print(outliers)
    
    user_data = user_data.drop(outliers.index)
print("✅ Outlier Complete")

# create variable for dataset we will use in modeling
X = user_data_scaled.copy()
X = X.drop('User ID', axis=1)

model_3 = KMeans(n_clusters=4, random_state=0)
model_3.fit(X)
print("✅ Model fitting Complete")

# -------- Silhouette Score --------
score= silhouette_score(X, model_3.fit_predict(X))
print(f"📌 Silhouette Score: {score:.3f}")


# -------- Cluster Visualization (first 2 features) --------
# get cluster labels
labels = model_3.labels_

# create numpy array of X for plotting purpose
X_np = np.array(X)

# get centroid position
centroids = model_3.cluster_centers_

# visualize cluster
plt.figure(figsize=(12, 8))
 
# plot data
plt.scatter(X_np[:, 0], X_np[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, edgecolors='w', marker='o')
 
# plot centroid
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')

# add centroid label on plot
for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1], f'Centroid {i+1}', color='red', fontsize=12, ha='center', va='center')
 
# add title and label
plt.title('Cluster Visualization With Centroid')
plt.xlabel('Instagram visit score')
plt.ylabel('Spending_rank(0 to 100)')
plt.legend()
 
plt.show()
 
# print centroid value
print("Centroids Value:")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i+1}: Instagram visit score = {centroid[0]:.2f}, Spending_rank(0 to 100) = {centroid[1]:.2f}")

