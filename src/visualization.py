# src/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------- Load Preprocessed Data --------
file_path = r"C:\Users\bhumi\Desktop\AI capstone project\my capstone 1\project\data\sample.csv"   # Change path if needed
df = pd.read_csv(file_path)
print("✅ Data Loaded for Visualization")

numeric_columns = ['Instagram visit score', 'Spending_rank(0 to 100)']

# create grid size
rows, cols = 1, 2

# create figure and subplots
plt.figure(figsize=(8, 5))

# plot each feature in subplot
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram of {col}")
    plt.ylabel('Frequency')

# layout
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()


# ==================================
# check correlation between features
# ==================================
sns.set_style('darkgrid')
sns.regplot(x='Instagram visit score', y='Spending_rank(0 to 100)', data=df)
plt.title("Scatter Plot")
plt.show()