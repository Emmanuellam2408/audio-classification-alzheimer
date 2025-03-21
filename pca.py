import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
cwd = os.getcwd()
file_path = os.path.join(cwd, 'au_cla_alz_master', 'final_important_features_10_15_sec.csv')# Change to your actual file path
df = pd.read_csv(file_path)

# convert class labels to numeric for PCA visualization
df["dx"] = df["dx"].map({"Control": 0, "ProbableAD": 1})

# Select only numeric features while excluding target variable and filenames
feature_columns = [col for col in df.columns if col not in ["File", "dx"]]

# standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_columns])

# apply PCA (reduce to 2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for visualization
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Diagnosis"] = df["dx"].map({0: "Control", 1: "ProbableAD"})

# Scatter plot of PCA components
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca["PC1"], y=df_pca["PC2"], hue=df_pca["Diagnosis"], alpha=0.7, palette=["cyan", "orange"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Features (2D)")
plt.legend(title="Diagnosis")
plt.show()
