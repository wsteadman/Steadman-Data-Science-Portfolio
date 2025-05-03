# KMeans Clustering Explorer 📊

## Project Overview 🚀
This interactive app helps users explore unsupervised learning through KMeans clustering. You can upload your own dataset or choose from built-in demos (Iris, Wine, Breast Cancer) to find natural clusters in the data. The app includes automatic preprocessing, feature scaling, PCA visualization, and methods to determine the optimal number of clusters — all with helpful visual aids and explanations, even for users new to clustering!

## Instructions 🛠️
### Click the link to get started!
[link_to_kmeans_app]()

**Using the App**
- Upload a CSV file or use one of the demo datasets
- (Optional) Select a target column to view ground truth labels (not used by the model)
- Choose how many clusters (k) to find
- Optionally scale the features for better KMeans performance
- Click Run Clustering to view cluster plots, elbow method charts, and clustered data

**Troubleshooting**
- Ensure your data is in CSV format

Check that all selected columns are numerical (non-numeric columns will be one-hot encoded)

If clustering results seem poor, try scaling features or adjusting k to better match the underlying data

## Required Libraries 📚
- matplotlib==3.10.1
- numpy==1.26.4
- pandas==2.2.3
- scikit_learn==1.6.1
- streamlit==1.37.1

## App Features 📁
*for more info about any step or visual, just use the app!*

Data Handling
📂 Load Datasets: Choose from Iris, Wine, Breast Cancer, or upload your own CSV

🧹 Preprocessing: Automatically handles missing values and encodes categorical features

📏 Scaling: Standardizes data (mean = 0, std = 1) to improve clustering results

Clustering Analysis
KMeans Clustering 🔍

Choose k (number of clusters) via slider

Optionally scale features before clustering

Uses scikit-learn's KMeans under the hood

Visual Tools 📈

🧭 2D PCA Projection: See how clusters are separated in two principal components

🧠 Cluster Interpretability: Visualize and label clusters by comparing with original class labels (if available)

Optimal K Analysis
Elbow Method 💡

Plots Within-Cluster Sum of Squares (WCSS) to show when adding more clusters no longer improves compactness significantly

Silhouette Score 📐

Measures how well-separated your clusters are

Helps determine the most "natural" number of clusters for your dataset

Results Exploration
📊 Clustered Data Table: View your dataset with added cluster assignments

📦 Cluster Size Bar Chart: See how many samples fall into each cluster

🎯 True Label Comparison (optional): See how well clusters align with ground truth classes

## Visuals
<br>
*App interface with clustering visualization:*
<br>
<img src="" alt="image" width="700" />
<br><br>

*Elbow Method & Silhouette Score:*
<br>
<img src="" alt="image" width="700" />
<br><br>

*Clustered Data Table:*
<br>
<img src="" alt="image" width="700" />
<br><br>

## References:
[_Notes]()

[_Notes]()