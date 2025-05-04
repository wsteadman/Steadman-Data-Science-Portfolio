# KMeans Clustering Explorer 📊

## ✨ Overview
- 📁 Upload your own dataset or choose from built-in demos (Iris, Wine, Breast Cancer)
- ⚙️ Automatically scale and preprocess your data
- 🔍 Explore clusters visually using PCA projection
- 📊 Use Elbow Method and Silhouette Score to find the optimal number of clusters
- 📋 View results in charts, tables, and detailed cluster assignments

## 🚀 Instructions
### Click the link to get started!
[link_to_kmeans_app](https://steadman-data-science-portfolio-pny9n2pvh9knl6q3kupnip.streamlit.app/)

Or, to run locally: 
1. Clone the Repository

        git clone https://github.com/your-username/classification-predictor.git
        cd classification-predictor 
3. Install Requirements

        pip install -r requirements.txt
4. Run the App

        streamlit run main.py
<br>

## How to Use the App

### 📁 Dataset Setup:
- Upload a CSV file or select one of the built-in demo datasets (Breast_Cancer, Iris, Wine).
- *(Optional) Select a target column to compare results to (not used by the model)*
- Choose how many clusters (k) to find
- Decide whether to scale the features for better KMeans performance

### 📈 Analysis
- 2D PCA Projection: Displays the data in two dimensions using PCA to show how well clusters are formed and separated
- Elbow Method Graph: Shows within-cluster sum of squares (WCSS) dropping for k values. The "elbow" (value after steepest drow-off) suggests best number of clusters
- Silhouette Score Graph: Measures how similar each point is to its own cluster vs. other clusters. Higher scores indicate better k values
- Clustered Data Table: View your dataset with added cluster (and optional true) label assignment(s)
- Cluster Size Bar Chart: See how many samples fall into each cluster and if any cluster(s) dominate

**🛠 Troubleshooting**
- Ensure your data is in CSV format
- If clustering results seem poor, try scaling features or adjusting k to better match the underlying data

## 📸 Visuals

<br>

### Main Interface

*App interface with clustering visualization:*
<br>
<div style="display: flex; justify-content: space-between;">
  <img src="" alt="image" width="58%">
  <img src="" alt="image" width="41%">
</div>
<br><br>

### Optimal Number of Clusters analysis

*Elbow Method & Silhouette Score:*
<br>
<div style="display: flex; justify-content: space-between;">
  <img src="" alt="image" width="58%">
  <img src="" alt="image" width="41%">
</div>
<br><br>


### Results

*Clustered Data Table:*
<br>
<img src="" alt="image" width="700" />
<br><br>

## References:
[_Notes]()

[_Notes]()
