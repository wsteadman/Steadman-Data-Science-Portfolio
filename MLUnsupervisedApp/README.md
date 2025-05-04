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

### 📈 Running the Model
- 2D PCA Projection: Displays the data in two dimensions using PCA to show how well clusters are formed and separated
- Elbow Method Graph: Shows within-cluster sum of squares (WCSS) dropping for k values. The "elbow" (value after steepest drow-off) suggests best number of clusters
- Silhouette Score Graph: Measures how similar each point is to its own cluster vs. other clusters. Higher scores indicate better k values

### 💡 Results Analysis
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
  <img src="https://github.com/user-attachments/assets/d2b59fd0-463c-4718-9507-2e0fe3fd2415" alt="image" width="55%">
  <img src="https://github.com/user-attachments/assets/31e1fa72-d600-46d6-a7b3-ec206fe85ba4" alt="image" width="43%">
</div>
<br><br>

### Optimal Number of Clusters analysis

*Elbow Method & Silhouette Score:*
<br>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/cff9c5d9-1ebc-41ef-8d76-e8c530bc8bb8" alt="image" width="48.5%">
  <img src="https://github.com/user-attachments/assets/c9e6c104-20af-42c6-9239-3e1da6f3a9e5" alt="image" width="47%">
</div>
<br><br>


### Results

*Clustered Data Table:*
<br>
<img src="https://github.com/user-attachments/assets/0e1f0fcb-32c8-4ad2-b4b5-fdcd01ad75cc" alt="image" width="700" />
<br><br>

## References:
[Kmeans_notes](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Notes/Week%2013/IDS_13_1_(4_15)_FINAL.ipynb)

[KMeans article](https://www.geeksforgeeks.org/k-means-clustering-introduction/)
