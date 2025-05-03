# Classification Predictor 📊

Welcome to the **Classification Predictor**, an interactive Streamlit application for exploring and comparing classification models. This tool is designed to help users — from beginners to data enthusiasts — build and evaluate machine learning models on custom or demo datasets.

<br>

## ✨ Overview
With this app, you can:
- 📁 Upload your own dataset or choose from built-in demo data
- 🎯 Select the column you want to predict (binary target required)
- 🧠 Compare Decision Tree and Logistic Regression models
- ⚙️ Customize model settings and view performance metrics
- 📊 Visualize results with interactive charts and reports

## 🚀 Instructions 
Click the link to get started in your browser:
[link_to_classification_app](https://steadman-data-science-portfolio-9hqd4epyr8xgguu8xhii5s.streamlit.app/)

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
- Upload a CSV file or select one of the built-in demo datasets (Heart Disease, Loan Approvals).
- Choose the target column (must be binary).

### 🌳 Decision Tree Model
- Customize settings:
    - Splitting criterion: Gini, Entropy, or Log Loss
    - Max depth: Slider (1–30)
- Metrics & Visuals:
    - Accuracy, confusion matrix, ROC curve
    - Classification report
    - Tree diagram visualization
### 📈 Logistic Regression Model
- Customize settings:
    -  Scaled or unscaled data
- Metrics & Visuals:
    - Accuracy, confusion matrix, ROC curve
    - Classification report
    - Coefficient interpretation

🛠 Troubleshooting
- Make sure your dataset is in CSV format
- Ensure the target column is binary (e.g., 0/1, yes/no)

<br>

## 📸 Visuals

### Main Interface

*Main interface with sidebar controls:*

<img src="https://github.com/user-attachments/assets/5cbd5327-43c8-4a62-b3ab-b33731f56aab" alt="Main interface with sidebar controls" width="100%">

### Model Visualization Outputs

*Decision Tree and Confusion Matrix:*

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/96efdfbb-6aa3-405c-bc9c-3608d7ce6aec" alt="Decision Tree" width="58%">
  <img src="https://github.com/user-attachments/assets/24dc10c7-a3ec-4f6c-b32f-1189868ab254" alt="Confusion Matrix" width="41%">
</div>

### ROC Curve

*ROC Curve with AUC Score:*

<img src="https://github.com/user-attachments/assets/8727378c-d731-4b10-9ab2-846731f5e1dd" alt="ROC curve with AUC score" width="100%">

## 📚 References:
[Decision_Tree_Notes](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Notes/Week%2010/IDS_Week_10_1_(3_25)_FINAL.ipynb)

[Logistic_Regression_Notes](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Notes/Week%209/IDS_Week_9_1_(3_18)_FINAL-1.ipynb)
