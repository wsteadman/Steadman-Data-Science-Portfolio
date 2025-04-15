# Classification Predictor 📊

## Project Overview 🚀
This is an interactive data science application that allows users to build, evaluate, and visualize machine learning classification models. Users have the option to upload their own datasets or choose from demo options, and then compare the performance of both Decision Tree and Logistic Regression models with customizable configurations. The app creates visualizations and metrics to help users understand how the models function, and does so in a way that allows anyone with no prior knowledge of classification models to understand the outputs!

## Instructions 🛠️
#### Click the link to get started!
[link_to_classification_app](https://steadman-data-science-portfolio-9hqd4epyr8xgguu8xhii5s.streamlit.app/)

**Using the App** 
- Upload or choose a demoset dataset
- Adjust model parameters using the sliders and checkboxes in the sidebar to see how they affect performance
- Compare both models to determine which works best for your specific dataset

**Troubleshooting**
- If you have errors, ensure your data is in CSV format
- Ensure that the target variable is binary and located in the last column
- If models perform poorly, try adjusting parameters or consider if your dataset needs more preprocessing

## Required Libraries 📚
- graphviz==0.20.3
- matplotlib==3.10.1
- pandas==2.2.3
- scikit_learn==1.6.1
- seaborn==0.13.2
- streamlit==1.37.1

## App Features 📁 
*for more info about any metric or visual just visit the app!*
### Data Processing
- Data Importing: Upload custom CSV files or use built-in demo datasets (Heart Disease, loan approvals)
- Preprocessing data: Automatically handles missing values and dummy variables

### Models
**Decision Tree** 🌳
- Configurable Parameters:
    - Splitting criterion: Choose between Gini impurity, entropy, or log loss 
    - Maximum tree depth: Control node depth with a slider (range: 1-30)

- Visualizations & Metrics:
    - Accuracy score
    - Confusion matrix heatmap
    - Classification report (precision, recall, F1-score)
    - Decision tree visualization
    - ROC curve with AUC value

**Logistic Regression** 📈
- Configurable Parameters:
    - Data scaling: Option to scale data 

- Visualizations & Metrics:
    - Accuracy score
    - Confusion matrix heatmap
    - Classification report (precision, recall, F1-score)
    - Coefficient analysis
    - ROC curve with AUC value

## Visual Examples
<br>

*Main interface with sidebar controls:*
<br>
<img src="https://github.com/user-attachments/assets/eb62906d-2400-4e39-bfe0-7c954a186af3" alt="image" width="700" />
<br><br>

*Decision Tree Visualization:*
<br>
<img src="https://github.com/user-attachments/assets/96efdfbb-6aa3-405c-bc9c-3608d7ce6aec" alt="image" width="700" />
<br><br>

*ROC curve with AUC score and confusion matrix heatmap:*
<br>
<img src="https://github.com/user-attachments/assets/8727378c-d731-4b10-9ab2-846731f5e1dd" alt="image" width="700" />
<br><br>

## References:
[Decision_Tree_Notes](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Notes/Week%2010/IDS_Week_10_1_(3_25)_FINAL.ipynb)

[Logistic_Regression_Notes](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Notes/Week%209/IDS_Week_9_1_(3_18)_FINAL-1.ipynb)
