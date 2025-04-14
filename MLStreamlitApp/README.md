# Classification Predictor 📊

## Project Overview 🚀
This is an interactive data science application that allows users to build, evaluate, and visualize machine learning classification models. Users have the option to upload their own datasets or choose from demo options, and then compare the performance of both Decision Tree and Logistic Regression models with customizable configurations. The app creates visualizations and metrics to help users understand how the models function, and do so in a way that allows anyone with no prior knowledge of classification models to understand the outputs. 

## Instructions 🛠️
[link to app]( )

## Required Libraries 📚
- streamlit==1.27.0
- pandas==2.0.3
- seaborn==0.12.2
- matplotlib==3.7.2
- scikit-learn==1.3.0
- graphviz==0.20.1

## App Features 📁
### Data Processing
- Data Importing: Upload custom CSV files or use built-in demo datasets (Titanic, Heart Disease)
- Preprocessing data: Automatically handles missing values and dummy variables

### Models
Decision Tree 🌳
- Configurable Parameters:
    - Splitting criterion: Choose between Gini impurity, entropy, or log loss 
    - Maximum tree depth: Control node depth with a slider (range: 1-30)

- Visualizations & Metrics:
    - Accuracy score
    - Confusion matrix heatmap
    - Classification report (precision, recall, F1-score)
    - Decision tree visualization
    - ROC curve with AUC value

Logistic Regression 📈
- Configurable Parameters:
    - Data scaling: Option to scale data 

- Visualizations & Metrics:
    - Accuracy score
    - Confusion matrix heatmap
    - Classification report (precision, recall, F1-score)
    - Coefficient analysis
    - ROC curve with AUC value

## Visual Examples
App Interface:


Main interface with sidebar controls and model:


Decision Tree Visualization:


ROC curve with AUC score and confusion matrix heatmap:


## References:
[Decision_Tree_Notes](Notes\Week 10\IDS_Week_10_1_(3_25)_FINAL.ipynb)

[Logistic_Regression_Notes](Notes\Week 9\IDS_Week_9_1_(3_18)_FINAL-1.ipynb)
