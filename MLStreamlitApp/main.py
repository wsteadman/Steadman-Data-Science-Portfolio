import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------
# Application Title and Desciption
# -----------------------------------------------

st.set_page_config(
    page_title="Classification Predictor",
    page_icon="📊",
)
st.title("Classification Predictor 🔍")
st.markdown("""
### This appp builds and evaluate classification models on your data with just a few clicks!

### How To Use
* **Select Data**: Upload a CSV file or choose a demo dataset from the sidebar
* **Select Model**: Decision Tree or Logistic Regression
* **Configure Parameters**: Adjust model-specific settings to optimize performance
* **Interpret Results**: Explore visualizations and metrics to understand your model's strengths and limitations

### Understanding Your Model
* The app uses the **last** column (rightmost) in your dataset as the target variable (variable being predicted) and other columns as features

* **Decision Tree**: 
  * Splits the dataset into branches using feature thresholds, based on criteria like Gini impurity, entropy, or log loss

* **Logistic Regression**: 
  * Models the logistic relationship between a dataset's features and the probability of a target variable with binary outcomes (ex: yes/no)
  * To convert probabilities into predictions, a cutoff (typically 0.5) is used with probabilities above the threshold classified as "positives" and below as "negatives"
""")

# -----------------------------------------------
# Helper Functions
# -----------------------------------------------

def format_data(df):
    # Remove rows with missing values
    df.dropna(inplace = True)

   #  filters the DataFrame to only include columns with the data type 'object' then converts these to a list
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    # One-hot encode categorical columns
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Make last column the target and the rest features
    target_col = df.columns[-1]
    features = df.columns[:-1]
    
    # Define features and target
    X = df[features]
    y = df[target_col]
    return df, X, y, features

# Splits data into training and testing split for models
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Trains decision tree model
def train_tree_model(X_train, y_train, criterion='gini', max_depth = 5):
    DT_model = DecisionTreeClassifier(
                            criterion=criterion,
                            max_depth = max_depth
                            )
    DT_model.fit(X_train, y_train)
    return DT_model

# Trains logistic_regression model
def train_logistic_regression_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Generate confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.clf()

# Plot the ROC curve
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess') # Plotting 50% line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)
    plt.clf()


# -----------------------------------------------
# Sidebar layout
# -----------------------------------------------


with st.sidebar: 
      # File uploader
      st.markdown("### 📂 Upload Your Own CSV File")
      upload_file = st.sidebar.file_uploader("Choose a file") 
      
      # Demo sets
      st.markdown("Don't have a dataset? Load a demo")

      demosets = {
        'Heart_Disease_Predictor': "https://raw.githubusercontent.com/wsteadman/Steadman-Data-Science-Portfolio/refs/heads/main/MLStreamlitApp/heart.csv",
        'Loan_Approval': "https://raw.githubusercontent.com/wsteadman/Steadman-Data-Science-Portfolio/refs/heads/main/MLStreamlitApp/loan_data.csv"
        }

    
      demo_selection = st.selectbox('Select a demo dataset', 
                                    ['None'] + list(demosets.keys()), 
                                    key='demo_selection')
      
      # model and parameter selection
      st.divider()
      st.markdown("## ⚙️ Model Settings")

      model_selector = st.radio("",options=["Decision_Tree", "Logistic_regression"])
      st.write("")
      
      # DT parameters
      if model_selector == "Decision_Tree":
          st.markdown("#### Decision Tree Parameters")
          tree_split_criterion = st.radio("Splitting Criterion", options=["gini", "entropy", "log_loss"])
          max_depth = st.slider("Select Max Depth of Tree", min_value=1, max_value=30, value=5)
          st.markdown("*Higher values may lead to overfitting*")

      # Log regression parameters
      elif model_selector == "Logistic_regression":
         scaler_selector = st.radio("Scale_Data", options=["Scaled", "Unscaled"])
          

# -----------------------------------------------
# Main Panel layout
# -----------------------------------------------


df = None

# Handle dataset selection (file upload or demo selection)
if upload_file is not None:
    # Read the uploaded file into a dataframe
    df = pd.read_csv(upload_file)
else:
    if demo_selection != 'None':
        # Load the selected demo dataset
        df = pd.read_csv(demosets[demo_selection])


st.divider()

# Only process if a dataframe is inputted
if df is not None:
    # Display the raw dataset
    with st.expander("Click to view Data Information"):
        st.write("### Overview of your Dataset")
        st.write("#### First 10 Rows of the Dataset")
        st.dataframe(df.head(10))
        st.write("#### Statistical Summary")
        st.dataframe(df.describe())

    # Process the data
    processed_df, X, y, features = format_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)


    # Decision Tree
    if model_selector == "Decision_Tree": 
        # Train Model
        model = train_tree_model(X_train, y_train, criterion=tree_split_criterion, max_depth=max_depth)
        
        # Make predictions based on trained model
        y_pred = model.predict(X_test)
        
      # Calculate accuracy
        col1, col2 = st.columns([1, 3])
        with col1:
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col2:
           st.markdown(f"""
            **Accuracy** measures the percent of correct predictions made by a model:
            - Accuracy of {accuracy:.2%} means the model correctly predicted {int(accuracy * 100)} out of 100 cases
            - ⚠️In imbalanced datasets, accuracy can be misleading. If 95% of samples belong to class A, a model can achieve 95% accuracy by predicting class A for everything!
            """)
        
         # Create a confusion matrix
        col1, col2 = st.columns(2)

        with col1:
            st.write("## Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm)
        with col2:
            st.markdown("##### Understanding the Matrix")
            st.markdown("""
                        A confusion matrix shows how well the model classifies:
                        - **True Positives (top left)**: Correctly predicted positive
                        - **False Positives (bottom left)**: Incorrectly predicted as positive
                        - **False Negatives (top right)**: Incorrectly predicted as negative
                        - **True Negatives (bottom right)**: Correctly predicted negatives
                        """)
            
        # Create Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))  
        with st.expander("Metrics Explanation"):
            st.markdown("""
                        - **Precision**: Measures how many of the predicted positive cases were true positives. High precision: when the model predicts a positive, it’s likely accurate.
                        - **Recall**: Measures how many of the true positive cases the model found. High recall: the model is catching most of the true positives.
                        - **F1-Score**: The harmonic mean of precision and recall, a single score that balances both.
                        - **Support**: The number of true instances for each class in the dataset.
                        """)

        # visualize decision tree
        st.markdown("### Decision Tree")
        dot_data = tree.export_graphviz(model,
                        feature_names=X_train.columns,
                        filled=True,
                        )
        st.graphviz_chart(dot_data)
        st.markdown("""
                    #### This tree is built based on your specified splitting criterion and depth (number of nodes from top to bottom)!
                    
                    Splitting criterion: 
                    - **gini**: The model splits data to minimize information gain based on entropy, ze Gini impurity, or how often a randomly chosen element would be incorrectly classified
                    - **entropy**: The model splits data based on the maximum reduction of disorder achieved in the dataset after each split
                    - **log_loss**: The model optimizes splits by minimizing the error in predicted probabilities for classification
                    """)

        ## ROC Curve  
        # Get the predicted probabilities for the "positive" data 
        y_probs = model.predict_proba(X_test)[:, 1]
        # Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        # Compute the Area Under the Curve (AUC) score
        roc_auc = roc_auc_score(y_test, y_probs)

        # Display Chart and Area Under Curve
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown("## ROC Curve")
        with col2: 
            st.metric("AUC", f"{roc_auc:.2}")
        plot_roc_curve(fpr, tpr, roc_auc)
        
        # Explanation
        with st.expander("ROC Curve and AUC Explanation"):
            st.markdown("""
                        The ROC (Receiver Operating Characteristic) curve is a graphical representation of the True Positive Rate (TPR) against the False Positive Rate (FPR).
                        The AUC (Area Under the Curve) represents the average area under the ROC curve.
                        - The AUC offers a metric to evaluate how well the model distinguishes between the positive and negative results
                        - An AUC of 1 represents a perfect test, while an AUC of 0.5 represents a test no better than random classification
                        """)


    # Logistic Regression
    else:  
        # Scale data (conditional on radio)
        if scaler_selector == "Scaled":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            # Convert the scaled arrays back to DataFrames
            X_train = pd.DataFrame(X_train, columns=features)
            X_test = pd.DataFrame(X_test, columns=features)

        # Train model
        model = train_logistic_regression_model(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate and present accuracy
        col1, col2 = st.columns([1, 3])
        with col1:
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col2:
           st.markdown(f"""
            **Accuracy** measures the percent of correct predictions made by a model:
            - Accuracy of {accuracy:.2%} means the model correctly predicted {int(accuracy * 100)} out of 100 cases
            - ⚠️In imbalanced datasets, accuracy can be misleading. If 95% of samples belong to class A, a model can achieve 95% accuracy by predicting class A for everything!
            """)
        
        # Create a confusion matrix
        col1, col2 = st.columns(2)

        with col1:
            st.write("## Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm)
        with col2:
            st.markdown("##### Understanding the Matrix")
            st.markdown("""
                        A confusion matrix shows how well the model classifies:
                        - **True Positives (top left)**: Correctly predicted positive
                        - **False Positives (bottom left)**: Incorrectly predicted as positive
                        - **False Negatives (top right)**: Incorrectly predicted as negative
                        - **True Negatives (bottom right)**: Correctly predicted negatives
                        """)
        
        # Create Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))   
        with st.expander("Metrics Explanation"):
            st.markdown("""
                        - **Precision**: Measures how many of the predicted positive cases were true positives. High precision: when the model predicts a positive, it’s likely accurate.
                        - **Recall**: Measures how many of the true positive cases the model found. High recall: the model is catching most of the true positives.
                        - **F1-Score**: The harmonic mean of precision and recall, a single score that balances both.
                        - **Support**: The number of true instances for each class in the dataset.
                        """)
            

        # Examine Coefficients
        col1, col2 = st.columns(2)

        with col1: 
            st.markdown("### Model Coefficients")
            coef = pd.Series(model.coef_[0], index=X.columns)
            st.write(coef.sort_values(ascending=False))
        with col2:  
            st.write("")
            st.write("")
            st.write("")
            st.markdown("""
                        - **Unscaled** : Each coefficient represents the change in the outcome probability (log-odds) for a one-unit change in the respective feature, holding all other features constant.
                        
                        - **Scaled** : Scaled coefficients indicate the change in outcome probability (log-odds) for a one standard deviation change in that feature. This standardization makes it easier to compare the relative importance of features.""")


        ## ROC Curve  
        # Get the predicted probabilities for the "positive" data 
        y_probs = model.predict_proba(X_test)[:, 1]
        # Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        # Compute the Area Under the Curve (AUC) score
        roc_auc = roc_auc_score(y_test, y_probs)

        # Display Chart and Area Under Curve
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown("## ROC Curve")
        with col2: 
            st.metric("AUC", f"{roc_auc:.2}")
        plot_roc_curve(fpr, tpr, roc_auc)
        
         # Explanation
        with st.expander("ROC Curve and AUC Explanation"):
            st.markdown("""
                        The ROC (Receiver Operating Characteristic) curve is a graphical representation of the True Positive Rate (TPR) against the False Positive Rate (FPR), relative to different model cutoffs.
                        The AUC (Area Under the Curve) represents the average area under the ROC curve.
                        - The AUC offers a metric to evaluate how well the model distinguishes between the positive and negative results
                        - An AUC of 1 represents a perfect test, while an AUC of 0.5 represents a test no better than random classification
                        """)

    
else:
        # Display welcome message when no data is loaded
        st.markdown("""👋 Welcome to the Classification Predictor!
                    Get started by uploading your dataset or selecting a demo dataset from the sidebar""")