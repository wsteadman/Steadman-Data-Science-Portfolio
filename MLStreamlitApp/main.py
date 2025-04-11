import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# -----------------------------------------------
# Application Information
# -----------------------------------------------
st.set_page_config(
    page_title="Classification Predictor",
    page_icon="📊",
)
st.title("Classification Predictor")
st.markdown("""This is my app!""")

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


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_tree_model(X_train, y_train, criterion='gini'):
    # Create model with the user-specified criterion
    DT_model = DecisionTreeClassifier(criterion=criterion)

    # Train the model
    DT_model.fit(X_train, y_train)
    
    return DT_model

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
    plt.figure(figsize=(8, 6))
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
      # File upload
      st.markdown("### 📂 Upload Your Own CSV File")
      upload_file = st.sidebar.file_uploader("Choose a file") 
      
      # Demo set options
      st.markdown("Don't have a dataset? Load a demo")
      
      demosets = {
        'Titanic': 'https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv',
        ## other data sets?
        }
    
      demo_selection = st.selectbox('Select a demo dataset', 
                                    ['None'] + list(demosets.keys()), 
                                    key='demo_selection')

      st.markdown("### ⚙️ Model Settings")

      # model selection
      model_selector = st.radio("Model", options=["Decision_Tree", "Logistic_regression"])
    
      if model_selector == "Decision_Tree":
          st.markdown("### Decision Tree Parameters")
          tree_split_criterion = st.radio("Splitting Criterion", options=["gini", "entropy", "log_loss"])
      
      elif model_selector == "Logistic_regression":
         # Data scaled or unscaled 
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


    #Decision Tree
    if model_selector == "Decision_Tree": 
        # Train Model
        model = train_tree_model(X_train, y_train, criterion=tree_split_criterion)
        
        # Make predictions based on trained model
        y_pred = model.predict(X_test)
        
      # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Model Performance")
        st.metric("Accuracy", f"{accuracy:.2%}")
        
        # Create a confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm)
        
        # Create Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))  

        # visualize decision tree
        st.markdown("### Decision Tree")
        dot_data = tree.export_graphviz(model,
                        feature_names=X_train.columns,
                        filled=True,
                        )
        st.graphviz_chart(dot_data)
        # ROC Curve 
        st.markdown("### ROC Curve")
            # Get the predicted probabilities for the positive class, only the second column of the array 
        y_probs = model.predict_proba(X_test)[:, 1]
            # Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
            # Compute the Area Under the Curve (AUC) score
        roc_auc = roc_auc_score(y_test, y_probs)
        plot_roc_curve(fpr, tpr, roc_auc)



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
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Model Performance")
        st.metric("Accuracy", f"{accuracy:.2%}")
        
        # Create a confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm)
        
        # Create Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))   
        

        col1, col2 = st.columns(2)

        # Examine Coefficients
        with col1: 
            st.markdown("### Model Coefficients")
            coef = pd.Series(model.coef_[0], index=X.columns)
            st.write(coef.sort_values(ascending=False))
            st.write({"Unscaled": "Each coefficient represents the change in the outcome probability (log-odds) for a one-unit change in the respective feature, holding all other features constant.", 
                      "Scaled": "Scaled coefficients indicate the change in outcome probability (log-odds) for a one standard deviation change in that feature. This standardization makes it easier to compare the relative importance of features."})

        # ROC Curve
        with col2:    
            st.markdown("### ROC Curve")
            # Get the predicted probabilities for the positive class, only the second column of the array 
            y_probs = model.predict_proba(X_test)[:, 1]
            # Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
            fpr, tpr, thresholds = roc_curve(y_test, y_probs)
            # Compute the Area Under the Curve (AUC) score
            roc_auc = roc_auc_score(y_test, y_probs)
            plot_roc_curve(fpr, tpr, roc_auc)
    
else:
        # Display welcome message when no data is loaded
        st.markdown("""👋 Welcome to the Classification Predictor!
                    Get started by uploading your dataset or selecting a demo dataset from the sidebar""")