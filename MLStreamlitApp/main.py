import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score



# -----------------------------------------------
# Application Information
# -----------------------------------------------

st.title("Logistic Regression Visualiser")
st.markdown("""Description""")


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


def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_logistic_regression_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Generate confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('COnfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.clf()

# -----------------------------------------------
# Streamlit App Layout
# -----------------------------------------------


upload_file = st.sidebar.file_uploader("Choose a file") 


st.header("Don't have a dataset? Load a demo")

demosets = {
        'Tennis': 'https://raw.githubusercontent.com/JadeMaveric/DecisionTreeViz/main/data/tennis.csv',
        'Cars': 'https://raw.githubusercontent.com/JadeMaveric/DecisionTreeViz/main/data/cars.csv',
        'Customers': 'https://raw.githubusercontent.com/JadeMaveric/DecisionTreeViz/main/data/customers.csv'
    }


df = None

# Handle dataset selection (file upload or demo selection)
if upload_file is not None:
    # Read the uploaded file into a dataframe
    df = pd.read_csv(upload_file)
else:
    demo_selection = st.selectbox('Select a demo dataset', ['None'] + list(demosets.keys()), key='demo_selection')
    
    if demo_selection != 'None':
        # Load the selected demo dataset
        df = pd.read_csv(demosets[demo_selection])

st.markdown("### Select Model")
model_selector = st.radio("Model", options=["Linear_regression", "Logistic_regression"])

# Only process if we have a dataframe
if df is not None:
    # Display the raw dataset first
    st.subheader("Raw Dataset")
    st.write(df.head())
    # Process the data
    processed_df, X, y, features = format_data(df)
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    if model_selector == "Linear_regression":
        model = train_linear_regression_model(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Model metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            st.metric("R² Score", f"{r2:.4f}")
        with col3: 
            st.metric("Root Mean Squared Error", f"{rmse:.4f}")
    
        
        # Examine coefficients
        st.subheader("Model Coefficients")
        coef = pd.Series(model.coef_, index=X.columns)
        intercept = model.intercept_
        
        # Create a bar chart for coefficients
        plt.figure(figsize=(10, 6))
        coef.sort_values().plot(kind='barh')
        plt.title('Feature Coefficients')
        plt.xlabel('Coefficient Value')
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()



    else:  # Logistic Regression
        model = train_logistic_regression_model(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Model accuracy
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
        
         # Examine coefficients 
        st.subheader("Examine Coefficients")
        coef = pd.Series(model.coef_[0], index=X.columns)
        st.write(coef.sort_values(ascending=False))




# is negative good??
