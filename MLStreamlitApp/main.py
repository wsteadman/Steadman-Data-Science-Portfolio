import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



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

def train_model(X_train, y_train):
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

# Only process if we have a dataframe
if df is not None:
    # Display the raw dataset first
    st.subheader("Raw Dataset")
    st.write(df.head())
    # Process the data
    processed_df, X, y, features = format_data(df)
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Train model
    model = train_model(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)


st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm)