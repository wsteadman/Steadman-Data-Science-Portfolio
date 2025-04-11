import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


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
    plt.title('Confusion Matrix')
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
        'Titanic': 'https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv',
    }


df = None

coeffiencents_message = "- Each coefficient represents the change in the Performance Index for a one-unit change " \
                        "in the respective feature, holding all other features constant"

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


st.markdown("### Select Scaled or Unscaled Data")
scaler_selector = st.radio("Scale_Data", options=["Scaled", "Unscaled"])



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

    # Scale data (conditional on radio)
    if scaler_selector == "Scaled":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert the scaled arrays back to DataFrames
        X_train = pd.DataFrame(X_train, columns=features)
        X_test = pd.DataFrame(X_test, columns=features)

        coeffiencents_message = "-Scaled coefficients indicate the change in the Performance Index for a one standard deviation change in that feature." \
                                "- This standardization makes it easier to compare the relative importance of features."


    #Linear Regression
    if model_selector == "Linear_regression": 
        # Train Model
        model = train_linear_regression_model(X_train, y_train)
        
        # Make predictions based on trained model
        y_pred = model.predict(X_test)
        
        # Calculate Model metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display metrics
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            st.metric("Root Mean Squared Error", f"{rmse:.4f}")
        with col3: 
            st.metric("R² Score", f"{r2:.4f}")
        
        st.write("""- MSE: average squared difference between actual and predicted values. """)
        
        st.write("""- RMSE: square root of MSE, gives an error metric in the same units as the target.
                    - Lower RMSE values indicate better predictive performance. """)
        
        st.write("""- R²: indicates the proportion of the variance in the target variable explained by the model.
                    - An R² close to 1 suggests a very good fit, while an R² near 0 indicates the model fails to capture much variance.""")

        # Examine coefficients
        st.write("Model Coefficients")
        st.dataframe(pd.Series(model.coef_, 
                               index=X_train.columns))
        st.write(coeffiencents_message)
        
    # Logistic Regression
    else:  
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
        
         # Examine Coefficients 
        st.subheader("Examine Coefficients")
        coef = pd.Series(model.coef_[0], index=X.columns)
        st.write(coef.sort_values(ascending=False))
        st.write(coeffiencents_message)
    



