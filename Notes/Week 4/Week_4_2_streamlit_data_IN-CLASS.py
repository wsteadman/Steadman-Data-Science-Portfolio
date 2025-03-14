import streamlit as st
import pandas as pd

# ================================
# Step 1: Displaying a Simple DataFrame in Streamlit
# ================================

st.subheader("Now, let's look at some data!")

# Creating a simple DataFrame manually
# This helps students understand how to display tabular data in Streamlit.
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
})

# Displaying the table in Streamlit
# st.dataframe() makes it interactive (sortable, scrollable)
st.write("Here's a simple table:")
st.dataframe(df)

# ================================
# Step 2: Adding User Interaction with Widgets
# ================================
city = st.selectbox('Select a City', df['City'].unique())
# .unique() ensures if there were repeats they wouldnt be included twice

# Using a selectbox to allow users to filter data by city

# Filtering the DataFrame based on user selection
city_filtered_df = df[df['City'] == city]
# Display the filtered results
st.write(f'People who live in {city}:')
st.dataframe(city_filtered_df)
# ================================  
# Step 3: Importing Data Using a Relative Path
# ================================

# Now, instead of creating a DataFrame manually, we load a CSV file
# This teaches students how to work with external data in Streamlit
# # Ensure the "data" folder exists with the CSV file
# Display the imported dataset

df2 = pd.read_csv('Data\sample_data.csv')
st.dataframe(df2)
# copy its 'relative path'


# Using a selectbox to allow users to filter data by city
# Students learn how to use widgets in Streamlit for interactivity

# Filtering the DataFrame based on user selection
salary = st.slider('Select a Salary range', min_value = df2['Salary'].min(), max_value = df2['Salary'].max())
# Display the filtered results
st.dataframe(df2[df2['Salary'] <= salary])
# ================================
# Summary of Learning Progression:
# 1️⃣ Displaying a basic DataFrame in Streamlit.
# 2️⃣ Adding user interaction with selectbox widgets.
# 3️⃣ Importing real-world datasets using a relative path.
# ================================