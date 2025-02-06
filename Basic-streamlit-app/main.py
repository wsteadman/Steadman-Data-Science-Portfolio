import streamlit as st
import pandas as pd

st.title("This is an app to analyze some penguins!")

## Importing the Penguins Df using relative path
df = pd.read_csv('Data\penguins.csv')

## Creating a slider to display all penguins with less than or equal to flipper length
flip_length = st.slider('Select a flipper range', min_value = df['flipper_length_mm'].min(), max_value = df['flipper_length_mm'].max())
st.dataframe(df[df['flipper_length_mm'] <= flip_length])


## Creating a button that will return the penguins with the exact flipper length of the slider
if st.button('Click me for exact flipper length'):
    df_2 = df[df['flipper_length_mm'] == flip_length]

    if not df_2.empty:
        st.dataframe(df_2)
    else:
        st.write('No penguin has that exact flipper length')
