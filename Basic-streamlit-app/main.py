import streamlit as st
import pandas as pd

st.title("This is an app to analyze some penguins!")

## Importing the Penguins Df using relative path
df = pd.read_csv('Data\penguins.csv')

st.dataframe(df)
## diplaying penguins df

flip_length = st.slider('Select a flipper range', min_value = df['flipper_length_mm'].min(), max_value = df['flipper_length_mm'].max())

st.dataframe(df[df['flipper_length_mm'] <= flip_length])