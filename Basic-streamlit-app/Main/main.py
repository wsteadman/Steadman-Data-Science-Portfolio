import streamlit as st
import pandas as pd

st.title("This is an app to analyze some penguins!")

## Importing the Penguins Df using relative path
df = pd.read_csv('Data\sample_data.csv')

st.dataframe(df)
## diplaying penguins df
