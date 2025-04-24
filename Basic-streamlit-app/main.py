import streamlit as st
import pandas as pd

## Use 'Streamlit run' in the terminal to open up the Streamlit link

st.title("This is an app to analyze some penguins!")
st.markdown('*This app filters penguin data based on their flipper size and home island!*')

st.markdown('---')

## Importing the Penguins Df using relative path
df = pd.read_csv('Data\penguins.csv')

## Adding an interactive island explorer widget
st.header("🐧 Interactive Flipper Length Analyzer")

## Creating a slider to display all penguins with similar flipper length
flip_length = st.slider('Select a flipper length', min_value = df['flipper_length_mm'].min(), max_value = df['flipper_length_mm'].max())

# Define a range for "similar" (within 5mm of the selected value)
similar_range = 5
df_similar = df[(df['flipper_length_mm'] >= flip_length - similar_range) & 
                (df['flipper_length_mm'] <= flip_length + similar_range)]

st.write(f"Showing penguins with flipper length between {flip_length - similar_range} and {flip_length + similar_range} mm:")
st.dataframe(df_similar)

st.markdown('---')

## Adding an interactive island explorer widget
st.header("🏝️ Interactive Island Explorer")

# Get list of islands
islands = sorted(df['island'].unique())

# Create a selection widget for islands
selected_island = st.selectbox("Select an island to explore:", islands)

# Filter data for the selected island
island_data = df[df['island'] == selected_island]

# Display basic island stats
st.write(f"**{selected_island} Island Stats:**")
st.write(f"Total penguins: {len(island_data)}")

# Display species distribution on the island
species_on_island = island_data['species'].value_counts()
st.write("Species distribution:")
st.write(species_on_island)

# Show average metrics for the selected island
st.write("**Average measurements for penguins on this island:**")
avg_metrics = island_data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].mean().round(2)
st.write(f"- Bill length: {avg_metrics['bill_length_mm']} mm")
st.write(f"- Bill depth: {avg_metrics['bill_depth_mm']} mm")
st.write(f"- Flipper length: {avg_metrics['flipper_length_mm']} mm")
st.write(f"- Body mass: {avg_metrics['body_mass_g']} g")

# Add a checkbox to show all data for this island
if st.checkbox(f"Show all penguin data from {selected_island}"):
    st.dataframe(island_data)