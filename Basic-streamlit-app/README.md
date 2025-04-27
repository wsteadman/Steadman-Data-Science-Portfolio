# 🐧 Penguin Explorer App

Welcome to the **Penguin Explorer**, an interactive and data-driven web application built using Streamlit. This app allows users to explore and analyze Palmer Archipelago penguin data through intuitive visual tools.

<br>

## 📊 Overview

This app enables you to:

- 🔍 Filter penguins based on flipper lengths
- 🏝️ Explore penguin populations on different islands
- 📈 View species distribution and average biometric data from each island

Built with simplicity and interactivity in mind, this tool is perfect for educational purposes, data exploration, and quick insights

<br>

## 🚀 Getting Started

1. **Clone the Repository**
    ```
    git clone https://github.com/your-username/penguin-explorer.git
    cd penguin-explorer
    ```

2. **Install Requirements**

    Ensure Python is installed, then run:
    ```
    pip install streamlit
    pip install pandas
    ```

3. **Run the App**

    Use the command below to launch the app in your browser:
    ```
    streamlit run main.py
    ```
    🛠 *Make sure the `penguins.csv` file is located at `Data/penguins.csv`.*

<br>

## 🧭 How to Use the App

### 🐧 Flipper Length Analyzer

- Use the slider to select a flipper length.
- The app displays penguins with flipper lengths within ±5 mm of the selected value.
- View the filtered data directly in an interactive table.

### 🏝️ Island Explorer

- Choose an island from the dropdown menu.
- See the total number of penguins, species distribution, and average metrics for that island.
- Use the checkbox to view all penguin data from the selected island.


<br>

## 🧠 Dataset Info

The dataset contains physical measurements of penguins (bill length, flipper size, etc.) from three islands in the Palmer Archipelago, Antarctica.

| Column             | Description                        |
|--------------------|------------------------------------|
| species            | Penguin species                    |
| island             | Island where the penguin was found |
| bill_length_mm     | Length of the bill                 |
| bill_depth_mm      | Depth of the bill                  |
| flipper_length_mm  | Flipper length in millimeters      |
| body_mass_g        | Body mass in grams                 |
| sex                | Sex of the penguin                 |

<br>

## 📸 Visuals

*Flipper Length Analyzer:*

<img src="https://github.com/user-attachments/assets/b61d2593-8413-4ecd-8e76-ae151245e346" alt="Flipper Length Analyzer" width="500">

*Island Explorer:*

<img src="https://github.com/user-attachments/assets/d44cc8fa-89bb-4ddf-a645-2f325c915640" alt="Island Explorer" width="500">

<br>

## Sources

- Dataset sourced from [Palmer Penguins Dataset](https://github.com/allisonhorst/palmerpenguins)
- [Week_3_Notes](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Notes/Week%203/Week_3_2_streamlit_IN-CLASS.py)
- [Week_4_Notes](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Notes/Week%204/Week_4_2_streamlit_data_IN-CLASS.py)

<br>

