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
    pip install -r requirements.txt
    ```
    💡 *Tip: The main packages used are `streamlit` and `pandas`.*

3. **Run the App**

    Use the command below to launch the app in your browser:
    ```
    streamlit run app.py
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

## 📂 Project Structure

penguin-explorer/
│
├── Data/
│ └── penguins.csv # Dataset used in the app
│
├── app.py # Main Streamlit application
├── README.md # You're here!
└── requirements.txt # Dependencies


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

## 📸 Screenshots (Optional)

Include screenshots here if needed for visual documentation.

<br>

## 🙌 Acknowledgements

- Dataset sourced from [Palmer Penguins Dataset](https://github.com/allisonhorst/palmerpenguins)
- Inspired by [Streamlit](https://streamlit.io/) and [Pandas](https://pandas.pydata.org/)

<br>
