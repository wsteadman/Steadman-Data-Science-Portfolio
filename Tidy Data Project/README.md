# Project: Cleaning and Visualizing 2008 Olympics Medal Data 

## ✨Overview

This project demonstrates the process of cleaning a "flawed" or "untidy" dataset and performing exploratory data analysis using Python's Pandas library. The primary goal is to transform the data into a tidy format better suited for analysis and then generate visualizations to explore the data.

**Link to the full Jupyter Notebook:** [Olympians.ipynb](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Tidy%20Data%20Project/Olympians.ipynb)

## 🏅Dataset

* **Source:** CSV file provided by a professor (`olympics_08_medalists.csv`).
* **Content:** Lists medalists from the 2008 Olympic Games.
* **Challenge:** The original dataset is structured in a "wide" format where ever athlete forms a row and is cross-referenced with every sport/event column, making analysis difficult. This format causes numerous empty cells. The sex of the atheltes and their events are also combined, even though these features represent distinct data points.
* **Link to flawed data:** [olympics_08_medalists.csv](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Tidy%20Data%20Project/olympics_08_medalists.csv)

## 🔍Process

The analysis follows these key steps implemented in the Jupyter Notebook:

1.  **Import Data:** Load the initial CSV file into a Pandas DataFrame.
2.  **Data Tidying (Reshaping):**
    * Use the `pandas.melt` function to transform the DataFrame from a wide format to a long format. This consolidates the multiple sport/event columns into two new columns: one for the sport/event category and one for the medal type (Gold, Silver, Bronze).
    * Remove rows with missing medal values (`NaN`)
    * Split the combined sport/event column into separate 'Sex' and 'Sport' columns.
    * Drop the original combined column.
3.  **Data Visualization:**
    * Create bar charts to visualize:
        * Total count of each medal type (Gold, Silver, Bronze)
        * Total count of medals won by male vs. female athletes
        * A horizontal bar chart showing the number of Gold medals per sport
        * Generate a pivot table to summarize the number of medals (Gold, Silver, Bronze) won by athletes, broken down by both Sport and Sex.

## 📋 Instructions
* Clone this repository
* Pip install pandas
* Open the `Olympians.ipynb` file in Jupyter Notebook or JupyterLab
* Run the cells sequentially from top to bottom to execute the data loading, cleaning, visualization, and analysis steps!

## 📈 Visualizations & Results Examples

### Data Transformation (Original vs. Tidied)

*Original Wide Format:*

<img src="https://github.com/user-attachments/assets/eacd4a39-ef69-4378-8ff0-cb9e3183aca6" alt="Original wide format dataframe" width="600">


*Tidied Long Format:*

<img src="https://github.com/user-attachments/assets/6c691210-c615-4a67-9c74-1fb6301bfb34" alt="Tidied long format dataframe" width="300">

### Example Charts

*Medal Count by Type:*

<img src="https://github.com/user-attachments/assets/f2658c6f-0794-43d4-bc1f-fae18af8bbea" alt="Bar chart showing counts of Gold, Silver, and Bronze medals" width="800">


*Gold Medals by Sport:*

<img src= "https://github.com/user-attachments/assets/3ef6445e-0f2f-4685-ac4d-085c853688de" alt="Horizontal bar chart showing Gold medals per sport" width="800">


### Pivot Table Summary

*Medal Counts by Sport and Sex:*

<img src="https://github.com/user-attachments/assets/fb0ca541-df3c-498c-8ce7-782b429f7dd5" alt="Pivot table showing medal counts broken down by sport and sex" width="700">

## 📚 References

* Internal Course Notes:
    * [Data_Tidy_Notes_1](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Notes/Week%206/IDS_6_1_FINAL.ipynb)
    * [Data_Tidy_Notes_2](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Notes/Week%206/IDS_6_2_FINAL.ipynb)
* Pandas Documentation:
    * [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
