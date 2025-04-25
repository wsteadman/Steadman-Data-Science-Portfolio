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

<img src="image-2.png" alt="Original wide format dataframe" width="600">

*Tidied Long Format:*

<img src="image-3.png" alt="Tidied long format dataframe" width="300">

### Example Charts

*Medal Count by Type:*

<img src="https://github.com/user-attachments/assets/f3cb906e-f7cd-4be6-9cbb-5315cf9d3e85" alt="Bar chart showing counts of Gold, Silver, and Bronze medals" width="800">

*Gold Medals by Sport:*

<img src="image-1.png" alt="Horizontal bar chart showing Gold medals per sport" width="500">

### Pivot Table Summary

*Medal Counts by Sport and Sex:*

<img src="image-4.png" alt="Pivot table showing medal counts broken down by sport and sex" width="500">

## 📚 References

* Internal Course Notes:
    * [Data_Tidy_Notes_1](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Notes/Week%206/IDS_6_1_FINAL.ipynb)
    * [Data_Tidy_Notes_2](https://github.com/wsteadman/Steadman-Data-Science-Portfolio/blob/main/Notes/Week%206/IDS_6_2_FINAL.ipynb)
* Pandas Documentation:
    * [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
