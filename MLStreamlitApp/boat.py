import pandas as pd

# Load the original dataset
df = pd.read_csv('titanic_.csv')  # Make sure this path matches your file location

# Print original shape
print(f"Original dataset shape: {df.shape}")

# Remove the 'boat' column
if 'boat' in df.columns:
    df = df.drop(columns=['boat'])
    print("'boat' column removed successfully")
else:
    print("'boat' column not found in the dataset")

# Print new shape
print(f"New dataset shape: {df.shape}")

# Save the updated dataset
df.to_csv('titanic_no_boat.csv', index=False)
print("Updated dataset saved as 'titanic_no_boat.csv'")

# Display the first few rows to verify
print("\nFirst 5 rows of the updated dataset:")
print(df.head())