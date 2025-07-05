import pandas as pd

# Set option to display all rows
pd.set_option('display.max_rows', None)

# Step 1: Load the dataset
file_path = "FAOSTAT_data - FAOSTAT_data_en_12-29-2024.csv"  # Update if necessary
df = pd.read_csv(file_path)

# Step 2: Drop irrelevant columns
df_cleaned = df.drop(columns=[
    'Domain Code', 'Domain', 'Area Code (M49)', 'Element Code',
    'Item Code (CPC)', 'Year Code', 'Flag', 'Flag Description', 'Note'
])

# Step 3: Pivot the table
df_pivot = df_cleaned.pivot_table(
    index=['Area', 'Item', 'Year'],
    columns='Element',
    values='Value'
).reset_index()

# Step 4: Rename important columns
df_pivot.columns.name = None
df_pivot = df_pivot.rename(columns={
    'Area harvested': 'Area_Harvested_ha',
    'Yield': 'Yield_kg_ha',
    'Production': 'Production_ton'
})

# Step 5: Drop rows with missing values in key columns
df_final = df_pivot.dropna(subset=['Area_Harvested_ha', 'Yield_kg_ha', 'Production_ton'])

# Step 6: Show all rows
print("Cleaned dataset shape:", df_final.shape)
print(df_final)  # This will show all rows
