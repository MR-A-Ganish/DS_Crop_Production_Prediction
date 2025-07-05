import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("FAOSTAT_data - FAOSTAT_data_en_12-29-2024.csv")

# Drop unnecessary columns
df_cleaned = df.drop(columns=[
    'Domain Code', 'Domain', 'Area Code (M49)', 'Element Code',
    'Item Code (CPC)', 'Year Code', 'Flag', 'Flag Description', 'Note'
])

# Pivot so that Area harvested, Yield, and Production become columns
df_pivot = df_cleaned.pivot_table(
    index=['Area', 'Item', 'Year'],
    columns='Element',
    values='Value'
).reset_index()

# Clean column names
df_pivot.columns.name = None
df_pivot = df_pivot.rename(columns={
    'Area harvested': 'Area_Harvested_ha',
    'Yield': 'Yield_kg_ha',
    'Production': 'Production_ton'
})

# Drop rows with missing values in any of the 3 main columns
df_eda = df_pivot.dropna(subset=['Area_Harvested_ha', 'Yield_kg_ha', 'Production_ton'])

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# -----------------------------
# Plot 1: Distribution of Production
plt.figure()
sns.histplot(df_eda['Production_ton'], bins=50, kde=True)
plt.title('Distribution of Crop Production (tons)')
plt.xlabel('Production (tons)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2: Yearly Trends of Total Production
yearly_prod = df_eda.groupby('Year')['Production_ton'].sum().reset_index()
plt.figure()
sns.lineplot(data=yearly_prod, x='Year', y='Production_ton', marker='o')
plt.title('Total Crop Production Over the Years')
plt.ylabel('Total Production (tons)')
plt.xlabel('Year')
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 3: Top 10 Crops by Total Production
top_crops = df_eda.groupby('Item')['Production_ton'].sum().sort_values(ascending=False).head(10).reset_index()
plt.figure()
sns.barplot(data=top_crops, x='Production_ton', y='Item', palette='viridis')
plt.title('Top 10 Crops by Total Production')
plt.xlabel('Total Production (tons)')
plt.ylabel('Crop')
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 4: Top 10 Regions by Total Production
top_regions = df_eda.groupby('Area')['Production_ton'].sum().sort_values(ascending=False).head(10).reset_index()
plt.figure()
sns.barplot(data=top_regions, x='Production_ton', y='Area', palette='mako')
plt.title('Top 10 Regions by Total Crop Production')
plt.xlabel('Total Production (tons)')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 5: Correlation Matrix
plt.figure()
corr = df_eda[['Area_Harvested_ha', 'Yield_kg_ha', 'Production_ton']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Area, Yield, Production')
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 6: Outlier Detection with Boxplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(data=df_eda, y='Area_Harvested_ha', ax=axes[0])
axes[0].set_title('Outliers in Area Harvested')

sns.boxplot(data=df_eda, y='Yield_kg_ha', ax=axes[1])
axes[1].set_title('Outliers in Yield')

sns.boxplot(data=df_eda, y='Production_ton', ax=axes[2])
axes[2].set_title('Outliers in Production')

plt.tight_layout()
plt.show()
