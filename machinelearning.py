import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------------------------
# Step 1: Load and clean the dataset
df = pd.read_csv("FAOSTAT_data - FAOSTAT_data_en_12-29-2024.csv")

df_cleaned = df.drop(columns=[
    'Domain Code', 'Domain', 'Area Code (M49)', 'Element Code',
    'Item Code (CPC)', 'Year Code', 'Flag', 'Flag Description', 'Note'
])

df_pivot = df_cleaned.pivot_table(
    index=['Area', 'Item', 'Year'],
    columns='Element',
    values='Value'
).reset_index()

df_pivot.columns.name = None
df_pivot = df_pivot.rename(columns={
    'Area harvested': 'Area_Harvested_ha',
    'Yield': 'Yield_kg_ha',
    'Production': 'Production_ton'
})

df_eda = df_pivot.dropna(subset=['Area_Harvested_ha', 'Yield_kg_ha', 'Production_ton'])

# ----------------------------
# Step 2: Prepare features and target
X = df_eda[['Area', 'Item', 'Year', 'Area_Harvested_ha', 'Yield_kg_ha']]
y = df_eda['Production_ton']

# ----------------------------
# Step 3: Preprocessing pipeline
categorical_features = ['Area', 'Item']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# ----------------------------
# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Step 5: Define and train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    results.append({
        'Model': name,
        'R2 Score': round(r2, 4),
        'MAE': round(mae, 2),
        'MSE': round(mse, 2)
    })

# ----------------------------
# Step 6: Show evaluation results
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:\n")
print(results_df.sort_values(by='R2 Score', ascending=False))

