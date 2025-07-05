import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import base64

st.set_page_config(page_title="Crop Production Predictor", layout="wide")
st.title("🌾 Crop Production Predictor (FAOSTAT)")

# Load dataset
try:
    df = pd.read_csv("FAOSTAT_data - FAOSTAT_data_en_12-29-2024.csv")
    st.success("✅ Dataset loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# Clean and prepare full version for display
try:
    keep_cols = [
        'Area', 'Item', 'Year', 'Year Code', 'Element', 'Element Code',
        'Unit', 'Value', 'Flag', 'Flag Description', 'Note'
    ]
    df = df[[col for col in keep_cols if col in df.columns]]

    # Create ML dataset
    df_ml = df.pivot_table(
        index=['Area', 'Item', 'Year'],
        columns='Element',
        values='Value'
    ).reset_index()

    df_ml.columns.name = None
    df_ml = df_ml.rename(columns={
        'Area harvested': 'Area_Harvested_ha',
        'Yield': 'Yield_kg_ha',
        'Production': 'Production_ton'
    })

    df_ml = df_ml.dropna(subset=['Area_Harvested_ha', 'Yield_kg_ha', 'Production_ton'])

except Exception as e:
    st.error(f"❌ Preprocessing error: {e}")
    st.stop()

# All Filters
st.subheader("🔎 Full Data Filters")
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
col7, col8 = st.columns(2)

with col1:
    selected_areas = st.multiselect("🌍 Region (Area)", sorted(df['Area'].unique()), default=None)
with col2:
    selected_items = st.multiselect("🌾 Crop (Item)", sorted(df['Item'].unique()), default=None)
with col3:
    selected_years = st.multiselect("📅 Year", sorted(df['Year'].unique()), default=None)
with col4:
    selected_year_codes = st.multiselect("🆔 Year Code", sorted(df['Year Code'].unique()), default=None)
with col5:
    selected_units = st.multiselect("📏 Unit", sorted(df['Unit'].dropna().unique()), default=None)
with col6:
    selected_values = st.slider("📊 Value Range", float(df['Value'].min()), float(df['Value'].max()), (float(df['Value'].min()), float(df['Value'].max())))
with col7:
    selected_element_codes = st.multiselect("🔢 Element Code", sorted(df['Element Code'].unique()), default=None)
with col8:
    selected_elements = st.multiselect("🔠 Element", sorted(df['Element'].dropna().unique()), default=None)

# Apply filters
df_filtered = df.copy()
if selected_areas:
    df_filtered = df_filtered[df_filtered['Area'].isin(selected_areas)]
if selected_items:
    df_filtered = df_filtered[df_filtered['Item'].isin(selected_items)]
if selected_years:
    df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]
if selected_year_codes:
    df_filtered = df_filtered[df_filtered['Year Code'].isin(selected_year_codes)]
if selected_units:
    df_filtered = df_filtered[df_filtered['Unit'].isin(selected_units)]
if selected_element_codes:
    df_filtered = df_filtered[df_filtered['Element Code'].isin(selected_element_codes)]
if selected_elements:
    df_filtered = df_filtered[df_filtered['Element'].isin(selected_elements)]
df_filtered = df_filtered[(df_filtered['Value'] >= selected_values[0]) & (df_filtered['Value'] <= selected_values[1])]

st.subheader("📋 Filtered Data with All Columns")
if not df_filtered.empty:
    st.dataframe(df_filtered)
else:
    st.warning("⚠️ No data matches the filters.")
    st.stop()

# ML model using df_ml
X = df_ml[['Area', 'Item', 'Year', 'Area_Harvested_ha', 'Yield_kg_ha']]
y = df_ml['Production_ton']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'Item'])
], remainder='passthrough')

model = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X, y)

# Evaluate model
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)

st.subheader("📊 Model Evaluation Metrics (on Test Set)")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MAE", f"{mae:,.2f}")
col3.metric("MSE", f"{mse:,.2f}")

# Prediction Section
st.subheader("🎯 Predict Crop Production")
pred_col1, pred_col2, pred_col3 = st.columns(3)
with pred_col1:
    selected_area = st.selectbox("🌍 Region", sorted(df_ml['Area'].unique()))
with pred_col2:
    selected_item = st.selectbox("🌾 Crop", sorted(df_ml[df_ml['Area'] == selected_area]['Item'].unique()))
with pred_col3:
    selected_year = st.selectbox("📅 Year", sorted(df_ml['Year'].unique()))

area_harvested = st.number_input("📐 Area Harvested (ha)", value=float(df_ml['Area_Harvested_ha'].mean()))
yield_kg_ha = st.number_input("📈 Yield (kg/ha)", value=float(df_ml['Yield_kg_ha'].mean()))

if st.button("🚀 Predict Production"):
    input_df = pd.DataFrame([{
        'Area': selected_area,
        'Item': selected_item,
        'Year': selected_year,
        'Area_Harvested_ha': area_harvested,
        'Yield_kg_ha': yield_kg_ha
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"📦 Predicted Production: **{prediction:,.2f} tons**")

    history = df_ml[(df_ml['Area'] == selected_area) & (df_ml['Item'] == selected_item)].sort_values("Year")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["Year"],
        y=history["Production_ton"],
        mode='lines+markers',
        name="Actual Production",
        line=dict(color="green")
    ))
    fig.add_trace(go.Scatter(
        x=[selected_year],
        y=[prediction],
        mode='markers+text',
        name="Predicted",
        marker=dict(size=12, color="red", symbol="star"),
        text=[f"{prediction:.2f} t"],
        textposition="top center"
    ))
    fig.update_layout(
        title=f"📈 Production Trend: {selected_item} in {selected_area}",
        xaxis_title="Year",
        yaxis_title="Production (tons)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    result_df = pd.DataFrame([{
        "Area": selected_area,
        "Crop": selected_item,
        "Year": selected_year,
        "Area_Harvested_ha": area_harvested,
        "Yield_kg_ha": yield_kg_ha,
        "Predicted_Production_ton": prediction
    }])
    csv = result_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f"📥 [Download Prediction CSV](data:file/csv;base64,{b64})", unsafe_allow_html=True)

    html_chart = fig.to_html(include_plotlyjs='cdn')
    st.download_button("📉 Download Chart as HTML", data=html_chart, file_name="production_chart.html", mime="text/html")
