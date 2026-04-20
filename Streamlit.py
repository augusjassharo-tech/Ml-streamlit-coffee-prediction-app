import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Coffee ML App", layout="wide")

st.title("☕ Coffee Quality Dashboard + Prediction")

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/adkwn1/data_visualization_using_streamlit/main/arabica_data_cleaned.csv")

# Clean data
df = df.dropna(subset=['aroma','flavor','acidity','total_cup_points'])

# Sidebar
st.sidebar.header("Filters")
country = st.sidebar.selectbox("Select Country", df['country_of_origin'].unique())
filtered_df = df[df['country_of_origin'] == country]

# Show data
st.subheader(f"Coffee Data - {country}")
st.dataframe(filtered_df)

# Charts
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(filtered_df, x="total_cup_points", title="Quality Score Distribution")
    st.plotly_chart(fig1)

with col2:
    fig2 = px.scatter(filtered_df, x="aroma", y="flavor",
                      size="total_cup_points", color="acidity",
                      title="Aroma vs Flavor")
    st.plotly_chart(fig2)

# ================= ML MODEL ================= #

st.header("🤖 Predict Coffee Quality Score")

X = df[['aroma','flavor','acidity']]
y = df['total_cup_points']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# User input
aroma = st.slider("Aroma", 0.0, 10.0, 7.0)
flavor = st.slider("Flavor", 0.0, 10.0, 7.0)
acidity = st.slider("Acidity", 0.0, 10.0, 7.0)

# Prediction
if st.button("Predict Score"):
    prediction = model.predict([[aroma, flavor, acidity]])
    st.success(f"Predicted Coffee Quality Score: {prediction[0]:.2f}")
