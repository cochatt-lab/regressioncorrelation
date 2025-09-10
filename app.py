
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import statsmodels.api as sm

# Load the original CSV file
csv_file = "TEST for DASH Student Performance Evaluation for 2025-2026 (Rising 9-12) (Responses) - WIP Rising 11.csv"
df = pd.read_csv(csv_file)

# Select relevant columns and drop rows with missing values
columns_of_interest = ["SPE average", "PreACT Composite Percentile PreACT", "PSAT Percentile Total"]
df_model = df[columns_of_interest].dropna()

# Rename columns for easier access
df_model.columns = ["SPE", "PreACT", "PSAT"]

# Regression model: SPE ~ PreACT + PSAT
X = df_model[["PreACT", "PSAT"]]
X = sm.add_constant(X)
y = df_model["SPE"]
model = sm.OLS(y, X).fit()

# Scatter matrix plot
fig = px.scatter_matrix(df_model, dimensions=["SPE", "PreACT", "PSAT"], color="SPE", title="Scatter Matrix: SPE vs PreACT and PSAT")

# Streamlit app layout
st.title("Regression Analysis: SPE vs PreACT and PSAT")
st.plotly_chart(fig)

st.subheader("Regression Model Summary")
st.text(model.summary())
