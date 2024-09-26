import streamlit as st
import pandas as pd
from recommend import recommend

st.title("End-to-End Recommendation System")

# Upload CSV file for data
uploaded_file = st.file_uploader("Upload a CSV file with user-item interactions", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:", data.head())

    user_id = st.number_input("Enter user ID for recommendations:", min_value=1)

    if st.button("Get Recommendations"):
        recommendations = recommend(user_id, data)
        st.write(f"Recommendations for user {user_id}:", recommendations)

