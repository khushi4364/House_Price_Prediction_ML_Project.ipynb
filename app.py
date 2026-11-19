
import streamlit as st
import pandas as pd
import joblib

# Load trained ML model
model = joblib.load("house_price_model.pkl")

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# Project heading
st.title("🏡 House Price Prediction System")

st.write(
    "Welcome! Enter the house details below "
    "to estimate its market value."
)

st.divider()

# Input form
st.subheader("Enter House Details")

with st.form("house_form"):

    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.number_input(
            "Median Income",
            min_value=0.5,
            max_value=15.0,
            value=5.0,
            step=0.1
        )

        house_age = st.number_input(
            "House Age (years)",
            min_value=1.0,
            max_value=52.0,
            value=20.0
        )

        ave_rooms = st.number_input(
            "Average Rooms",
            min_value=1.0,
            max_value=20.0,
            value=6.0
        )

        ave_bedrms = st.number_input(
            "Average Bedrooms",
            min_value=0.5,
            max_value=5.0,
            value=1.0
        )

    with col2:
        population = st.number_input(
            "Population",
            min_value=1.0,
            max_value=40000.0,
            value=1000.0
        )

        ave_occup = st.number_input(
            "Average Occupancy",
            min_value=1.0,
            max_value=20.0,
            value=3.0
        )

        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=34.0
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-118.0
        )

    predict_button = st.form_submit_button(
        "Predict House Price"
    )

# Make prediction
if predict_button:

    input_data = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    prediction = model.predict(input_data)[0]

    price = prediction * 100000

    st.divider()

    st.success("Prediction completed!")

    st.subheader("Estimated House Price")

    st.metric(
        label="Predicted Value (USD)",
        value=f"${price:,.2f}"
    )

    st.info(
        "This is an educational estimate based on "
        "the California Housing dataset. It is not "
        "a professional property valuation."
    )

# Footer
st.divider()

st.caption(
    "Developed as a Machine Learning portfolio project "
    "| Python | Pandas | Scikit-learn | Streamlit"
)
