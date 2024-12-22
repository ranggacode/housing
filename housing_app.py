import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model (ensure model.pkl exists)
with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict house price
def predict_price(features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit page configuration
st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f8f9;
        }
        .header {
            font-size: 40px;
            font-weight: 700;
            color: #2a3d66;
            text-align: center;
            padding-top: 20px;
        }
        .subheader {
            font-size: 20px;
            font-weight: 500;
            color: #2a3d66;
        }
        .input-container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .predict-button {
            background-color: #2a3d66;
            color: white;
            padding: 12px 25px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .predict-button:hover {
            background-color: #1a2a44;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 14px;
            color: #888;
        }
        .feature-description {
            font-size: 14px;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown('<div class="header">House Price Prediction in South Jakarta</div>', unsafe_allow_html=True)

# Description of the app
st.markdown("""
    <p class="feature-description">
    We understand that finding the right price for your dreamhouse is very difficult! Especially if this is your first time buying a house.
    Use this app to predict the price of a house based on various features like the number of bedrooms, bathrooms, area, and more.
    Simply input the details and click "Predict" to estimate the price of the house.
    </p>
""", unsafe_allow_html=True)

# Create a form for user inputs
with st.form(key='house_form'):
    st.markdown('<div class="subheader">Enter the house details:</div>', unsafe_allow_html=True)

    # Collect user inputs for features
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        rm = st.number_input("Number of bedrooms", min_value=1, max_value=10)
        age = st.number_input("Age of property", min_value=1, max_value=100)
        dis = st.number_input("Distance to employment center in km", min_value=1, max_value=100)
        tax = st.number_input("Tax rate %", min_value=0, max_value=100)
        chas = st.selectbox("Near river?", options=["Yes", "No"])
        # Convert 'Yes'/'No' to 1/0 for model prediction
        chas = 1 if chas == "Yes" else 0
        rad = st.slider("Accessibility to the radial highways (the higher the closer)", min_value=0., max_value=1.)
        ptratio = st.slider("Proportion of student compare to teacher in the area %", min_value=0, max_value=100)

    with col2:
        crim = st.slider("Criminality rate %", min_value=0, max_value=100)
        zn = st.slider("Proportion of residential zone in the area %", min_value=0, max_value=100)
        indus = st.slider("Proportion of non-retail business in the area %", min_value=0, max_value=100)
        lstat = st.slider("Economic status of the population {0 = rich < range < 100 = poor} %", min_value=0, max_value=100)
        black = st.slider("Proportion of black people in the population %", min_value=0, max_value=100)
        nox = st.slider("nitrogen oxides concentration (parts per 10 million) (the lesser the healthier)", min_value=0., max_value=1.)
        
    
    # Submit button for the form
    submit_button = st.form_submit_button(label="Predict House Price", use_container_width=True)

# Prediction and Result Display
if submit_button:
    # Combine user inputs into a feature list
    user_features = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]
    
    # Get the predicted price from the model
    predicted_price = predict_price(user_features)

    # Display the result
    st.markdown(f'<div class="subheader">Predicted House Price:</div>', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color: #2a3d66;">${predicted_price:,.2f}k</h3>', unsafe_allow_html=True)

# Footer (credits or additional info)
st.markdown("""
    <div class="footer">
        Built with ❤️ for real estate enthusiasts. 
        <br>Model powered by machine learning algorithms.
    </div>
""", unsafe_allow_html=True)
