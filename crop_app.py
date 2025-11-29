
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app
st.title("🌾 Crop Recommendation System")

st.write("""
Enter the soil and weather conditions to get a recommendation for the best crop to grow.
""")

# User input
st.header("Enter the following values:")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90)
P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=42)
K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=43)
temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=44.0, value=20.8, step=0.1, format="%.1f")
humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=100.0, value=82.0, step=0.1, format="%.1f")
ph = st.number_input("pH", min_value=3.5, max_value=9.9, value=6.5, step=0.1, format="%.1f")
rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=299.0, value=202.9, step=0.1, format="%.1f")

# Predict button
if st.button("Predict Crop"):
    # Create a dataframe from user input
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the result
    st.success(f"🌱 Recommended Crop: {prediction[0]}")
