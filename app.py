import streamlit as st
import numpy as np
import joblib

model = joblib.load("insurance_model.pkl")

st.title("Medical Insurance Cost Estimator")

age = st.slider("Age", 18, 65)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 15.0, 45.0)
children = st.slider("Children", 0, 5)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# One-hot encoding manually
sex_male = 1 if sex == 'male' else 0
smoker_yes = 1 if smoker == 'yes' else 0
region_dummies = {
    'southeast': [0, 0, 0],
    'southwest': [1, 0, 0],
    'northeast': [0, 1, 0],
    'northwest': [0, 0, 1]
}
region_vals = region_dummies[region]

# Create feature vector
input_data = np.array([
    age, bmi, children,
    sex_male, smoker_yes,
    *region_vals,
    age * smoker_yes,
    bmi * smoker_yes
]).reshape(1, -1)

prediction = model.predict(input_data)[0]
st.success(f"Estimated Insurance Charges: ${prediction:,.2f}")


# Step 8: Run Streamlit
# In terminal, run:
# streamlit run app.py