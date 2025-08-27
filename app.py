import streamlit as st 
import pickle
import numpy as np
import plotly.express as px

model = pickle.load(open('Health_risk_predictor.pkl','rb'))
encoders = pickle.load(open('label_encoders.pkl','rb'))

st.title("Health Risk Predictor")

age = st.slider('Age', 18, 80, 22)

diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
exercise = st.selectbox("Exercise Days per Week", options=list(range(0, 8)), index=3)
sleep = st.slider("Sleep Hours", 2, 12, 6)
stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
bmi = st.number_input("BMI", 10.0, 40.0, 22.0)
smoking = st.selectbox("Smoking Habit", ["No", "Yes"])
alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])    
family_history = st.selectbox("Family History of Disease", ["No", "Yes"])


# helper: safely encode labels (fallback if unseen)
def safe_encode(encoder, value):
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]


if st.button("Predict Health Risk"):
    input_data = [
        age,
        safe_encode(encoders['diet'], diet),
        exercise,
        sleep,
        safe_encode(encoders['stress'], stress),
        bmi,
        safe_encode(encoders['smoking'], smoking),
        safe_encode(encoders['alcohol'], alcohol),    
        safe_encode(encoders['family_history'], family_history)
    ]

    prediction = model.predict([input_data])
    risk_label = encoders['risk_level'].inverse_transform([prediction[0]])[0]

    # set color based on risk
    color_map = {
        "Low": "#4CAF50",     # Green
        "Medium": "#FFC107",  # Yellow
        "High": "#F44336"     # Red
    }
    color = color_map.get(risk_label, "#FFFFFF")

    # Colored box output
    st.markdown(
        f"""
        <div style="padding:15px; border-radius:10px; background-color:{color}; text-align:center; font-size:20px; font-weight:bold; color:white;">
            Predicted Health Risk Level: {risk_label}
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Lifestyle factor bar chart ---
    factors = {
        "Age": age,
        "Diet": diet,
        "Exercise (days)": exercise,
        "Sleep (hrs)": sleep,
        "Stress": stress,
        "BMI": bmi,
        "Smoking": smoking,
        "Alcohol": alcohol,
        "Family History": family_history
    }

    # Convert categorical factors into strings for visualization
    factors_str = {k: str(v) for k, v in factors.items()}

    st.subheader("Lifestyle Factors Overview")
    fig = px.bar(
        x=list(factors_str.keys()),
        y=list(factors_str.values()),
        labels={'x': "Factors", 'y': "Values"},
        text=list(factors_str.values()),
        title="Lifestyle Factors Entered by User"
    )
    st.plotly_chart(fig)
