import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Hemophilia AI", layout="wide")

st.title("🧬 Hemophilia Risk Predictor")

# Sidebar input
st.sidebar.header("Patient Input")

age = st.sidebar.slider("Age", 0, 20)
dose = st.sidebar.slider("Dose", 0, 100)
exposure = st.sidebar.slider("Exposure Days", 0, 100)

# Button
if st.sidebar.button("Predict"):

    url = "http://127.0.0.1:8000/predict"
    params = {"age": age, "dose": dose, "exposure": exposure}

    response = requests.get(url, params=params)

    if response.status_code == 200:

        data = response.json()

        # SAFE ACCESS (NO ERROR)
        risk = data.get("risk_score", 0)
        reason = data.get("reason", "Not available")
        importance = data.get("importance", {})

        st.subheader(f"Risk Score: {round(risk,2)}")

        if risk > 0.6:
            st.error("⚠️ HIGH RISK")
        else:
            st.success("✅ LOW RISK")

        st.info(f"Main Factor: {reason}")

        # GRAPH
        if importance:
            imp_df = pd.DataFrame(
                importance.items(),
                columns=["Feature", "Impact"]
            )
            imp_df = imp_df.sort_values(by="Impact", ascending=False)

            st.subheader("📊 Feature Importance")
            st.bar_chart(imp_df.set_index("Feature"))

    else:
        st.error("API Error")