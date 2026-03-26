import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import csv

# ---------------- SAVE FUNCTION ----------------
def save_patient(data):
    with open("patients.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data)

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="Hemophilia AI Platform", layout="wide")

st.markdown("<h1 style='text-align:center;'>🧬 Hemophilia AI Clinical Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered inhibitor risk prediction & clinical support</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.title("👤 Patient Details")

name = st.sidebar.text_input("Patient Name")
age = st.sidebar.slider("Age", 0, 80)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
ethnicity = st.sidebar.selectbox("Ethnicity", ["Asian", "African", "European"])

severity = st.sidebar.selectbox("Severity", ["Mild", "Moderate", "Severe"])
mutation = st.sidebar.selectbox("Mutation Type", ["Intron22", "Missense", "Nonsense", "Frameshift"])

family_history = st.sidebar.selectbox("Family History", ["Yes", "No"])

dose = st.sidebar.slider("Dose Intensity", 0, 100)
exposure = st.sidebar.slider("Exposure Days", 0, 100)

predict_btn = st.sidebar.button("🔍 Predict Risk")

# ---------------- API ----------------
url = "https://hemophilia-api.onrender.com/predict"

# ---------------- MEDICAL ADVICE ----------------
def generate_medical_advice(risk, severity, mutation, dose, exposure):

    advice = ""

    if risk > 0.6:
        advice += "⚠️ HIGH RISK of inhibitor development\n\n"

        if severity == "Severe":
            advice += "- Severe hemophilia significantly increases inhibitor formation risk.\n"

        if mutation.lower() == "intron22":
            advice += "- Intron22 mutation is strongly linked with inhibitor development.\n"

        if dose > 50:
            advice += "- High dose intensity can trigger immune response.\n"

        if exposure > 20:
            advice += "- Increased exposure days raise inhibitor formation probability.\n"

        advice += "\n🩺 Recommended Precautions:\n"
        advice += "- Regular inhibitor screening\n"
        advice += "- Monitor factor VIII activity closely\n"
        advice += "- Consult hematologist urgently\n"
        advice += "- Consider immune tolerance therapy\n"

    else:
        advice += "✅ LOW RISK\n\n"
        advice += "- Continue regular treatment\n"
        advice += "- Maintain periodic monitoring\n"
        advice += "- Watch for unusual bleeding symptoms\n"

    return advice

# ---------------- PREDICTION ----------------
if predict_btn:

    params = {
        "age": age,
        "dose": dose,
        "exposure": exposure
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        risk = data.get("risk_score", 0)
        reason = data.get("reason", "Unknown")
        importance = data.get("importance", {})

        # ---------------- SAVE DATA ----------------
        save_patient([name, age, gender, severity, mutation, dose, exposure, risk, reason])

        # ---------------- RESULT ----------------
        st.markdown("## 📊 Prediction Result")

        st.metric("Risk Score", round(risk, 2))

        if risk > 0.6:
            st.error("⚠️ HIGH RISK")
        else:
            st.success("✅ LOW RISK")

        # ---------------- AI EXPLANATION ----------------
        st.markdown("### 🧠 AI Clinical Explanation")

        explanation = f"""
        This patient shows a **{'high' if risk > 0.6 else 'low'} probability** of inhibitor development.

        The most influential factor is **{reason}**.

        Clinical indicators:
        - Severity: {severity}
        - Mutation Type: {mutation}
        - Dose Intensity: {dose}
        - Exposure Days: {exposure}

        These factors influence how the immune system reacts to treatment.
        """

        st.info(explanation)

        # ---------------- MEDICAL ADVICE ----------------
        st.markdown("### 🩺 Clinical Recommendations")

        advice = generate_medical_advice(
            risk, severity, mutation, dose, exposure
        )

        st.warning(advice)

        # ---------------- GRAPH ----------------
        if importance:
            st.markdown("### 📈 Feature Impact")

            imp_df = pd.DataFrame(
                importance.items(),
                columns=["Feature", "Impact"]
            ).sort_values(by="Impact", ascending=False).head(8)

            fig, ax = plt.subplots()
            ax.barh(imp_df["Feature"], imp_df["Impact"])
            ax.invert_yaxis()

            st.pyplot(fig)

        # ---------------- PATIENT SUMMARY ----------------
        st.markdown("### 🧾 Patient Summary")

        st.write({
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Ethnicity": ethnicity,
            "Severity": severity,
            "Mutation": mutation,
            "Family History": family_history,
            "Dose": dose,
            "Exposure": exposure
        })

        # ---------------- CHATBOT ----------------
        elif page == "Chatbot":

    st.title("🤖 AI Medical Assistant")

    if "data" not in st.session_state:
        st.warning("Run prediction first")
    else:
        d = st.session_state.data

        question = st.text_input("Ask about patient condition")

        if question:

            prompt = f"""
            You are a medical assistant.

            Patient details:
            Risk: {d['risk']}
            Severity: {d['severity']}
            Mutation: {d['mutation']}
            Reason: {d['reason']}

            Question: {question}

            Give a detailed, helpful, medical explanation in simple language.
            """

            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response.choices[0].message.content
                st.write(answer)

            except Exception as e:
                st.error(f"Error: {e}")

        # ---------------- PATIENT HISTORY ----------------
        st.markdown("### 📂 Patient History")

        try:
            df = pd.read_csv("patients.csv")
            st.dataframe(df)
        except:
            st.write("No patient data available yet")

        # ---------------- DISCLAIMER ----------------
        st.markdown("---")
        st.caption("⚠️ This AI system provides decision support only. Always consult a medical professional.")

    except:
        st.error("❌ Unable to connect to AI server. Please try again.")
