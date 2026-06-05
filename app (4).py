import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("employee_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="👨‍💼",
    layout="wide"
)

# ---------------------------
# Header
# ---------------------------
st.title("👨‍💼 Employee Performance Prediction System")
st.markdown("Predict employee performance using HR analytics.")
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=120
)

st.sidebar.title("HR Analytics")

st.sidebar.markdown("""
### About

This application predicts employee performance based on:

- Employee Details
- Work History
- Education
- Travel Frequency
- Overtime Status
""")

st.divider()

# ---------------------------
# Input Form
# ---------------------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age", 18, 60, 30)

        attrition = st.selectbox(
            "Attrition",
            ["Yes", "No"]
        )

        business_travel = st.selectbox(
            "Business Travel",
            [
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel"
            ]
        )

        department = st.selectbox(
            "Department",
            [
                "Sales",
                "Research & Development",
                "Human Resources"
            ]
        )

        distance = st.number_input(
            "Distance From Home",
            min_value=1,
            max_value=50,
            value=5
        )

        education = st.selectbox(
            "Education Level",
            [1, 2, 3, 4, 5]
        )

    with col2:

        education_bg = st.selectbox(
            "Education Background",
            [
                "Marketing",
                "Medical",
                "Life Sciences",
                "Technical Degree",
                "Human Resources",
                "Other"
            ]
        )

        emp_department = st.selectbox(
            "Employee Department",
            [
                "Sales",
                "Development",
                "Research & Development",
                "Human Resources",
                "Finance"
            ]
        )

        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Manager",
                "Research Scientist",
                "Developer",
                "HR",
                "Laboratory Technician"
            ]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Divorced"
            ]
        )

        overtime = st.selectbox(
            "OverTime",
            ["Yes", "No"]
        )

    submitted = st.form_submit_button("🔍 Predict Performance")

# ---------------------------
# Prediction
# ---------------------------
if submitted:

    input_data = pd.DataFrame({
        "Age": [age],
        "Attrition": [attrition],
        "BusinessTravelFrequency": [business_travel],
        "Department": [department],
        "DistanceFromHome": [distance],
        "Education": [education],
        "EducationBackground": [education_bg],
        "EmpDepartment": [emp_department],
        "EmpJobRole": [job_role],
        "Gender": [gender],
        "MaritalStatus": [marital_status],
        "OverTime": [overtime]
    })

    transformed = preprocessor.transform(input_data)

    prediction = model.predict(transformed)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction >= 4:
        st.success(
            f"⭐ Predicted Performance Rating: {prediction}"
        )
    elif prediction >= 3:
        st.info(
            f"📈 Predicted Performance Rating: {prediction}"
        )
    else:
        st.warning(
            f"⚠️ Predicted Performance Rating: {prediction}"
        )

    st.dataframe(input_data)