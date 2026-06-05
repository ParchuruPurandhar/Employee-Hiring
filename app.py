import streamlit as st
import pandas as pd
import joblib

# --------------------
# Page Config
# --------------------
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="👨‍💼",
    layout="wide"
)

# --------------------
# Load Files
# --------------------
model = joblib.load("employee_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# --------------------
# Header
# --------------------
st.title("👨‍💼 Employee Performance Prediction")
st.markdown(
    "Predict employee performance using HR analytics and machine learning."
)

# --------------------
# Sidebar
# --------------------
st.sidebar.header("About")

st.sidebar.info(
    """
    Employee Performance Prediction System
    
    Fill employee information and predict
    the expected performance rating.
    """
)

# --------------------
# Form
# --------------------
with st.form("employee_form"):

    st.subheader("Employee Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.slider("Age", 18, 60, 30)

        distance = st.number_input(
            "Distance From Home",
            1,
            50,
            5
        )

        emp_education_level = st.selectbox(
            "Education Level",
            [1, 2, 3, 4, 5]
        )

        env_satisfaction = st.selectbox(
            "Environment Satisfaction",
            [1, 2, 3, 4]
        )

        hourly_rate = st.number_input(
            "Hourly Rate",
            30,
            150,
            80
        )

        job_involvement = st.selectbox(
            "Job Involvement",
            [1, 2, 3, 4]
        )

    with col2:

        job_level = st.selectbox(
            "Job Level",
            [1, 2, 3, 4, 5]
        )

        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            [1, 2, 3, 4]
        )

        num_companies = st.number_input(
            "Companies Worked",
            0,
            20,
            2
        )

        salary_hike = st.number_input(
            "Last Salary Hike %",
            0,
            50,
            15
        )

        relationship_satisfaction = st.selectbox(
            "Relationship Satisfaction",
            [1, 2, 3, 4]
        )

        total_experience = st.number_input(
            "Total Experience (Years)",
            0,
            40,
            8
        )

    with col3:

        training_times = st.number_input(
            "Training Times Last Year",
            0,
            10,
            2
        )

        work_life_balance = st.selectbox(
            "Work Life Balance",
            [1, 2, 3, 4]
        )

        years_company = st.number_input(
            "Years At Company",
            0,
            40,
            5
        )

        years_role = st.number_input(
            "Years In Current Role",
            0,
            20,
            3
        )

        years_promotion = st.number_input(
            "Years Since Last Promotion",
            0,
            20,
            1
        )

        years_manager = st.number_input(
            "Years With Current Manager",
            0,
            20,
            3
        )

    st.subheader("Categorical Information")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        attrition = st.selectbox(
            "Attrition",
            ["Yes", "No"]
        )

    with c2:

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

        overtime = st.selectbox(
            "OverTime",
            ["Yes", "No"]
        )

    with c3:

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Divorced"
            ]
        )

        business_travel = st.selectbox(
            "Business Travel",
            [
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel"
            ]
        )

    with c4:

        emp_department = st.selectbox(
            "Department",
            [
                "Sales",
                "Development",
                "Research & Development",
                "Human Resources",
                "Finance"
            ]
        )

        emp_job_role = st.selectbox(
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

    predict = st.form_submit_button(
        "🔍 Predict Performance"
    )

# --------------------
# Prediction
# --------------------
if predict:

    input_data = pd.DataFrame({

        "Age":[age],
        "DistanceFromHome":[distance],
        "EmpEducationLevel":[emp_education_level],
        "EmpEnvironmentSatisfaction":[env_satisfaction],
        "EmpHourlyRate":[hourly_rate],
        "EmpJobInvolvement":[job_involvement],
        "EmpJobLevel":[job_level],
        "EmpJobSatisfaction":[job_satisfaction],
        "NumCompaniesWorked":[num_companies],
        "EmpLastSalaryHikePercent":[salary_hike],
        "EmpRelationshipSatisfaction":[relationship_satisfaction],
        "TotalWorkExperienceInYears":[total_experience],
        "TrainingTimesLastYear":[training_times],
        "EmpWorkLifeBalance":[work_life_balance],
        "ExperienceYearsAtThisCompany":[years_company],
        "ExperienceYearsInCurrentRole":[years_role],
        "YearsSinceLastPromotion":[years_promotion],
        "YearsWithCurrManager":[years_manager],

        "Gender":[gender],
        "EducationBackground":[education_bg],
        "MaritalStatus":[marital_status],
        "EmpDepartment":[emp_department],
        "EmpJobRole":[emp_job_role],
        "BusinessTravelFrequency":[business_travel],
        "OverTime":[overtime],
        "Attrition":[attrition]
    })

    transformed = preprocessor.transform(input_data)

    prediction = model.predict(transformed)[0]

    st.success(
        f"Predicted Performance Rating: {prediction}"
    )
