
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------
# 1. Page configuration
# --------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------
# 2. Load trained model
# --------------------------------

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "student_performance_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------
# 3. Application heading
# --------------------------------

st.title("Student Performance Predictor")
st.subheader("Machine Learning | Regression")

st.markdown("""
Explore how machine learning can estimate a student's
final mathematics or Portuguese course grade using
selected student-related information.
""")

st.caption("Built with Python, Scikit-learn and Streamlit.")

st.info(
    "This is an experimental model for learning purposes. "
    "Its predictions should not be used to judge students "
    "or make academic decisions."
)

st.divider()

# --------------------------------
# 4. Input form
# --------------------------------

with st.expander("About the Machine Learning Model"):
    st.write("**Algorithm:** Random Forest Regressor")
    st.write("**Training dataset:** UCI Student Performance")
    st.write("**Input features:** 30")
    st.write("**Evaluation results:**")
    st.write("- MAE: 2.023")
    st.write("- RMSE: 2.764")
    st.write("- R²: 0.217")

    st.caption(
        "Evaluation metrics are from the held-out test set. "
        "The model has modest predictive performance."
    )

with st.expander("Project Overview"):
    st.markdown("""
    ### Objective
    Predict students' final course grades using supervised
    machine learning.

    ### Workflow
    1. Data collection from the UCI Machine Learning Repository.
    2. Data cleaning and preprocessing.
    3. Categorical feature encoding and numerical scaling.
    4. Training and evaluation of regression models.
    5. Deployment using Streamlit.

    ### Models Compared
    - Linear Regression
    - Random Forest Regressor

    ### Best Model
    Random Forest Regressor

    ### Important Limitation
    The model achieved an R² score of 0.217 on the test set.
    Its predictions have limited accuracy and should only
    be used for educational demonstration.
    """)

st.header("Enter Student Information")

with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.number_input("Age", 15, 22, 17)
        address = st.selectbox("Address", ["U", "R"])
        famsize = st.selectbox("Family Size", ["GT3", "LE3"])
        Pstatus = st.selectbox("Parent Status", ["T", "A"])

        Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4])
        Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4])

        Mjob = st.selectbox(
            "Mother's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        Fjob = st.selectbox(
            "Father's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        reason = st.selectbox(
            "Reason for School Choice",
            ["home", "reputation", "course", "other"]
        )

        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )

        traveltime = st.selectbox("Travel Time", [1, 2, 3, 4])
        studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
        failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])

    with col2:
        schoolsup = st.selectbox("School Support", ["yes", "no"])
        famsup = st.selectbox("Family Support", ["yes", "no"])
        paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
        activities = st.selectbox("Extra Activities", ["yes", "no"])
        nursery = st.selectbox("Attended Nursery", ["yes", "no"])
        higher = st.selectbox("Wants Higher Education", ["yes", "no"])
        internet = st.selectbox("Internet Access", ["yes", "no"])
        romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])

        famrel = st.slider("Family Relationship Quality", 1, 5, 4)
        freetime = st.slider("Free Time", 1, 5, 3)
        goout = st.slider("Going Out Frequency", 1, 5, 3)
        Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)
        Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)
        health = st.slider("Self-reported Health", 1, 5, 3)
        absences = st.number_input(
            "School Absences",
            min_value=0,
            max_value=100,
            value=2
        )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        use_container_width=True
    )

# --------------------------------
# 5. Make prediction
# --------------------------------

if submitted:

    student_data = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences
    }])    
    
    prediction = model.predict(student_data)[0]

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            label="Estimated Final Grade",
            value=f"{prediction:.2f} / 20"
        )

    with result_col2:
        st.metric(
            label="Rounded Estimate",
            value=f"{round(prediction):.0f} / 20"
        )

    st.progress(
        max(0.0, min(float(prediction) / 20, 1.0))
    )

    st.info(
        "This is an experimental machine learning estimate. "
        "It is not an actual grade or a reliable assessment "
        "of a student's ability or potential."
    )
