import streamlit as st
import numpy as np
import joblib
from preprocessing_data import dataFrame

admit_type_options = [
    (0, "ROUTINE"),
    (1, "URGENT"),
    (2, "ELECTIVE"),
    (3, "NEWBORN"),
    (5, "TRAUMA CENTER"),
]

adm_src = joblib.load('pkl/adm_src.pkl')
ms_drg = joblib.load('pkl/ms_drg.pkl')

unique_diag_label = joblib.load('pkl/unique_diag_label.pkl')
unique_diag_label_2 = joblib.load('pkl/unique_diag_label_2.pkl')
chronic_condition = joblib.load('pkl/chronic_condition.pkl')

st.title('🛌 Length Of Stay Prediction')
st.write("Please provide the patient's details below to help predict their hospital stay duration")

with st.container():
    with st.expander("👤 Basic Patient Information"):
        gender = st.selectbox(
            'Select your gender 🧑‍⚕️:',
            ['MALE', 'FEMALE'],
            help="Choose the patient's gender for personalized predictions."
        )

        age = st.number_input(
            'Select your age 🗓️:',
            min_value=0, max_value=100, step=1,
            help="Enter the patient's age. Age plays a significant role in predicting hospital stay."
        )

    with st.expander("🌍 Patient's Origin and Insurance Details"):
        state = st.selectbox(
            'Select your state origin 🌎:',
            ['MI - MICHIGAN', 'HI - HAWAII', 'CA - CALIFORNIA', 'WA - WASHINGTON'],
            help="Select the state where the patient resides. This can impact healthcare services."
        )

        insurance = st.selectbox(
            'Select your insurance 💳:',
            ['COMMERCIAL', 'MEDICARE', 'MEDICAID', 'MEDICARE SUPPLEMENT'],
            help="Choose the patient's insurance type to tailor the prediction accordingly."
        )


    with st.expander("🏥 Admission Information"):
        admit_type = st.selectbox(
            'Select your admit type 🏥:',
            options=[f"{code} - {desc}" for code, desc in admit_type_options],
            help="Select the type of admission to understand the severity of the condition."
        )

        admit_source = st.selectbox(
            'Select your admit source 🏠:',
            options=['9' if code == None else code for code in adm_src],
            help="Choose the source from which the patient was admitted to the hospital."
        )

        ms_drg = st.selectbox(
            'Select your MS-DRG (inpatient code) 📊:',
            options=[code for code in ms_drg],
            help="Select the appropriate MS-DRG code for inpatient diagnosis."
        )

    with st.expander("🔍 Diagnosis and Discharge Information"):
        dis_stat = st.number_input(
            'Select your UB Discharge Status 📅:',
            min_value=0, max_value=95, step=1,
            help="Provide the UB Discharge Status code to assist in determining recovery times."
        )

        diag_label_1 = st.selectbox(
            'Select your primary diagnosis 🩺:',
            ['Unknown' if label == None else label for label in unique_diag_label],
            help="Choose the patient's primary diagnosis code. This will impact the prediction of stay length."
        )

        diag_label_2 = st.selectbox(
            'Select your secondary diagnosis 🩻:',
            ['Unknown' if label == None else label for label in unique_diag_label_2],
            help="Select the secondary diagnosis to factor in additional conditions."
        )

    with st.expander("⚠️ Chronic Conditions"):
        chronic_condition = st.selectbox(
            'Select your chronic disease 💔:',
            ['-1 - UNKNOWN ' if label == None else label for label in chronic_condition],
            help="Select any chronic condition the patient has, which may impact their stay duration."
        )

    if st.button("Predict🕰️"):
        st.write("🔍 Predicting the length of stay based on the provided information...")

        data = [
            (1 if gender == 'MALE' else 0 ),
            age,
            insurance,
            int(chronic_condition.split(' - ')[0]),
            diag_label_1,
            ms_drg,
            diag_label_2,
            admit_source,
            int(admit_type.split(' - ')[0]),
            dis_stat,
            (1 if state == 'HI - HAWAII' else 0),
            (1 if state == 'WA - WASHINGTON' else 0),
            (1 if state == 'MI - MICHIGAN' else 0),
            (1 if state == 'CA - CALIFORNIA' else 0)
            ]
        
        dataframe = dataFrame(data)
        prediction = dataframe.predict()[0]
        st.success(f"Predicted days: {prediction:,.0f}",icon="✅")

