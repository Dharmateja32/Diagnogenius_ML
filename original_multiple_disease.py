import pickle
import streamlit as st
from streamlit_option_menu import option_menu

import numpy as np
import pandas as pd

loaded_model_cardio = pickle.load(open("C:/Users/dharm/OneDrive/Documents/College_Projects/Diagnogenious_Project/cardio_model.sav", 'rb'))
cluster_cardio = pickle.load(open("C:/Users/dharm/OneDrive/Documents/College_Projects/Diagnogenious_Project/cardio_cluster_model.sav",'rb'))

loaded_model_diabetes = pickle.load(open("C:/Users/dharm/OneDrive/Documents/College_Projects/Diagnogenious_Project/diabetes_model.sav",'rb'))
transformer_diabetes = pickle.load(open("C:/Users/dharm/OneDrive/Documents/College_Projects/Diagnogenious_Project/diabetes_transformer_model.sav",'rb'))

parkinsons_model = pickle.load(open("C:/Users/dharm/OneDrive/Documents/College_Projects/Diagnogenious_Project/parkinsons_model.sav", 'rb'))


with st.sidebar:
    selected = option_menu('DIAGNOGENIUS - VDPP',
                           ['CVD Prediction',
                            'Parkinsons Prediction',
                            'Diabetes Prediction'],
                           menu_icon='hospital-fill',
                           icons=['heart','person','activity'],
                           default_index=0)

# age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
# CVD Prediction Page
if selected == 'CVD Prediction':

    # page title
    st.title('Cardiovascular Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        gender = st.text_input('Gender')

    with col3:
        height = st.text_input('Heigth')

    with col1:
        weight = st.text_input('Weight')

    with col2:
        ap_hi = st.text_input('ap_hi')

    with col3:
        ap_lo = st.text_input('ap_lo')

    with col1:
        cholesterol = st.text_input('Cholesterol')

    with col2:
        gluc = st.text_input('Glucose')

    with col3:
        smoke = st.text_input('Smoke')

    with col1:
        alco = st.text_input('Alcohol')

    with col2:
        active = st.text_input('Active')

    # code for Prediction
    cardio_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        # user_input = [age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]

        # user_input = [float(x) for x in user_input]

        age_edges = [30, 35, 40, 45, 50, 55, 60, 65]
        age_labels = [0, 1, 2, 3, 4, 5, 6]

        bmiMin = 15
        bmiMax = 48
        bin_width = 5
        bmi_edges = list(range(bmiMin, bmiMax + bin_width, bin_width))
        bmi_labels = list(range(len(bmi_edges)-1))

        mapMin = 73
        mapMax = 123
        bin_width = 10
        map_edges = list(range(mapMin, mapMax + bin_width, bin_width))
        map_labels = list(range(len(map_edges)-1))

        ageY = int(round(float(age) / 365))
        bmi = float(weight)/((float(height)/100)**2)
        map = ((2* float(ap_lo)) + float(ap_hi)) / 3

        age_group = pd.cut([ageY], bins=age_edges, labels=age_labels, right=True, include_lowest=True)[0]
        bmi = pd.cut([bmi], bins=bmi_edges, labels=bmi_labels, right=True, include_lowest=True)[0]
        map = pd.cut([map], bins=map_edges, labels=map_labels, right=True, include_lowest=True)[0]

        instance = 	[gender,cholesterol,gluc,smoke,alco,active,age_group,bmi,map]

        instance = [float(x) for x in instance]

        if instance[0] == 2:
            instance[0] = 1
        elif instance[0] == 1:
            instance[0] = 0
        
        input_data = np.array(instance).reshape(1, -1)
        input_data = pd.DataFrame(input_data, columns=['gender','cholesterol','gluc','smoke','alco','active','age_group','bmi','map'])

        clusters = cluster_cardio.predict(input_data)
        input_data.insert(0,"clusters",clusters,True)


        cardio_prediction = loaded_model_cardio.predict(input_data)

        if cardio_prediction[0] == 1:
            cardio_diagnosis = 'The person is having cardio disease'
        else:
            cardio_diagnosis = 'The person does not have any cardio disease'

    st.success(cardio_diagnosis)

# Diabetes Prediction Page

gen = ('Male', 
         'Female')
smoking = ('never', 
                  'No Info', 
                  'current', 
                  'former', 
                  'not current', 
                  'ever')

if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("gender", gen)
    with col2:
        age = st.text_input("age")
    with col3:
        hypertension = st.text_input("hypertension")
    with col1:
        heart_disease = st.text_input("heart_disease")
    with col2:
        smoking_history = st.selectbox("smoking_history", smoking)
    with col3:
        bmi = st.number_input("bmi")
    with col1:
        HbA1c_level = st.number_input("HbA1c_level")
    with col2:
        blood_glucose_level = st.text_input("blood_glucose_level")

    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [gender, age, hypertension, heart_disease, smoking_history, bmi, 
                      HbA1c_level, blood_glucose_level]
        
        gender_mapping = {'Male': 0, 'Female': 1}
        smoking_history_mapping = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'not current': 4, 'ever': 5}

        # Replace categorical variables with numerical values
        user_input[0] = gender_mapping[user_input[0]]
        user_input[4] = smoking_history_mapping[user_input[4]]
        
        user_input = [float(x) for x in user_input]
        
        user_input = np.array(user_input).reshape(1, -1)

        # Convert the input to a DataFrame
        user_input = pd.DataFrame(user_input, columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

        transformed_data = transformer_diabetes.transform(user_input)
        
        diab_prediction = loaded_model_diabetes.predict(transformed_data)
        
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)
        
        
        
# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP: (Fo(Hz))')

    with col2:
        fhi = st.text_input('MDVP: (Fhi(Hz))')

    with col3:
        flo = st.text_input('MDVP: (Flo(Hz))')

    with col4:
        Jitter_percent = st.text_input('MDVP: (Jitter(%))')

    with col5:
        Jitter_Abs = st.text_input('MDVP: (Jitter(Abs))')

    with col1:
        RAP = st.text_input('MDVP: (RAP)')

    with col2:
        PPQ = st.text_input('MDVP: (PPQ)')

    with col3:
        DDP = st.text_input('Jitter: (DDP)')

    with col4:
        Shimmer = st.text_input('MDVP: (Shimmer)')

    with col5:
        Shimmer_dB = st.text_input('MDVP: (Shimmer(dB))')

    with col1:
        APQ3 = st.text_input('Shimmer: (APQ3)')

    with col2:
        APQ5 = st.text_input('Shimmer: (APQ5)')

    with col3:
        APQ = st.text_input('MDVP: (APQ)')

    with col4:
        DDA = st.text_input('Shimmer: (DDA)')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)