import pickle
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Loading the saved model

diabetes_model=pickle.load(open('./diabetes_model.sav','rb'))
lungcancer_model=pickle.load(open('./lung_cancer.sav','rb'))
parkinsons_model=pickle.load(open('./parkinsons_model.sav','rb'))
heartdisease_model=pickle.load(open('./heartdisease.sav','rb'))
breast_model=pickle.load(open('./breast_cancer.sav','rb'))

# to scale the input data 
scaler = pickle.load(open('./scaler.sav', 'rb'))  # Load the same scaler used in training
scaler2=pickle.load(open('./scaler2.sav','rb'))
scaler3=pickle.load(open('./scaler3.sav','rb'))
scaler4=pickle.load(open('./scaler4.sav','rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction",
        options=["Diabetes Prediction", 
                 "Lung Cancer Prediction",
                   "Parkinson's Disease Prediction",
                   "Heart Disease Prediction",
                   "Breast Cancer Prediction"
                   ],
                   icons=["activity", "lungs", "person-arms-up", "heart","heart-pulse","virus"],
                   default_index=0

    )

# diabetes prediction page
if(selected=="Diabetes Prediction"):
    st.title("Diabetes Prediction using ML")
    st.write("This is a simple diabetes prediction model using machine learning")
    # collecting user input features

    col1, col2,col3 = st.columns(3)
    with col1:
        Pregnancies=st.text_input("Number of pregnancies ")
    with col2:
        Glucose=st.text_input("Glucose Level ")
    with col3:
        BloodPressure=st.text_input("BloodPressure Value ")
    with col1:
        SkinThickness=st.text_input("Skin Thickness Value ")
    with col2:
        Insulin=st.text_input("Insulin level ")        
    with col3:
        BMI=st.text_input("BMI value ")
    with col1:
        DiabetesPedigreeFunction=st.text_input("DiabetesPedigreeFunction Value ")
    with col2:
        Age=st.text_input("Age of the person ")
                # making prediction using the model

    # code for prediction
    diab_diagnosis=''

    # creating button for prediction
    if st.button("Result"):
        # code for prediction
        try:

            user_input = np.array([[
            float(Pregnancies), float(Glucose), float(BloodPressure),
            float(SkinThickness), float(Insulin), float(BMI),
            float(DiabetesPedigreeFunction), float(Age)
        ]])
        except ValueError:
            st.error("Please enter valid numerical values.")
            diab_predict = [-1]  # Prevent further errors        # displaying the result

        user_input_scaled = scaler.transform(user_input)
    
        diab_predict = diabetes_model.predict(user_input_scaled)
        # print(diab_predict[0])
        if(diab_predict[0]==1):
            diab_diagnosis="You are Diabetic"
        elif (diab_predict[0]==-1):
            diab_diagnosis="Error !!"
        else:
            diab_diagnosis="You are not Diabetic"
    st.success(diab_diagnosis)    


# Lung Cancer prediction page    
elif(selected=="Lung Cancer Prediction"):
    st.title("Lung Cancer Prediction using ML")

    col1, col2,col3 = st.columns(3)
    with col1:
        Gender=st.text_input("Gender ")
    with col2:
        age=st.text_input("age of a person ")
    with col3:
        Smoking=st.text_input("Smoking Value ")
    with col1:
        Yellow_fingers=st.text_input("Yellow_fingers Value ")
    with col2:
        Anxiety=st.text_input("Anxiety level ")        
    with col3:
        Peer_Pressure=st.text_input("Peer_Pressure value ")
    with col1:
        Chronic_Disease=st.text_input("Chronic_Disease Value ")
    with col2:
        Fatigue=st.text_input("Fatigue ")
    with col3:
        Allergy=st.text_input("Allergy value ")
    with col1:
        Wheezing=st.text_input("Wheezing Value ")
    with col2:
        Alcohol_Cosumption=st.text_input("Alcohol_Cosumption by a  person ")
    with col3:
        Coughing=st.text_input("Coughing ")
    with col1:
        Shortness_of_Breath=st.text_input("Shortness_of_Breath Value ")
    with col2:
        Swallowing_Difficulty=st.text_input("Swallowing_Difficulty to a  person ")
    with col3:
        ChestPain=st.text_input("ChestPain value ")

    lung_diagnosis=''
    if st.button("Predict Lung Cancer"):
        try:
            user_input = np.asarray([[float(Gender),float(age),float(Smoking),float(Yellow_fingers),float(Anxiety)
                                        ,float(Peer_Pressure),float(Chronic_Disease),float(Fatigue),float(Allergy),float(Wheezing)
                                        ,float(Alcohol_Cosumption),float(Coughing),float(Shortness_of_Breath),float(Swallowing_Difficulty)
                                        ,float(ChestPain)]])
            
        except ValueError:
            st.error("Please enter valid input values")
            lung_cancer_predict=[-1]

        user_input_scaled=scaler2.transform(user_input)
        lung_cancer_predict=lungcancer_model.predict(user_input_scaled)
        print(lung_cancer_predict[0])
        if( lung_cancer_predict[0]==1):
            lung_diagnosis="You are likely to have Lung Cancer"
        elif(lung_cancer_predict[0]==-1):
            lung_diagnosis="Please enter valid input values"
        else:
            lung_diagnosis="You are not likely to have Lung Cancer"
    st.success(lung_diagnosis)

# Parkinson's Disease prediction page    
elif(selected=="Parkinson's Disease Prediction"):
    st.title("Parkinson's Disease Prediction using ML")

    col1,col2,col3=st.columns(3)

    with col1:
        fo=st.text_input("MDVP Fo(hz)")
    with col2:
        fhi=st.text_input("MDVP_Fhi(hz)")
    with col3:
        flo=st.text_input("MDVP_Flo(hz) ")
    with col1:
        Jitterper=st.text_input("Jitter(%) ")
    with col2:
        abs=st.text_input("Jitter(abs) ")
    with col3:
        rap=st.text_input("MDVP(RAP) ")    
    with col1:
        ppq=st.text_input("MDVP(PPQ) ")
    with col2:
        ddp=st.text_input("Jitter(DDP) ")
    with col3:
        shimmer=st.text_input("MDVP(Shimmer) ")
    with col1:
        shimmerdb=st.text_input("MDVP(Shimmer)(DB) ")
    with col2:
        apq3=st.text_input("APQ3 ")
    with col3:
        apq5=st.text_input("APQ5 ")
    with col1:
        apq=st.text_input("MDVP(APQ) ")
    with col2:
        dda=st.text_input("Shimmer(DDA) ")
    with col3:
        nhr=st.text_input("NHR ")
    with col1:
        hnr=st.text_input("HNR ")
    with col2:
        rpde=st.text_input("RPDE ")
    with col3:
        dfa=st.text_input("DFA ")
    with col1:
        spread1=st.text_input("Spread1 ")
    with col2:
        spread2=st.text_input("Spread2 ")
    with col3:
        d2=st.text_input("D2 ")
    with col1:
        ppe=st.text_input("PPE ")

    parkinsons_diagnosis=''
    if st.button('Predict'):
        try:
            user_input=np.array([[float(fo),float(fhi),float(flo),float(Jitterper),float(abs),float(rap),float(ppq),float(ddp),float(shimmer),float(shimmerdb),float(apq3),float(apq5),
                                  float(apq),float(dda),float(nhr),float(hnr),float(rpde),float(dfa),
                                  float(spread1),float(spread2),float(d2),float(ppe)
                                  ]])
        except:
            st.error("Please fill all the fields")
            parkinsons_predict=[-1]

        user_input_scaled=scaler4.transform(user_input)

        parkinsons_predict=parkinsons_model.predict(user_input_scaled)
        print(parkinsons_predict[0])

        if(parkinsons_predict[0]==1):
            parkinsons_diagnosis='Patient has Parkinson\'s disease'
        elif (parkinsons_predict[0]==-1):
            parkinsons_diagnosis='Unable to predict'
        else:
            parkinsons_diagnosis='Patient does not have Parkinson\'s disease'
        st.success(parkinsons_diagnosis)





# Heart Disease  prediction page    
elif(selected=="Heart Disease Prediction"):
    st.title("Heart Disease Prediction using ML")

    col1, col2,col3 = st.columns(3)
    with col1:
        age=st.text_input("Age of the person ")
    with col2:
        sex=st.text_input("Sex ")
    with col3:
        cp=st.text_input("Chest pain type")
    with col1:
        trestbps=st.text_input("resting blood pressure Value ")
    with col2:
        chol=st.text_input("serum cholestoral in mg/dl ")        
    with col3:
        fbs=st.text_input("fasting blood sugar value >120 mg/dl")
    with col1:
        restecg=st.text_input("restecg Value ")
    with col2:
        thalach=st.text_input("thalach of the person ")
    with col3:
        exang=st.text_input("exercise induced angina")
    with col1:
        oldpeak=st.text_input("oldpeak Value ")
    with col2:
        slope=st.text_input("Slope of peak exercise st segment")
    with col3:
        ca=st.text_input("ca value ")
    with col1:
        thal=st.text_input("thal Value ")


    heart_diagonis=''
    if(st.button("Predict")):
        try:
            user_input=np.asarray([[float(age),float(sex),float(cp),float(trestbps),float(chol),float(fbs),float(restecg),float(thalach),float(exang),float(oldpeak),float(slope),float(ca),float(thal)]])
        except ValueError:
            st.error("Invalid input")
            heart_predict=[-1]

        user_input_scaler=scaler3.transform(user_input)
        heart_predict=heartdisease_model.predict(user_input_scaler)

        if(heart_predict[0]==1):
            heart_diagonis="Heart Disease"
        elif (heart_predict[0]==-1):
            heart_diagonis="Error!!"
        else:
            heart_diagonis="No Heart Disease"
    st.success(heart_diagonis)


elif (selected == "Breast Cancer Prediction"):
    st.title("Breast Cancer Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        mean_radius = st.text_input("Mean Radius")
    with col2:
        mean_texture = st.text_input("Mean Texture")
    with col3:
        mean_perimeter = st.text_input("Mean Perimeter")
    with col1:
        mean_area = st.text_input("Mean Area")
    with col2:
        mean_smoothness = st.text_input("Mean Smoothness")        
    with col3:
        mean_compactness = st.text_input("Mean Compactness")
    with col1:
        mean_concavity = st.text_input("Mean Concavity")
    with col2:
        mean_concave_points = st.text_input("Mean Concave Points")
    with col3:
        mean_symmetry = st.text_input("Mean Symmetry")
    with col1:
        mean_fractal_dimension = st.text_input("Mean Fractal Dimension")
    with col2:
        radius_error = st.text_input("Radius Error") 
    with col3:
        texture_error = st.text_input("Texture Error")
    with col1:
        perimeter_error = st.text_input("Perimeter Error")
    with col2:
        area_error = st.text_input("Area Error")
    with col3:
        smoothness_error = st.text_input("Smoothness Error")
    with col1:
        compactness_error = st.text_input("Compactness Error")
    with col2:
        concavity_error = st.text_input("Concavity Error")
    with col3:
        concave_points_error = st.text_input("Concave Points Error")
    with col1:
        symmetry_error = st.text_input("Symmetry Error")
    with col2:
        fractal_dimension_error = st.text_input("Fractal Dimension Error")
    with col3:
        worst_radius = st.text_input("Worst Radius")
    with col1:
        worst_texture = st.text_input("Worst Texture")
    with col2:
        worst_perimeter = st.text_input("Worst Perimeter")
    with col3:
        worst_area = st.text_input("Worst Area")
    with col1:
        worst_smoothness = st.text_input("Worst Smoothness")
    with col2:
        worst_compactness = st.text_input("Worst Compactness")
    with col3:
        worst_concavity = st.text_input("Worst Concavity")
    with col1:
        worst_concave_points = st.text_input("Worst Concave Points")
    with col2:
        worst_symmetry = st.text_input("Worst Symmetry")
    with col3:
        worst_fractal_dimension = st.text_input("Worst Fractal Dimension")
        
    cancer_diagnosis = ''
    if(st.button('Predict')):
        try:
            user_input = np.array([[float(mean_radius), float(mean_texture), float(mean_perimeter), float(mean_area), 
                                    float(mean_smoothness), float(mean_compactness), float(mean_concavity), float(mean_concave_points), 
                                    float(mean_symmetry), float(mean_fractal_dimension), float(radius_error), float(texture_error), 
                                    float(perimeter_error), float(area_error), float(smoothness_error), float(compactness_error), 
                                    float(concavity_error), float(concave_points_error), float(symmetry_error), float(fractal_dimension_error), 
                                    float(worst_radius), float(worst_texture), float(worst_perimeter), float(worst_area), 
                                    float(worst_smoothness), float(worst_compactness), float(worst_concavity), float(worst_concave_points), 
                                    float(worst_symmetry), float(worst_fractal_dimension)]])

        except ValueError:
            st.error("Invalid input!!")
            cancer_predict = [-1]
        
        
        cancer_prediction = breast_model.predict(user_input)
        if(cancer_prediction[0] == 0):
            cancer_diagnosis = "The person has breast cancer"
        elif(cancer_prediction[0] == -1):
            cancer_diagnosis = "Invalid input!!"
        else:
            cancer_diagnosis = "The person does not have breast cancer"
        st.success(cancer_diagnosis)
