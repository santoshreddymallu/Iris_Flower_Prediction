import os
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Get directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Full path to model
MODEL_PATH = os.path.join(BASE_DIR, "iris.pkl")

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()
    
#Title of app

st.title("🌸Iris Flower Predction App")
st.markdown("This app predicts the species of flower")

st.divider()

# code for creating user inputs
st.header("Enter the Flower measurements in cms")

col1,col2=st.columns(2)
with col1:
    sepal_length=st.number_input("Sepal_length",min_value=0.0,value=5.2,step=0.1)
    petal_length=st.number_input("Petal_length",min_value=0.0,value=1.8,step=0.1)

with col2:
    sepal_width=st.number_input("Sepal_width",min_value=0.0,value=5.4,step=0.1)
    petal_width=st.number_input("Petal_width",min_value=0.0,value=4.2,step=0.1)
    
#code to create a predict button
if st.button("Predict Specices",use_container_width=True):
    input_data=np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction=model.predict(input_data)
    #[0],[1],[2]
    iris_species_map={0: "Setos", 1: "Versicolor", 2: "Virginica"}
    predicted_species=iris_species_map[prediction[0]]
    
    #display predction in highlight box
    st.success(f"The predicted Species is: **{predicted_species}**")
    
    if predicted_species=="Setosa":
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg", caption="Iris_Setosa")
    elif predicted_species=="Versicolor":
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/1024px-Iris_versicolor_3.jpg", caption="Iris_Versicolor")
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/1024px-Iris_virginica.jpg", caption="Iris_Virginica")
    