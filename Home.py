
import streamlit as st   #streamlit is used to build app in quick
from PIL import Image  #used to make changes in the web app
# Loading Image using PIL
im = Image.open('./images/App_Icon.png') 
st.set_page_config(page_title="CardioVascular Disease Predictor App",page_icon=im)
import pandas as pd 
import numpy as np
import pickle
header=st.container()
dataset=st.container()
features=st.container()
model_training=st.container()
model=pickle.load(open("model.pkl",'rb'))
#the below is the front end work
with header:
    st.title("Hi !!!! We're Glad you're here!")
    st.write("This project predicts chances of having CardioVascular disease.The best Machine learning with highest Accuracy is chosen for real time prediction.")
    st.write("In this project three different datasets with different features are used to create best model.")
    st.subheader("The datasets used are:")
    st.subheader("1.Framingham Dataset")
    st.subheader("2.Cleveland Dataset")
    st.subheader("3.Heart Disease Dataset")
    st.text("@Can be found in  UCI repository")
with dataset:
    st.header("This Project Uses various methods to improve accuracy of the Machine learning model")
    st.subheader("1.Feature Selection using chi^2")
    st.subheader("2.SMOTE")

with features:
    st.header("More revelant features more accuracy!!")
    st.text("3 datasets with 14 different features each")
st.sidebar.header("User Input Features")
st.sidebar.markdown(""" [Example CSV input file]""")
uploaded_file=st.sidebar.file_uploader("upload your CSV file",type=["csv"])
if uploaded_file is not None:
    input_df=pd.read_csv(uploaded_file)
    st.subheader("Uploaded dataset")
    st.write(input_df)
    st.subheader("Prediction")
    st.write(model.predict(input_df))
else: 
#this function is called when the user make any changes in the sidebar 
#this function returns the given input from the user
    def user_input():
        sysBP=st.sidebar.slider("sysBP", 133,120,154)
        age=st.sidebar.slider("age",0,25,100)
        totChol=st.sidebar.slider("Chol",200,212,240)
        cigsperDay=st.sidebar.slider("CigsperDay",0,2,50)
        diaBP=st.sidebar.slider("diaBP",71,75,85)
        data={"sysBP":sysBP,
                "age":age,
                "totChol":totChol,
                "cigsPerDay":cigsperDay,
                "diaBP":diaBP}
        features=pd.DataFrame(data,index=[0])
        return features
    df=user_input()
    st.subheader("Inputs given")
    st.write(df)
    data=pd.DataFrame(df)
    print(data)
    st.subheader("Prediction :  ")
#used to load the model for predicting the user values 
    if model.predict(data):
        st.write("Chance of CardioVascular disease")
    else:
        st.write("Less chance for CardioVascular disease")