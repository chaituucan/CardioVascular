import streamlit as st
import pandas as pd 
import numpy as nm 
st.header("Datasets Details")
st.subheader("1.Framingham Dataset")
data=pd.read_csv("datasets/framingham.csv")
st.dataframe(data)
st.subheader("2.Cleveland dataset")
data=pd.read_csv("datasets/cleveland.csv")
st.dataframe(data)
data=pd.read_csv("datasets/heart.csv")
st.subheader("3.Heart Disease Dataset")
st.dataframe(data)