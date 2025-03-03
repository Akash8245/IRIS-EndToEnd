import streamlit as st
import joblib
import numpy as np

model = joblib.load("IRIS_Model.joblib")
st.title('IRIS Preciction ')
st.text("Choose the values : ")

sepal_length = st.slider("Sepel Length :- ", 0.0, 10.0,key='sepal_length')
sepal_width = st.slider("Sepel Width :- ", 0.0, 10.0,key='sepal_width')
petal_length = st.slider("Petal Length :- ", 0.0, 10.0,key='petal_length')
petal_width = st.slider("Petal Width :- ", 0.0, 10.0,key='petal_width')

predict = st.button("Predict !")

if predict:
    value = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    
    pred_val = model.predict(value)[0]

    text = ''
    if pred_val == 0:
        text = 'setosa'
    elif pred_val == 1:
        text = 'versicolor'
    else:
        text = 'virginica'


    st.subheader(text)

