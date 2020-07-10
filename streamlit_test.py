import streamlit as st
import joblib
import pandas as pd

from PIL import Image

model = open("knn_iris.pkl", "rb")
classifier = joblib.load(model)


st.title("IRIS CLASSIFICATION USING KNN")


st.sidebar.title("Choose the features:")

#initializing

parameter_list = ["sepal length", "sepal width", "petal length", "petal width"]
parameter_input = []
parameter_default_values= ["0.0","0.0","0.0","0.0"]

values=[]

#display knn

for p, p_df in zip(parameter_list, parameter_default_values):
    values= st.sidebar.slider(label=p, key=p, value=float(p_df), min_value=0.0, max_value=8.0, step=0.1)
    parameter_input.append(values)
    
input_variables = pd.DataFrame([parameter_input], columns= parameter_list, dtype=float)
st.write("\n\n")


#LOADING IMAGES

setosa= Image.open("setosa.png")
versicolor = Image.open("versicolor.png")
virginica = Image.open("virginica.png")



if st.button("Click to classify-using knn"):
    prediction = classifier.predict(input_variables)
    if prediction==0:
        st.write("Setosa")
        st.image(setosa)
    elif prediction==1:
        st.write("Versicolor")
        st.image(versicolor)
    else:
        st.write("Virginica")
        st.image(virginica)
        
