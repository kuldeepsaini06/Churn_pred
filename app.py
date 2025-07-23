import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

# title of the app
st.title("Simple Streamlit App with TensorFlow")

# load standard scaler, label encoder, and one-hot encoder
with open('label_ecoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('standard_scaler.pkl', 'rb') as file:
    standard_scaler = pickle.load(file)

# load the model
model = tf.keras.models.load_model('model.h5')

# heder for input
st.header("Churn Prediction Input")
# User inputs

geography = st.selectbox("Geography",one_hot_encoder.categories_[0].tolist())
gender= st.selectbox("Gender",label_encoder.classes_)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, value=1000.0)
estimated_salary = st.number_input("Estimated Salary")
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# prepare the input data

#  order match with model input 
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode categorical features
geo_encoded = one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

# concat data

input_data = pd.concat([input_data, geo_encoded_df], axis=1)

scaled_features = standard_scaler.transform(input_data)

prediction = model.predict(scaled_features)

if prediction[0][0]> 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

st.write("Prediction Result:",prediction[0][0])