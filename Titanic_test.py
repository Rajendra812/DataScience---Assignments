import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

# Define the user input function
def user_input_features():
    Pclass = st.selectbox('Pclass', [1, 2, 3])
    Age = st.slider('Age', 0, 80, 30)
    SibSp = st.slider('SibSp', 0, 8, 0)
    Parch = st.slider('Parch', 0, 6, 0)
    Fare = st.slider('Fare', 0, 500, 35)
    Sex_male = st.selectbox('Sex', ['male', 'female']) == 'male'
    Embarked_Q = st.selectbox('Embarked', ['Q', 'S', 'C']) == 'Q'
    Embarked_S = st.selectbox('Embarked', ['S', 'Q', 'C']) == 'S'
    
    data = {
        'Pclass': Pclass,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Sex_male': Sex_male,
        'Embarked_Q': Embarked_Q,
        'Embarked_S': Embarked_S
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Make predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
survival = 'Survived' if prediction[0] else 'Not Survived'
st.write(survival)

st.subheader('Prediction Probability')
st.write(prediction_proba)