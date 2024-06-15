#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# LOGISTIC REGRESSION
1. Data Exploration:
a. Load the dataset and perform exploratory data analysis (EDA).
b. Examine the features, their types, and summary statistics.
c. Create visualizations such as histograms, box plots, or pair plots to visualize the distributions and relationships between features.
Analyze any patterns or correlations observed in the data.
2. Data Preprocessing:
a. Handle missing values (e.g., imputation).
b. Encode categorical variables.
3. Model Building:
a. Build a logistic regression model using appropriate libraries (e.g., scikit-learn).
b. Train the model using the training data.
4. Model Evaluation:
a. Evaluate the performance of the model on the testing data using accuracy, precision, recall, F1-score, and ROC-AUC score.
Visualize the ROC curve.
5. Interpretation:
a. Interpret the coefficients of the logistic regression model.
b. Discuss the significance of features in predicting the target variable (survival probability in this case).
6. Deployment with Streamlit:
In this task, you will deploy your logistic regression model using Streamlit. The deployment can be done locally or online via Streamlit Share. Your task includes creating a Streamlit app in Python that involves loading your trained model and setting up user inputs for predictions. 

(optional)For online deployment, use Streamlit Community Cloud, which supports deployment from GitHub repositories. 
Detailed deployment instructions are available in the Streamlit Documentation.


# In[5]:


# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


# In[7]:


# Load the dataset
data=pd.read_csv("E:\Assignment\Logistic Regression\Titanic_train.csv")
data


# In[8]:


data.info()


# In[9]:


# Display summary statistics
data.describe()


# In[12]:


# Check for missing values
print(data.isnull().sum())


# In[13]:


# Visualizations
# Histograms for numerical features
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
data[numerical_features].hist(bins=20, figsize=(15, 10))
plt.show()


# In[14]:


# Box plot for numerical features
plt.figure(figsize=(15, 10))
sns.boxplot(data=data[numerical_features])
plt.show()


# In[15]:


# Count plot for categorical features
categorical_features = ['Survived', 'Pclass', 'Sex', 'Embarked']
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, data=data)
    plt.title(f'Count plot of {feature}')
    plt.show()


# In[16]:


# Pair plot to visualize relationships between numerical features
sns.pairplot(data[numerical_features])
plt.show()


# In[17]:


# Heatmap to visualize correlations between numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(data[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[18]:


# Handle missing values
# Impute missing values for numerical features with mean or median
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)


# In[19]:


# For categorical feature 'Embarked', impute missing values with the mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)


# In[20]:


# Drop 'Cabin' feature due to high number of missing values
data.drop('Cabin', axis=1, inplace=True)


# In[21]:


# Encode categorical variables
# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)


# In[22]:


# Separate features and target variable
X = data.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
y = data['Survived']


# In[23]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


# Initialize logistic regression model
logreg_model = LogisticRegression()


# In[25]:


# Train the model
logreg_model.fit(X_train, y_train)


# In[26]:


# Make predictions on the testing data
y_pred = logreg_model.predict(X_test)


# In[27]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[28]:


# Generate classification report
print(classification_report(y_test, y_pred))


# In[29]:


# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score:", roc_auc)


# In[30]:


# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[31]:


# Coefficients of the logistic regression model
coefficients = pd.DataFrame(logreg_model.coef_[0], index=X.columns, columns=['Coefficient'])
print("Coefficients of the logistic regression model:")
print(coefficients)


# In[32]:


# Discussion of feature significance
# Positive coefficients indicate an increase in the odds of survival with an increase in the corresponding feature,
# while negative coefficients indicate a decrease in the odds of survival.
print("\nSignificance of features in predicting survival probability:")
print("- Features with positive coefficients have a positive impact on survival probability.")
print("- Features with negative coefficients have a negative impact on survival probability.")


# In[33]:


#""""""To run the app, navigate to the directory containing this script in your terminal and run:""""""
#  """"""streamlit run your_script_name.py""""""
#""""""Replace 'your_script_name.py' with the name of the script file containing the Streamlit app code.""""""


# In[34]:


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Function to preprocess input data
def preprocess_input_data(df):
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    
    return df

# Function to train the model
def train_model(df):
    # Separate features and target variable
    X = df.drop(['Survived'], axis=1)
    y = df['Survived']
    
    # Preprocess input data
    X = preprocess_input_data(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize logistic regression model
    logreg_model = LogisticRegression()
    
    # Train the model
    logreg_model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = logreg_model.score(X_test, y_test)
    
    return logreg_model, accuracy

# Streamlit app
def main():
    st.title('Titanic Survival Prediction')
    
    st.sidebar.header('Training the Model')
    
    # Upload dataset
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success('Dataset successfully uploaded.')
        st.sidebar.subheader('Sample of the dataset:')
        st.sidebar.write(df.head())
        
        if st.sidebar.button('Train Model'):
            try:
                model, accuracy = train_model(df)
                st.success(f'Model trained with accuracy: {accuracy:.2f}')
            except Exception as e:
                st.error(f'Error occurred while training the model: {e}')
            
            # Option to save the trained model
            if st.sidebar.button('Save Model'):
                model_path = 'trained_model.pkl'
                joblib.dump(model, model_path)
                st.success(f'Model saved as {model_path}')

    st.sidebar.header('User Input Features')
    
    # Collect user input features
    def collect_user_input():
        sex = st.sidebar.selectbox('Sex', ['male', 'female'])
        age = st.sidebar.slider('Age', 0, 100, 30)
        pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
        sibsp = st.sidebar.slider('Siblings/Spouses Aboard', 0, 10, 0)
        parch = st.sidebar.slider('Parents/Children Aboard', 0, 10, 0)
        fare = st.sidebar.slider('Fare', 0, 100, 10)
        embarked = st.sidebar.selectbox('Embarked', ['C', 'Q', 'S'])
        
        # Create a dictionary with user input
        user_input = {
            'Sex': sex,
            'Age': age,
            'Pclass': pclass,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked
        }
        
        return pd.DataFrame([user_input])
    
    input_df = collect_user_input()
    
    if st.sidebar.button('Predict'):
        try:
            model_path = 'trained_model.pkl'
            model = joblib.load(model_path)
            prediction = model.predict(preprocess_input_data(input_df))
            if prediction[0] == 1:
                st.success('The passenger is predicted to survive!')
            else:
                st.error('The passenger is predicted not to survive.')
            st.dataframe(input_df)
        except Exception as e:
            st.error(f'Error occurred while loading the model: {e}')

if __name__ == '__main__':
    main()


# In[ ]:




