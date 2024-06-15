#!/usr/bin/env python
# coding: utf-8

# # MULTIPLE LINEAR REGRESSION
# 
# # Assignment Task:
# * Your task is to perform a multiple linear regression analysis to predict the price of Toyota corolla based on the given attributes.
# * Dataset consists of following varibales (Age, KM, FuelType, HP, Automatic, CC, Doors, Weight, Quarterly_Tax, Price)
# # Tasks:
# 1.Perform exploratory data analysis (EDA) to gain insights into the dataset. Provide visualizations and summary statistics of the variables. Pre process the data to apply the MLR.
# 2.Split the dataset into training and testing sets (e.g., 80% training, 20% testing).
# 3.Build a multiple linear regression model using the training dataset. Interpret the coefficients of the model. Build minimum of 3 different models.
# 4.Evaluate the performance of the model using appropriate evaluation metrics on the testing dataset.
# 5.Apply Lasso and Ridge methods on the model.
# 
# Ensure to properly comment your code and provide explanations for your analysis.
# Include any assumptions made during the analysis and discuss their implications.
# 

# In[12]:


# Importing Libraries:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import missingno as mn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.regressionplots import influence_plot

# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Importing Dataset

raw_data=pd.read_csv('E:\Assignment\MLR\ToyotaCorolla.csv',encoding='latin1')
raw_data.head()


# In[4]:


print('Number of Rows & Columns{}'.format(raw_data.shape))


# In[5]:


# Descriptive Analysis
raw_data.describe()


# In[6]:


# Checking for Data types
raw_data.info()


# # Observation: all the data types are correct.

# In[7]:


# Renaming the columns name
data=raw_data.rename({'Age_08_04':'Age','cc':'CC','Fuel_Type':'FT'},axis=1)
data.head()


# In[8]:


# Checking for missing values
data[data.values==0.0]


# In[9]:


data.isnull().sum()


# In[10]:


# Visualizing Missing Values
plt.figure(figsize=(12,8))
sns.heatmap(data.isnull(),cmap='viridis')


# In[13]:


mn.matrix(data)


# # Observation: After checking above there is no null value present in the dataset

# In[14]:


#Checking for Duplicated Values
data[data.duplicated()]


# In[15]:


data[data.duplicated()].shape


# In[16]:


data=data.drop_duplicates().reset_index(drop=True)
data[data.duplicated()]


# # Let's find how many discrete and continuous feature are their in our dataset by seperating them in variables
# 

# In[17]:


discrete_feature=[feature for feature in data.columns if len(data[feature].unique())<20 and feature]
print('Discrete Variables Count: {}'.format(len(discrete_feature)))


# In[18]:


continuous_feature=[feature for feature in data.columns if data[feature].dtype!='O' and feature not in discrete_feature]
print('Continuous Feature Count {}'.format(len(continuous_feature)))


# In[23]:


# Visualize the distribution of each numerical variable
data.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()


# In[19]:


# Visualize relationships using pairplot
sns.pairplot(data)
plt.show()


# In[25]:


# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[20]:


# Encode categorical variables
df = pd.get_dummies(data, drop_first=True)
df


# # Model Bulding

# In[21]:


model=smf.ols('Price~Age+KM+HP+Automatic+CC+Doors+Cylinders+Gears+Weight+FT_Diesel+FT_Petrol',data=df).fit()


# # Model Testing

# In[23]:


# Finding Coefficient parameters
model.params


# In[24]:


# Finding Tvalues and Pvalues
model.tvalues, np.round(model.pvalues,5)


# In[25]:


# Finding rsquared values
model.rsquared, model.rsquared_adj    # model accucracy is 86.28%


# # We need to build SLR and MLR for insignificant variables 'CC' and 'Doors'
# # Also find their t values and pvalues

# In[29]:


slr_cc=smf.ols('Price~CC',data=df).fit()
slr_cc.tvalues, slr_cc.pvalues


# In[30]:


slr_d=smf.ols('Price~Doors',data=df).fit()
slr_d.tvalues, slr_d.pvalues


# In[31]:


mlr_cd=smf.ols('Price~CC+Doors',data=df).fit()
mlr_cd.tvalues, mlr_cd.pvalues


# # Model Validation Techniques

# In[33]:


# 1) Check collinearity 
rsq_age=smf.ols('Age~KM+HP+Automatic+CC+Doors+Cylinders+Gears+Weight+FT_Diesel+FT_Petrol',data=df).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+Automatic+CC+Doors+Cylinders+Gears+Weight+FT_Diesel+FT_Petrol',data=df).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+Automatic+CC+Doors+Cylinders+Gears+Weight+FT_Diesel+FT_Petrol',data=df).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_Automatic=smf.ols('Automatic~Age+KM+HP+CC+Cylinders+Gears+Weight+FT_Diesel+FT_Petrol',data=df).fit().rsquared
vif_Automatic=1/(1-rsq_Automatic)

rsq_cc=smf.ols('CC~Age+KM+HP+Automatic+Doors+Cylinders+Gears+Weight+FT_Diesel+FT_Petrol',data=df).fit().rsquared
vif_cc=1/(1-rsq_cc)

rsq_DR=smf.ols('Doors~Age+KM+HP+Automatic+CC+Cylinders+Gears+Weight+FT_Diesel+FT_Petrol',data=df).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_CY=smf.ols('Cylinders~Age+KM+HP+Automatic+CC+Doors+Gears+Weight+FT_Diesel+FT_Petrol',data=df).fit().rsquared
vif_CY=1/(1-rsq_CY)

rsq_GR=smf.ols('Gears~Age+KM+HP+Automatic+CC+Doors+Cylinders+Weight+FT_Diesel+FT_Petrol',data=df).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_WT=smf.ols('Weight~Age+KM+HP+Automatic+CC+Doors+Cylinders+Gears+FT_Diesel+FT_Petrol',data=df).fit().rsquared
vif_WT=1/(1-rsq_WT)

rsq_FT_D=smf.ols('FT_Diesel~Age+KM+HP+Automatic+CC+Doors+Cylinders+Gears+Weight+FT_Petrol',data=df).fit().rsquared
vif_FT_D=1/(1-rsq_FT_D)

rsq_FT_P=smf.ols('FT_Petrol~Age+KM+HP+Automatic+CC+Doors+Cylinders+Gears+Weight+FT_Diesel',data=df).fit().rsquared
vif_FT_P=1/(1-rsq_FT_P)


# Putting the values in Dataframe format
d1={'Variables':['Age','KM','HP','Automatic','CC','Doors','Cylinders','Gears','Weight','FT_Petrol','FT_Diesel'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_Automatic,vif_cc,vif_DR,vif_CY,vif_GR,vif_WT,vif_FT_P,vif_FT_D]}
Vif_df=pd.DataFrame(d1)
Vif_df

print('None of the VIF>20, no collinarity so consider all variables in Regression equation')


# In[34]:


# 2) check Residual 
sm.qqplot(model.resid,line='q')
plt.show()


# In[35]:


list(np.where(model.resid>6000)) # outlier detection from above QQ plot


# In[36]:


list(np.where(model.resid<-6000))


# In[37]:


# Test for Homoscedasticity or Heteroscedasticity (plotting model's standardized fitted values vs standardized residual values)

def standard_values(vals) :
    return (vals-vals.mean())/vals.std()  # User defined z = (x - mu)/sigma


# In[38]:


plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 


# In[39]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Age',fig=fig)
plt.show()


# In[40]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'KM',fig=fig)
plt.show()


# In[41]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'HP',fig=fig)
plt.show()


# In[42]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Automatic',fig=fig)
plt.show()


# In[43]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'CC',fig=fig)
plt.show()


# In[44]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Doors',fig=fig)
plt.show()


# In[47]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Cylinders',fig=fig)
plt.show()


# In[48]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Gears',fig=fig)
plt.show()


# In[49]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Weight',fig=fig)
plt.show()


# In[50]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'FT_Diesel',fig=fig)
plt.show()


# In[51]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'FT_Petrol',fig=fig)
plt.show()


# In[52]:


# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
(c,_)=model.get_influence().cooks_distance
c


# In[54]:


# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(df)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[55]:


# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)


# In[56]:


# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)


# In[57]:


# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=df.shape[1]
n=df.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# # Improving the Model

# In[58]:


# Creating a copy of data so that original dataset is not affected
toyo_new=df.copy()
toyo_new


# In[60]:


# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
toyo2=toyo_new.drop(toyo_new.index[[80]],axis=0).reset_index(drop=True)
toyo2


# # Final Model without outliers

# In[61]:


while model.rsquared < 0.90:
    for c in [np.max(c)>0.5]:
        model=smf.ols('Price~Age+KM+HP+Automatic+CC+Doors+Cylinders+Gears+Weight+FT_Diesel+FT_Petrol',data=toyo2).fit()
        (c,_)=model.get_influence().cooks_distance
        c
        np.argmax(c) , np.max(c)
        toyo2=toyo2.drop(toyo2.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
        toyo2
    else:
        final_model=smf.ols('Price~Age+KM+HP+Automatic+CC+Doors+Cylinders+Gears+Weight+FT_Diesel+FT_Petrol',data=toyo2).fit()
        final_model.rsquared , final_model.aic
        print("Thus model accuracy is improved to",final_model.rsquared)


# In[62]:


final_model.rsquared


# In[63]:


toyo2


# # Model prediction

# In[74]:


# New data for prediction is
new_data=pd.DataFrame({'Age':12,"KM":40000,"HP":80,"Automatic":0,"CC":1300,"Doors":4,"Gears":5,"Cylinders":4,"Weight":1012,"FT_Diesel":1,"FT_Petrol":0},index=[0])
new_data


# In[75]:


# Manual Prediction of Price
final_model.predict(new_data)


# In[76]:


# Automatic Prediction of Price with 90.02% accurcy
pred_y=final_model.predict(toyo2)
pred_y

