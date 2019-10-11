#!/usr/bin/env python
# coding: utf-8

# In[241]:


#Import necessary variables and dataset from NBA shot logs from 2015 season
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

dataset=pd.read_csv(r'C:\Users\Abdul\Desktop\Fall 2019\Job Search\NBA\shot_logs.csv')
dataset.head()


# In[242]:


#Exploring dataset
dataset['FGM'].value_counts()
sns.countplot(x='FGM',data=dataset, palette='hls')
dataset.groupby(y).mean()


# In[243]:


#Describe the dataset
dataset.describe()


# In[244]:


#Encode labels
le = LabelEncoder()
dataset['PERIOD'] = le.fit_transform(dataset['PERIOD'])
dataset['GAME_CLOCK'] = le.fit_transform(dataset['GAME_CLOCK'])
dataset['LOCATION'] = le.fit_transform(dataset['LOCATION'])
dataset.head()


# In[245]:


#Begin feature engineering by estabilishing variable of interest, selecting features and putting them in X
y = dataset['FGM']

features = ['GAME_CLOCK','DRIBBLES','SHOT_DIST','CLOSE_DEF_DIST','LOCATION']

X=dataset[features]

dataset.sample(5)


# In[246]:


#Scale, fit, and transform the data to put into a consistent form
s = StandardScaler()
X = s.fit_transform(X)


# In[247]:


#Split the dataset into a training and testing grouping
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)


# In[248]:


#Fit the trained values to a LogisticRegression and predict the test values
log = LogisticRegression()
log.fit(X_train,y_train)

y_pred = log.predict(X)
y_prob = log.predict_proba(X)


# In[254]:


#Put the prediction and probabilities back into model and print results
dataset['predictions'] = y_pred
dataset['probability of making shot'] = y_prob[:,1]
print("Model coefficients =")
print(temp.coef_[0])
print("Model Accuracy =") 
print(metrics.accuracy_score(y,y_pred))


# In[255]:


#Check the null score
nullscore = max(y_test.mean(), 1-y_test.mean())
print(nullscore)


# In[257]:


#Confusion matrix to show the model accuracy
metrics.confusion_matrix(y,y_pred)


# In[266]:


#Look at model results in summary page
import statsmodels.api as sm
Stats_model=sm.Logit(y,X)
result=Stats_model.fit()
print(result.summary2())


# In[271]:




