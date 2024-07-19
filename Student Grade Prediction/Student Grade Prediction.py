#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[5]:


df=pd.read_csv("E:/zomato Power BI/archive (1)/student-mat.csv")


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.head()


# In[10]:


df.shape


# In[9]:


df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()


# In[11]:


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()


# In[12]:


sns.pairplot(df)
plt.show()


# In[15]:


sns.pairplot(df[['G1', 'G2', 'G3', 'studytime', 'absences']])
plt.show()


# In[14]:


df.head()


# In[16]:


sns.histplot(df['G3'], kde=True)
plt.show()


# In[18]:


sns.boxplot(x='studytime', y='G3', data=df)
plt.show()


# In[21]:


df.info()


# In[23]:


X = df[['studytime', 'absences', 'G1']]  
y = df['G3'] 


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[25]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[26]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[28]:


y_pred = model.predict(X_test)


# In[29]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2


# In[ ]:




