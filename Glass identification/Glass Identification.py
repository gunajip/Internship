#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[6]:


df= pd.read_csv('E:/zomato Power BI/archive/glass.csv')


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[15]:


df.columns


# In[17]:


df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(6, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[24]:


df.rename(columns={'Type of glass':'Type'},inplace=True)


# In[25]:


sns.pairplot(df, hue='Type')
plt.show()


# In[28]:


plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 4, i)
    sns.boxplot(y=df[column])
plt.tight_layout()
plt.show()


# In[30]:


plt.figure(figsize=(3, 3))
sns.countplot(df['Type'])
plt.title('Distribution of Glass Types')
plt.show()


# In[34]:


plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 4, i)
    sns.violinplot(x='Type', y=column, data=df)
plt.tight_layout()
plt.show()


# In[32]:


plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 4, i)
    sns.kdeplot(df[column], shade=True)
plt.tight_layout()
plt.show()


# In[37]:


plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 4, i)
    sns.scatterplot(x=column, y='Type', data=df)
plt.tight_layout()
plt.show()


# In[38]:


correlation_with_target = df.corr()['Type'].sort_values(ascending=False)
print(correlation_with_target)


# In[40]:


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr()[['Type']].sort_values(by='Type', ascending=False), annot=True, cmap='coolwarm')
plt.show()


# In[41]:


X = df.drop(columns=['Type'])
y = df['Type']


# In[45]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[52]:


model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)


# In[51]:


y_pred = model.predict(X_test)


# In[53]:


print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))


# In[ ]:




