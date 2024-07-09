#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


df = pd.read_csv("D:\Red Wine\World Happiness.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[9]:


plt.figure(figsize=(10, 4))
sns.histplot(df['Happiness Score'], kde=True, bins=20)
plt.title('Distribution of Happiness Scores')
plt.xlabel('Happiness Score')
plt.ylabel('Frequency')
plt.show()


# In[10]:


plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# In[15]:


plt.figure(figsize=(6, 3))
sns.scatterplot(x='Economy (GDP per Capita)', y='Happiness Score', data=df)
plt.title('Happiness Score vs. GDP per Capita')
plt.xlabel('GDP per Capita')
plt.ylabel('Happiness Score')
plt.show()


# In[35]:


plt.figure(figsize=(6, 3))
sns.scatterplot(x='Freedom', y='Happiness Score', data=df)
plt.title('Happiness Score vs. Freedom')
plt.xlabel('Freedom')
plt.ylabel('Happiness Score')
plt.show()


# In[36]:


plt.figure(figsize=(6, 3))
sns.scatterplot(x='Generosity', y='Happiness Score', data=df)
plt.title('Happiness Score vs. Generosity')
plt.xlabel('Generosity')
plt.ylabel('Happiness Score')
plt.show()


# In[21]:


plt.figure(figsize=(6,3))
sns.scatterplot(x='Health (Life Expectancy)', y='Happiness Score', data=df)
plt.title('Happiness Score vs. Health Life Expectancy')
plt.xlabel('Health Life Expectancy')
plt.ylabel('Happiness Score')
plt.show()


# In[22]:


df.drop(columns=['Country', 'Region'], inplace=True)


# In[23]:


df.head()


# In[26]:


X = df.drop(columns=['Happiness Score'])
y = df['Happiness Score']


# In[ ]:


# spliting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[27]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[28]:


regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_scaled, y_train)


# In[29]:


y_pred = regressor.predict(X_test_scaled)


# In[30]:


print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'R^2 Score: {r2_score(y_test, y_pred):.2f}')


# In[33]:


plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Happiness Score')
plt.ylabel('Predicted Happiness Score')
plt.title('Actual vs. Predicted Happiness Scores')
plt.show()

