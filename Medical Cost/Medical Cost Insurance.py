#!/usr/bin/env python
# coding: utf-8

# # MEDICAL COST INSURANCE

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[24]:


file_path = r"D:\Medical Cost Insurance\medical_insurance.csv"
insurance_df = pd.read_csv(file_path)


# In[25]:


print(insurance_df.head())
print(insurance_df.info())


# In[27]:


plt.figure(figsize=(2, 2))
sns.countplot(x='sex', data=insurance_df)
plt.title('Distribution of Sex')
plt.show()


# In[28]:


plt.figure(figsize=(2, 2))
sns.histplot(insurance_df['age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.show()


# In[29]:


plt.figure(figsize=(2, 2))
sns.histplot(insurance_df['bmi'], bins=20, kde=True)
plt.title('Distribution of BMI')
plt.show()


# In[30]:


plt.figure(figsize=(2, 2))
sns.boxplot(x='smoker', y='charges', data=insurance_df)
plt.title('Smoker vs Charges')
plt.show()


# In[31]:


X = insurance_df.drop(columns=['charges'])
y = insurance_df['charges']


# In[32]:


cat_cols = ['sex', 'smoker', 'region']
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[34]:


scaler = StandardScaler()
num_cols = ['age', 'bmi']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# In[35]:


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)


# In[36]:


y_pred = rf_regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# In[37]:


print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:




