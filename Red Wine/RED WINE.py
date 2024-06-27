#!/usr/bin/env python
# coding: utf-8

# # RED WINE

# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[82]:


wine_df = pd.read_csv("D:/RedWine/winequality-red.csv")


# In[83]:


print("Dataset Shape:", wine_df.shape)
print(wine_df.info())


# In[84]:


print(wine_df.describe())
print(wine_df.head())


# In[85]:


plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=wine_df)
plt.title('Distribution of Wine Quality')
plt.show()


# In[90]:


features = wine_df.columns[:-1]
fig, axes = plt.subplots(3, 4, figsize=(15, 15))
for i, feature in enumerate(features):
    sns.barplot(x='quality', y=feature, data=wine_df, ax=axes[i//4, i%4])
    axes[i//4, i%4].set_title(f'Quality vs {feature}')
plt.tight_layout()
plt.show()


# In[91]:


wine_df['quality'] = wine_df['quality'].apply(lambda x: 1 if x > 6 else 0)


# In[92]:


X = wine_df.drop('quality', axis=1)
y = wine_df['quality']


# In[93]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[94]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[95]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


# In[96]:


y_pred = rf_classifier.predict(X_test)


# In[97]:


cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
total_predictions = np.sum(cm)
correct_predictions = np.trace(cm)
incorrect_predictions = total_predictions - correct_predictions


# In[98]:


print("Confusion Matrix:")
print(cm)


# In[99]:


print("\nTotal Predictions:", total_predictions)
print("Correct Predictions:", correct_predictions)
print("Incorrect Predictions:", incorrect_predictions)


# In[100]:


print("\nAccuracy Score:")
print(accuracy)


# In[101]:


print("\nAccuracy Percentage:")
print(f"{accuracy * 100:.2f}%")


# In[102]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[103]:


importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns


# In[105]:


plt.figure(figsize=(6, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()


# In[ ]:




