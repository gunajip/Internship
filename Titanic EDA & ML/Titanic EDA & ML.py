#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:


df = pd.read_csv("D:\Red Wine\Titanic-Dataset.csv")


# In[58]:


df.head()


# In[60]:


df.info()


# In[61]:


df.describe()


# In[62]:


df.sample(5)


# In[63]:


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)


# In[64]:


df.drop(columns=['Cabin'], inplace=True)


# In[65]:


df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})


# In[66]:


df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# In[67]:


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)


# In[68]:


df.head()


# In[70]:


plt.figure(figsize=(3,3))
sns.countplot(x='Survived', data=df)
plt.title('Distribution of Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


# In[88]:


plt.figure(figsize=(4, 4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.show()


# In[74]:


plt.figure(figsize=(4, 3))
sns.countplot(x='Embarked_Q', hue='Survived', data=df)
plt.title('Survival Count by Embarkation Point (Q)')
plt.xlabel('Embarked (Q)')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.show()

plt.figure(figsize=(4, 3))
sns.countplot(x='Embarked_S', hue='Survived', data=df)
plt.title('Survival Count by Embarkation Point (S)')
plt.xlabel('Embarked (S)')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.show()


# In[77]:


plt.figure(figsize=(6, 3))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[78]:


X = df.drop(columns=['Survived'])
y = df['Survived']


# In[79]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[81]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[82]:


classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_scaled, y_train)


# In[83]:


y_pred = classifier.predict(X_test_scaled)


# In[84]:


print(classification_report(y_test, y_pred))


# In[87]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(3, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[85]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[ ]:




