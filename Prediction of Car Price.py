#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[109]:


df = pd.read_csv('car data.csv')
df.head()


# In[110]:


df.shape


# In[111]:


#These 4 are our category functions
#Categorical variables are those which can be further divided into groups. 
print(df['Seller_Type'].unique())
print(df["Transmission"].unique())
print(df["Owner"].unique())
print(df["Fuel_Type"].unique())


# In[112]:


#is null is used to check null values 
df.isnull().sum()


# In[113]:


df.describe()


# In[114]:


df.columns


# In[115]:


final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[116]:


final_dataset.shape


# In[117]:


final_dataset.head()


# In[120]:


final_dataset["Current_Year"] = 2022


# In[121]:


final_dataset.head()


# In[122]:


final_dataset['Number_of_Years'] = final_dataset["Current_Year"] - final_dataset["Year"]


# In[123]:


final_dataset.head()


# In[124]:


final_dataset.drop(["Current_Year"], axis = 1, inplace = True)


# In[125]:


final_dataset.head()


# In[126]:


final_dataset.drop(["Year"], axis = 1, inplace = True)


# In[127]:


final_dataset.head()


# In[128]:


final_dataset= pd.get_dummies(final_dataset, drop_first = True)
#dummy variable trap
# considering columns as n-1
#  y = mx+c
# y  =m1x1 + m2x2 + c
# y = m1x1 +m2(1-x2) + c
# y = m1x1 +m2 - m2x1 + c
# one hot encoding - It converts categorical variable into a bunch of  0s and 1s
#categorical variables are those variables which can be further divided. 


# In[129]:


final_dataset.head()


# In[130]:


#correlation
final_dataset.corr()


# In[131]:


sns.pairplot(final_dataset)
#w3 school


# In[132]:


corrmat=final_dataset.corr() 
top_corr_features=corrmat.index 
plt.figure(figsize=(20,20)) 
#plot heat map
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#w3 school


# In[133]:


final_dataset.head()


# In[134]:


# selling price will be dependent and rest all will be independent features 
X = final_dataset.iloc[:,1:]       #independent
y = final_dataset.iloc[:,0]        #dependent
#w3 school


# In[135]:


X.head()


# In[136]:


y.head()


# In[137]:


#feature importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)


# In[138]:


#plot graph of feature importances for better visualization
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(5).plot(kind='barh')
plt.show()
# feature importance is a technique that calculates a score for all input features.
# The score represents the importance of each feature.
# high score means that the specefic feature will have a larger effect on the model


# In[139]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2) # usual manner to write this
# test_size=?
# train_test_split


# In[140]:


X_train


# In[141]:


y_train


# In[142]:


from sklearn.ensemble import RandomForestRegressor
rf_random  = RandomForestRegressor()


# In[143]:


###hypeparameters
n_estimators = [int(x)for x in np.linspace(start = 100, stop = 1200, num =12)]
print(n_estimators)
# n_estimators are the number of decision trees we  will be running in the model


# In[144]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[145]:


from sklearn.model_selection import RandomizedSearchCV


# In[146]:


# key value pair
random_grid = {'n_estimators':n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf':min_samples_leaf
              }
print(random_grid)
# n_estimators


# In[147]:


#use the random grid to search for best hyperparameters
#first create the base model to tune
rf = RandomForestRegressor()


# In[148]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose = 2, random_state = 42, n_jobs = 1)


# In[149]:


rf_random.fit(X_train, y_train) 


# In[150]:


prediction = rf_random.predict(X_test)


# In[151]:


prediction


# In[152]:


sns.distplot(y_test-prediction)


# In[153]:


plt.scatter(y_test,prediction)

