#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


bikesharing=pd.read_csv('day.csv')


# In[3]:


bikesharing


# # Step 1: Data Preprocessing

# In[4]:


#convert the datatype of dteday column to datetime.

bikesharing['dteday'] =  pd.to_datetime(bikesharing['dteday'],format='%d-%m-%Y')
bikesharing['dteday'].dtype


# In[5]:


#Dropping the redundant variable holiday as the workingday column covers required information.

bikesharing.drop('holiday',axis=1,inplace=True)
bikesharing.head()


# In[6]:


# Dropping few more redundant columns.

bikesharing.drop(['dteday','instant','casual','registered'],axis=1,inplace=True)
bikesharing.head()


# In[7]:


# Renaming some columns for better understanding

bikesharing.rename(columns={'hum':'humidity','cnt':'count','yr':'year','mnth':'month'},inplace=True)
bikesharing.head()


# # Step 2: Encoding the labels and Visualization

# ## 1.Season

# In[8]:


codes = {1:'spring',2:'summer',3:'fall',4:'winter'}
bikesharing['season'] = bikesharing['season'].map(codes)


# In[9]:


bikesharing


# In[10]:


sns.barplot(data=bikesharing, x='season', y='count')
plt.show()


# There are more rentals during Fall season

# ## 2.Weathersit

# In[11]:


codes = {1:'Clear', 2:'Mist', 3:'Light Snow', 4:'Heavy Rain'}
bikesharing['weathersit'] = bikesharing['weathersit'].map(codes)


# In[12]:


bikesharing


# In[13]:


sns.barplot(x='weathersit',y='count',data=bikesharing)
plt.show()


# Clearly more number of people rented when the weather was clear 

# ## 3.Working day 

# In[14]:


codes = {1:'working_day',0:'Holiday'}
bikesharing['workingday'] = bikesharing['workingday'].map(codes)


# In[15]:


bikesharing


# In[16]:


sns.barplot(x='workingday',y='count',data=bikesharing)
plt.show()


# There is no much difference

# ## 4. Year 

# In[17]:


sns.barplot(x='year',y='count',data=bikesharing)


# ## 5 Month 

# In[18]:


codes = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
bikesharing['month'] = bikesharing['month'].map(codes)


# In[19]:


plt.figure(figsize=(12, 6))
sns.barplot(x='month', y='count', hue='year', data=bikesharing)
plt.show()


# ## 6.weekday 

# In[20]:


codes = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
bikesharing['weekday'] = bikesharing['weekday'].map(codes)


# In[21]:


bikesharing.groupby('weekday')['count'].max().plot(kind='bar')


# In[22]:


bikesharing


# In[23]:


X=bikesharing.iloc[:,:-1].values
y=bikesharing.iloc[:,-1].values


# In[24]:


X


# In[25]:


y


# # Step 3: One-Hot Encoding

# In[26]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[27]:


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 2, 3, 4, 5])], remainder='passthrough')


# In[28]:


X = np.array(ct.fit_transform(X))


# In[29]:


X


# In[30]:


# Get the names of the one-hot encoded columns
one_hot_encoder = ct.named_transformers_['encoder']
encoded_column_names = one_hot_encoder.get_feature_names_out()

# Convert the encoded columns to a DataFrame for better visualization
encoded_df = pd.DataFrame(X[:, :len(encoded_column_names)], columns=encoded_column_names)



# In[31]:


# Display the DataFrame
encoded_df.head()


# oneHotEncoding for 'Season' column is X0 and:-
# 
# ![image.png](attachment:image.png)
# 
# 
# that is for fall :- 1000
# 

# oneHotEncoding for 'month' column is X1 and:
# 
# ![image.png](attachment:image.png)

# oneHotEncoding for 'weekday' column is X2 and :-
# 
# 
# ![image.png](attachment:image.png)

# oneHotEncoding for 'workingday' column is X3 and :-
# 
# ![image.png](attachment:image.png)

# oneHotEncoding for 'weathersit' column is X4 and :-
# 
# ![image.png](attachment:image.png)

# # Step 4:- Training and Test dataset

# In[32]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[33]:


X_train


# In[34]:


y_train


# # Step 5 :- Model Training

# In[35]:


regressor=LinearRegression()


# In[36]:


regressor.fit(X_train,y_train)


# In[37]:


y_pred=regressor.predict(X_test)


# In[38]:


y_pred


# In[39]:


y_test


# # Step 6 :- Model Evaluation

# In[40]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[41]:


print(regressor.coef_)


# In[42]:


print(regressor.intercept_)

