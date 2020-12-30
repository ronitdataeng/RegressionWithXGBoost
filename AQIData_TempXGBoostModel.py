
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn import preprocessing,model_selection
from scipy.stats import pearsonr,zscore
import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV


# In[3]:


trn_data= pd.read_excel(r"C:\Users\RONIT\Downloads\AirQualityUCI\AirQualityUCI.xlsx")


# In[4]:


trn_data.describe()


# In[5]:


# missing values handle, removing feature columns which have more than 50% of missing values 
missing_columns = pd.DataFrame(trn_data.isnull().sum())
missing_columns.columns = ['Missing']
missing_columns.sort_values(by=['Missing'],ascending=False, inplace=True)
missing_columns = missing_columns[missing_columns.Missing>0]
missingList=missing_columns.index.tolist()
for col in missingList:
    print('{} : {:.2f} %.'.format(col, trn_data[col].isnull().sum()/len(trn_data)*100))


# In[6]:


# calculate Pearson's correlation, remove the features which have 
# less than +/-0.5 correlation with the dependent varriable
numeric_col=trn_data.select_dtypes(include=[np.number]).columns.tolist()
numeric_data=trn_data[numeric_col]
col_drop=[]
for cols in numeric_col:
    corr, _ = pearsonr(numeric_data[[cols]], numeric_data[['T']])
    if np.round(np.abs(corr[0]),2)<0.5:
        col_drop.append(cols)
trn_data.drop(columns=col_drop,inplace=True)        


# In[7]:


# calculate the Z value and remove data where Z<+/-3
z_thresh = 3
constrains = trn_data.select_dtypes(include=[np.number])         .apply(lambda x: np.abs(zscore(x)) < z_thresh, reduce=False)         .all(axis=1)
# Drop (inplace) values set to be rejected
trn_data.drop(trn_data.index[~constrains], inplace=True)


# In[8]:


# extracting and restructuring Date and Time features 
trn_data['year'] =   pd.DatetimeIndex(trn_data['Date']).year
trn_data['month'] = pd.DatetimeIndex(trn_data['Date']).month
trn_data['Time'] = trn_data['Time'].astype(str)
trn_data['Hour'] =  trn_data['Time'].str.split(':').str[0]
trn_data['Hour'] = trn_data['Hour'].astype(int)


# In[9]:


# spliting the data into train and test set
trn_data.drop(columns=['Date','Time'],inplace=True)
x_train,x_test,y_train,y_test=model_selection.train_test_split(trn_data.drop(columns=['T']),trn_data[['T']],test_size=0.2)


# In[10]:


x_train.head()


# In[11]:


# creating an xtreame gradient boost regression model
regressor=xgb.XGBRegressor(gamma=0,max_depth=10, 
                           n_estimators=100,reg_alpha=0, 
                           reg_lambda=1,objective='reg:squarederror'
                          ,subsample=0.5)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


# In[12]:


print('MSE:',mean_squared_error(y_true=np.array(y_test), y_pred=y_pred))
print("R2 score:",r2_score(y_true=y_test, y_pred=y_pred))

