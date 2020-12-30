import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn import preprocessing,model_selection
from scipy.stats import pearsonr,zscore
import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV


trn_data= pd.read_excel(r"C:\Users\RONIT\Downloads\AirQualityUCI\AirQualityUCI.xls")

def handleMissingValues(trn_data):
    # missing values handle, removing feature columns which have more than 50% of missing values 
    missing_columns = pd.DataFrame(trn_data.isnull().sum())
    missing_columns.columns = ['Missing']
    missing_columns.sort_values(by=['Missing'],ascending=False, inplace=True)
    missing_columns = missing_columns[missing_columns.Missing>0]
    missingList = missing_columns.index.tolist()
    col_to_drp = []
    for col in missingList:
        if (trn_data[col].isnull().sum()/len(trn_data))*100>45:
            col_to_drp.append(col)
    trn_data.drop(col_to_drp)
    return trn_data

def calcPrCorelation(trn_data):
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
    return trn_data 

def removeoutliers_withZscore(trn_data):
    # calculate the Z value and remove data where Z<+/-3
    z_thresh = 3
    constrains = trn_data.select_dtypes(include=[np.number]).apply(lambda x: np.abs(zscore(x)) < z_thresh, reduce=False).all(axis=1)
    # Drop (inplace) values set to be rejected
    trn_data.drop(trn_data.index[~constrains], inplace=True)
    return trn_data

def extractDateTimefeature(trn_data):
    # extracting and restructuring Date and Time features 
    trn_data['year'] =  pd.DatetimeIndex(trn_data['Date']).year
    trn_data['month'] = pd.DatetimeIndex(trn_data['Date']).month
    trn_data['Time'] = trn_data['Time'].astype(str)
    trn_data['Hour'] =  trn_data['Time'].str.split(':').str[0]
    trn_data['Hour'] = trn_data['Hour'].astype(int)
    return trn_data

def traintestspit(trn_data):
    # spliting the data into train and test set
    trn_data.drop(columns=['Date','Time'],inplace=True)
    x_train,x_test,y_train,y_test=model_selection.train_test_split(trn_data.drop(columns=['T']),trn_data[['T']],test_size=0.2)
    return x_train,x_test,y_train,y_test

def XGBoostRegressorModel(x_train,y_train):
    # creating an xtreame gradient boost regression model
    regressor=xgb.XGBRegressor(gamma=0,max_depth=10, 
                           n_estimators=100,reg_alpha=0, 
                           reg_lambda=1,objective='reg:squarederror'
                          ,subsample=0.5)
    regressor.fit(x_train, y_train)
    return regressor

def modelprediction(regressor,x_test,y_test):
    y_pred = regressor.predict(x_test)
    MSE = mean_squared_error(y_true=np.array(y_test), y_pred=y_pred)
    R2Score = r2_score(y_true=y_test, y_pred=y_pred)
    return y_pred,MSE,R2Score

    

