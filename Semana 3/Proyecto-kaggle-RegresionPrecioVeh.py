import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import LabelEncoder

import os


dataTraining = pd.read_csv('https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/datasets/dataTrain_carListings.zip')
dataTesting = pd.read_csv('https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/datasets/dataTest_carListings.zip', index_col=0)



dataTraining['Model']=dataTraining['Make']+' '+dataTraining['Model']
dataTesting['Model']=dataTesting['Make']+' '+dataTesting['Model']

y=dataTraining.Price
X=dataTraining.drop('Price', axis=1)

#XTOTAL_TRAINyTEST=X.append(dataTesting)
#XTOTAL_TRAINyTEST['Model']=XTOTAL_TRAINyTEST['Make']+"_"+XTOTAL_TRAINyTEST['Model']
#X['State']=X['State'].astype('category')
#X['Make']=X['Make'].astype('category')
#X['Model']=X['Model'].astype('category')

le1=LabelEncoder()
le2=LabelEncoder()
le3=LabelEncoder()
#print(XTOTAL_TRAINyTEST['Model'])
print("le0")
#XTOTAL_TRAINyTEST['NewState']=le1.fit_transform(XTOTAL_TRAINyTEST['State']).astype('int')
#XTOTAL_TRAINyTEST['NewModel']=le2.fit_transform(XTOTAL_TRAINyTEST['Model']).astype('int')
#XTOTAL_TRAINyTEST['NewMake']=le3.fit_transform(XTOTAL_TRAINyTEST['Make']).astype('int')



X['State']=le1.fit_transform(X['State']).astype('int')
X['Model']=le2.fit_transform(X['Model']).astype('int')
X['Make']=le3.fit_transform(X['Make']).astype('int')

#print("le1")
#print(le1.classes_)
#print(XTOTAL_TRAINyTEST.shape)
#print(XTOTAL_TRAINyTEST[['NewState','State']])
print("le2")
print(le2.classes_)
#print(XTOTAL_TRAINyTEST.shape)
#print(XTOTAL_TRAINyTEST[['NewModel','Model']])
#print("le3")
#print(le3.classes_)
#print(XTOTAL_TRAINyTEST.shape)
#print(XTOTAL_TRAINyTEST[['NewMake','Make']])






#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#X = pd.DataFrame(scaler.fit_transform(X),columns = X.columns)
#X.drop(['Mileage','Year'], axis=1, inplace=True)
#print(X)




#X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)
X_train


print("transfom test")
dataTesting['State']=le1.transform(dataTesting['State']).astype('int')
dataTesting['Model']=le2.transform(dataTesting['Model']).astype('int')
dataTesting['Make']=le3.transform(dataTesting['Make']).astype('int')
print("fin transfom train")

#dataTesting = pd.DataFrame(scaler.transform(dataTesting),columns = dataTesting.columns)



#dataTesting=pd.get_dummies(dataTesting, drop_first=True)
dataTesting
print(X_test.shape)
print(dataTesting.shape)

a=set(X_test.columns)
b=set(dataTesting.columns)
if (a == b):
    print('have the same columns')
else:
    print('have different columns')
print('a-b')
print(a-b)  
print('b-a')
print(b-a)
print('a ^ b')
print(a ^ b)

#dataTesting['Make_Freightliner']=0


#dataTraining['Make'].value_counts()
X_train.info()





from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

params = {
    'min_child_weight': [ 1, 3, 5, 7 ],
    'gamma': np.linspace(0, 30, 200),
    'eta': np.linspace(0, 1, 25),
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [ 0.3, 0.4, 0.5 , 0.7 ],
    'max_depth': [ 3, 4, 5, 6, 8, 10, 12, 15]
}

xg_class = XGBRegressor(eval_metric='rmse')
#xg_class = XGBRegressor()

random_search = RandomizedSearchCV(xg_class, param_distributions=params, n_iter=100,  n_jobs=-1, cv=3, verbose=3, random_state=1001,scoring='neg_mean_squared_error')

random_search.fit(X_train.values, y_train.values)
print(random_search.best_estimator_)

joblib.dump(random_search, filename='phishing_clf.pkl', compress=3)

from sklearn.metrics import mean_squared_error
y_predict=random_search.predict(X_test.values)

print("XGBRegressor - Calibrado con RandomizedSearchCV")
print(f'MSE: {mean_squared_error(y_test.values, y_predict, squared=False)}')


y_pred = random_search.predict(dataTesting.values)
y_pred_df=pd.DataFrame(y_pred, columns=['Price'])
y_pred_df.to_csv('test_submission.csv', index_label='ID')
print(y_pred_df.head())

