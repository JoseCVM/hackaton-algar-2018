# Load All Libraries
from __future__ import print_function
from collections import Counter
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import random
import re
import warnings
warnings.filterwarnings("ignore")

train_filename = 'train.csv'
test_filename = 'test.csv'
train_df = pd.read_csv(train_filename, header=0)
test_df = pd.read_csv(test_filename, header=0)
cols=train_df.columns
train_df['source']='train'
test_df['source']='test'
data = pd.concat([train_df, test_df],ignore_index=True)

data = data.fillna(0)
    
train_df = data.loc[data['source']=="train"]
test_df = data.loc[data['source']=="test"]

train_target = np.ravel(np.array(train_df['Churn'].values))
train_df = train_df.drop(['Churn'],axis=1)

float_columns=[]
cat_columns=[]
int_columns=[]
    
for i in train_df.columns:
    if i not in cols:
        continue
    if train_df[i].dtype == 'float' : 
        float_columns.append(i)
    elif train_df[i].dtype == 'int64' or train_df[i].dtype == 'int32' or train_df[i].dtype == 'int16' or train_df[i].dtype == 'int8':
        int_columns.append(i)
    elif train_df[i].dtype == 'object':
        cat_columns.append(i)
          
train_cat_features = train_df[cat_columns]
train_float_features = train_df[float_columns]
train_int_features = train_df[int_columns]

train_cat_features_ver2 = train_cat_features.apply(LabelEncoder().fit_transform)
    
temp_1 = np.concatenate((train_cat_features_ver2,train_float_features),axis=1)
train_transformed_features = np.concatenate((temp_1,train_int_features),axis=1)
train_transformed_features = pd.DataFrame(data=train_transformed_features)
    
array = train_transformed_features.values
number_of_features = len(array[0])
X = array[:,0:number_of_features]
Y = train_target

validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

scoring = 'accuracy'

# Model 1 - Logisitic Regression
model_logreg = LogisticRegression()
model_logreg.fit(X_train, Y_train)
#print("LR: " + str(accuracy_score(Y_validation, model_logreg.predict(X_validation))))

# Model 2 - RandomForest Classifier
#model_rf = RandomForestClassifier()
#model_rf.fit(X_train, Y_train)
#print("RF: " + str(accuracy_score(Y_validation, model_rf.predict(X_validation))))

# Model 3 - XGB Classifier
#model_xgb = XGBClassifier()
#model_xgb.fit(X_train, Y_train)
#print("XGB: " + str(accuracy_score(Y_validation, model_xgb.predict(X_validation))))

#model_logreg = LogisticRegression()
#model_logreg.fit(X, Y)

#model_rf = RandomForestClassifier()
#model_rf.fit(X, Y)

#model_xgb = XGBClassifier()
#model_xgb.fit(X, Y)

# LIME SECTION

predict_fn_logreg = lambda x: model_logreg.predict_proba(x).astype(float)

# Line-up the feature names
feature_names_cat = list(train_cat_features_ver2)
feature_names_float = list(train_float_features)
feature_names_int = list(train_int_features)

feature_names = sum([feature_names_cat, feature_names_float, feature_names_int], [])
#print(feature_names)

# Create the LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train ,feature_names = feature_names,class_names=['Stay','Leave'],
                                                   categorical_features=cat_columns, 
                                                   categorical_names=feature_names_cat, kernel_width=2)

# Pick the observation in the validation set for which explanation is required

# Get the explanation for Logistic Regression
k = 0
procn = 0
procp = 0
positive = []
dp = dict()
dn = dict()
negative = []
examples = 10
while k < len(X_validation) and (procn < examples or procp < examples):
    #n = random.randint(0,len(X_validation)-1)
    k+=1
    exp = explainer.explain_instance(X_validation[k], predict_fn_logreg, num_features=6)
    #print(exp.as_list())
    if model_logreg.predict(X_validation[k].reshape(1,-1)) == 'Yes':    
        for i in range(2):        
            s = re.sub("[^a-zA-Z]+","",exp.as_list()[i][0])    
            if s in feature_names_cat:
                if exp.as_list()[i][1] > 0:    
                    final = s+" estado: "+str(data[s][k])
                    if final not in dn:                       
                        dp[final] = exp.as_list()[i][1]
                    else:
                        dp[final] = dp.get(final) +  exp.as_list()[i][1]
                    positive.append(s+" estado: "+str(data[s][k]))              
            else:
                if exp.as_list()[i][1] > 0:                
                    final = s+" estado: "+str(exp.as_list()[i][0])
                    if final not in dp:                       
                        dp[final] = exp.as_list()[i][1]
                    else:
                        dp[final] = dp.get(final) +  exp.as_list()[i][1]
                    positive.append(final)      
        procp+=1
    else:
        for i in range(2):
            s = re.sub("[^a-zA-Z]+","",exp.as_list()[i][0])
            if s in feature_names_cat:
                if exp.as_list()[i][1] < 0:    
                    final = s+" estado: "+str(data[s][k])
                    if final not in dn:                       
                        dn[final] = exp.as_list()[i][1]
                    else:
                        dn[final] = dn.get(final) +  exp.as_list()[i][1]
                    negative.append(s+" estado: "+str(data[s][k]))              
            else:
                if exp.as_list()[i][1] < 0:                
                    final = s+" estado: "+str(exp.as_list()[i][0])
                    if final not in dn:                       
                        dn[final] = exp.as_list()[i][1]
                    else:
                        dn[final] = dn.get(final) +  exp.as_list()[i][1]
                    negative.append(final)       
        procn+=1
    
    #print(exp.as_list())    
    
print("Fez o cliente sair: ")
for key,value in Counter(positive).most_common(3):
    print(str(key)+", surgiu "+str(value)+" vezes, puxando a probabilidade para essa classe em "+str("{:.2f}".format(abs(dp[key]/value) * 100))+"%")
print("Fez o cliente ficar: ")
for key,value in Counter(negative).most_common(3):
    print(str(key)+", surgiu "+str(value)+" vezes"+" vezes, puxando a probabilidade para essa classe em "+str("{:.2f}".format(abs(dn[key]/value) * 100))+"%")
