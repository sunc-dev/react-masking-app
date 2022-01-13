from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler
from collections import defaultdict
from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ShuffleSplit,train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd 
#import streamlit as st 
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def xgbclf(params, X_train, y_train,X_test, y_test):
  
    eval_set=[(X_train, y_train), (X_test, y_test)]
    model = XGBClassifier(**params).\
        fit(X_train, y_train, eval_set=eval_set, \
        eval_metric='auc', early_stopping_rounds = 100, verbose=100)

    model.set_params(**{'n_estimators': model.best_ntree_limit})
    model.fit(X_train, y_train)
    # Predict target variables y for test data
    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit) #model.best_iteration
    # Create and print confusion matrix    
    abclf_cm = confusion_matrix(y_test,y_pred)
    #y_pred = model.predict(X_test)
    validation_indx = {'recall': recall_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred)}
    
    df_valuation = pd.DataFrame.from_dict(validation_indx, orient='index').T
    fig = go.Figure(data=[go.Table(header=dict(values=['recall', 'precision', 'f1', 'accuracy']),
                    cells=dict(values=[round(recall_score(y_test, y_pred), 2),
                                        round(precision_score(y_test, y_pred), 2),
                                        round(f1_score(y_test, y_pred), 2),
                                        round(accuracy_score(y_test, y_pred), 2)]))])


    fig.update_layout(width=500, height=50, margin=dict(l=0, r=0, b=0, t=0))
        
    #st.plotly_chart(fig)

    # Predict probabilities target variables y for test data
    y_pred_proba = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)[:,1] #model.best_iteration
    #get_roc (y_test,y_pred_proba)
    return model, validation_indx

def get_roc (y_test,y_pred):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    #Plot of a ROC curve
    #width = st.sidebar.slider("plot width", 1, 25, 3)
    #height = st.sidebar.slider("plot height", 1, 25, 1)
    #fig, ax = plt.subplots(figsize=(width, height))
    fig, ax = plt.subplots()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.show()
    #st.pyplot(fig)
    return

def featureImportance(dataframe, x, y, color):
    fig = px.line(dataframe, x="features", y="importance", color='file type') #, color='country')
    fig.show()
    #st.pyplot(fig)
    return

params={
    'n_estimators':3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.005,
    #'gamma':0.01,
    'subsample':0.555,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'max_depth':8,
    #'seed':1024,
    'n_jobs' : -1
}

def modeling(df_model, target, numvars, catvars, unvars):
    data = df_model.copy()
# Standardization
    numvars_ = [x for x in numvars if x in df_model.columns]
    numdata_std = pd.DataFrame(StandardScaler().fit_transform\
        (data[numvars_]))

    d = defaultdict(LabelEncoder)
    # Encoding the variable
    catvars_ = [x for x in catvars if x in df_model.columns]
    #lecatdata = data[catvars_].apply(lambda x: d[x.name].fit_transform(x))
    #One hot encoding, create dummy variables for every category of every categorical variable
    dummyvars = pd.get_dummies(data[catvars_])
    data_clean = pd.concat([data[numvars_], dummyvars], axis = 1)

    X_clean = data_clean #.drop([target], axis=1)
    #st.write(X_clean.columns)
    y_clean = data[target]
    X_train_clean, X_test_clean, y_train_clean, y_test_clean \
        = train_test_split(X_clean,y_clean,test_size=0.2, \
        random_state=1)

    model, results = xgbclf(params, X_train_clean, y_train_clean, X_test_clean, y_test_clean)
   
        

    
    result = []
    for key,value in results.items():
        measure = {'measure': key,
                   'value': round(value,2)
             }
        result.append(measure)
         
   
    return list(model.feature_importances_), list(X_clean.columns), result

