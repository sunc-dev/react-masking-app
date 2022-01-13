# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 08:27:41 2021

@author: csun
"""
#library imports
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
import pandas as pd
import os
import uuid
model_components = r'model_components'
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),model_components))

#custom module imports
from modeller import Model
from fetcher import FetchData
from preprocessor import Prepper
from model import modeling, featureImportance
from maskfile import maskingFunc
import time


description = """
Masker API applies gradient boosting on a obfuscation dataset. 🚀

## Models

### Vanilla Model endpoint: \n
    train and & test xgboost without obfuscation applied to dataset \n
### Masked Model endpoint: \n
    train & teest xgboost with obfuscation applied to dataset \n
    

### Comparison endpoint: \n
    output of results to compare the models

\n

## Actions: \n
   downloadable EDA report - WIP


"""


#initialize FastAPI
app = FastAPI(
    title="Masker API",
    description = description,
    version="0.0.1",

)

origins = [
    "http://localhost:3000"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#initialize data
#inital data load
data = 'data.csv'
df, root, folder = FetchData().get_data(data)
df_masked = maskingFunc(os.path.join(root,folder), df, 'encryption_keys')

df.name ='df'
df_masked.name="df_masked"
data_list = [df,df_masked]



#initialize actions
prep = Prepper()
modellize = Model()


@app.get("/")
def home():
    return {"Welcome to the MASKER"}


@app.get("/data")
def GetData():
    dataset = df.head(10)
    
    dataset['id'] = [uuid.uuid4() for _ in range(len(dataset.index))]

    dataset = dataset.to_dict('index')
    columns = list(df.columns.values)
    
   
    columnResponse = []
    for col in columns:
        columnItem = {
                    'name': col,
                   'uid': col}
        columnResponse.append(columnItem)

    dataResponse = {
                    'data': dataset,
                    'columns': columnResponse
        }
    
    
    return dataResponse


@app.get("/model")
def Vanilla():
    
    unique_cols, col_list, features_categorical, features_unique, target, features, features_numerical, fair_features = modellize.get_variables(df)
    disp_result = modellize.get_impact(df,fair_features)
    results, feature_importance =  modellize.run_model(df,features, target,features_numerical, features_categorical,features_unique)
    response = modellize.get_response(df, col_list, disp_result,feature_importance,results, df.name)

    return  response


@app.get("/unique")
def Unique():
     
     uniqueResults = []   
     for data in data_list:
         
         
         if data.name == 'df':
             modelName = "Vanilla"
                
         elif data.name =="df_masked":
                modelName = "Masked"
         else:
                modelName = ''
         
         unique_cols,col_list, features_categorical, features_unique, target, features, features_numerical, fair_features = modellize.get_variables(data)
         uniqueRes = {
              "modelName": modelName,
              "uniqueResult": col_list
              }
         
         uniqueResults.append(uniqueRes)
     
     return uniqueResults
     
@app.get("/masker")
def Mask():
    #cleanup required
    unique_cols, col_list, features_categorical, features_unique, target, features, features_numerical, fair_features = modellize.get_variables(df_masked)
    disp_result = modellize.get_impact(df_masked,fair_features)
    results, feature_importance =  modellize.run_model(df_masked,features, target,features_numerical, features_categorical,features_unique)
    response = modellize.get_response(df_masked, col_list, disp_result,feature_importance,results, df_masked.name)

    return  response

@app.get("/compare")
def Compare():
    
    
    modelResult = []
    for data in data_list:
        
        unique_cols,col_list, features_categorical, features_unique, target, features, features_numerical, fair_features = modellize.get_variables(data)
        disp_result, columns, colvalues = modellize.get_impact(data,fair_features)
        results, feature_importance =  modellize.run_model(data,features, target,features_numerical, features_categorical,features_unique)
        
        time.sleep(2)
        response = modellize.get_response(df, col_list, disp_result,feature_importance,results, data.name)
        
        modelResult.append(response)
        
    res = {
            "count": len(data_list),
            "result": modelResult
        
        }
    
        
    return  res


@app.get("/Report")
def getReport():

    return


if __name__ == "__main__":
    uvicorn.run("masker_fastAPI:app")