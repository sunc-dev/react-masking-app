# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:58:49 2021

@author: csun
"""

import sys
import pandas as pd
import json
import os
import numpy as np
from operator import itemgetter
model_components = r'model_components'

#For local add - directory to PATH for custom module imports
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),model_components))


#custom module imports
from fetcher import FetchData
from preprocessor import Prepper
from model import modeling, featureImportance



#initial explorers
prep = Prepper()

class Model():
    
    def load_data(self):
        data = 'data_masked.csv'
        df, root, folder = FetchData().get_data(data)
    
        return df, root, folder

    
    def get_variables(self, dataframe):
        unique_cols, col_list, features_categorical, features_unique, target = prep.uniques(dataframe)
        
        features, features_numerical = prep.getFeatures(dataframe, features_categorical, target)
        
        fair_features = prep.getFairFeatures(dataframe, features_categorical, features_unique)
        
        return unique_cols, col_list, features_categorical, features_unique, target, features, features_numerical, fair_features
   

    def get_impact(self, dataframe, fairFeatures):
        disp_impact = prep.disparateImpact(dataframe, fairFeatures)
        return disp_impact
            
    def run_model(self, dataframe, features, target, featuresNumerical, featuresCategory, uniqueFeatures ):
        
        df_model = dataframe[[x for x in features if x not in uniqueFeatures]]
        
        feat_im, feat, result = modeling(df_model, target, featuresNumerical, featuresCategory, uniqueFeatures)
 
        #convert to float64
        feat_im = [float(v) for v in feat_im]

        feature_importance =[]
        for name,value in zip(feat,feat_im):
            
            feat_dict = {'column': name,
                        'value':round(value,2)
                  }
            feature_importance.append(feat_dict)
      
        return  result, feature_importance
    
    
    
    def get_response(self, dataframe, columns, dis_impact_result,feature_importance, result, name):

        if name == 'df':
            modelName = "Vanilla"
            
        elif name =="df_masked":
            modelName = "Masked"
        else:
            modelName = ''
            
        dis_impact_result = sorted(dis_impact_result, key=itemgetter('value')) 
        feature_importance = sorted(feature_importance, key=itemgetter('column')) 
        result = sorted(result, key=itemgetter('measure')) 



            
        
        body = {
                "modelName": modelName,
                "uniqueResult": columns,
                 "disparateResult":dis_impact_result,
                 "importance": feature_importance,
                 "modelResult": result,
                 }        
        
  
        
        return body
    

if __name__ == '__main__':
    
    predictor = Model()
    df, root, folder = predictor.load_data()
    unique_cols, col_list, features_categorical, features_unique, target, features, features_numerical, fair_features = predictor.get_variables(df)
    disp_result, columns, colvalues  = predictor.get_impact(df,fair_features)
    result, feature_importance =  predictor.run_model(df,features, target,features_numerical, features_categorical,features_unique)
    response = predictor.get_response(df, col_list, columns,feature_importance,result, 'df')

