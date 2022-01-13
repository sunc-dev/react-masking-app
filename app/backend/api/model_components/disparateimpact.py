import pandas as pd 



class initActions():
    
    
    def Uniques(dataframe):
        target = 'Creditability'
        features = [x for x in dataframe.columns] # if x != target]
        features_categorical = [x for x in features if (x not in dataframe._get_numeric_data().columns or len(dataframe[x].unique()) <5) and (x != target)]
        features_numerical = [x for x in features if x != target and x not in features_categorical]
        features_unique = [x for x in features if len(dataframe[x].unique()) == len(dataframe)]
        
        
        df_unique = {} 
        for i,j in enumerate(features_unique): 
            df_unique[i] = j
            
        return df_unique



    
    
    def DisparateI(dataframe, features, target):
        disparate_impact = {}
        di = []
        di_f = []
    
        pre_out = dataframe[target].dropna().unique()
    
        for i in range(len(features)):
            print(features[i])
            df_ = dataframe[[target, features[i]]].dropna()
            thsh = 1
            for i1 in range(len(dataframe[features[i]].dropna().unique())-1):
                for i2 in range(i1, len(dataframe[features[i]].dropna().unique())):
                    
                    t1 = len(df_[(df_[target]==pre_out[0]) & (df_[features[i]]==df_[features[i]].unique()[i1])])/len(df_[df_[features[i]]==df_[features[i]].unique()[i1]])
                    t2 = len(df_[(df_[target]==pre_out[0]) & (df_[features[i]]==df_[features[i]].unique()[i2])])/len(df_[df_[features[i]]==df_[features[i]].unique()[i2]])
                    
                    if max(t1, t2) ==0:
                        t1 = t2 = 1E-9
                    
                    if thsh > (min(t1, t2)/max(t1, t2)):
                        thsh = (min(t1, t2)/max(t1, t2))
                        ind = [min(t1, t2), max(t1, t2), [t1, t2].index(min(t1, t2)), [t1, t2].index(max(t1, t2))]
    
            disparate_impact[features[i]] = thsh
    
            if thsh<0.8:
                di.append(thsh)
                di_f.append(features[i])
    
        return di, di_f