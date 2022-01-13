
import os
from encryptioncode import code_creation
from obfuscation import obfuscation 
#from disparateimpact import disparateI


def maskingFunc(filedirectory, df, keysfolder): 
    coeff = code_creation(keysfolder) #'encryption_keys')
    df_masked = obfuscation(df, coeff)
    df_masked.to_csv(os.path.join(filedirectory,'data_masked.csv'), index= False)

    return df_masked


#feature_fair = list(set(features_categorical))
#feat_fair = [x for x in feature_fair if x not in features_unique]

#st.write("Dataset has potential DISPARATE IMPACT (80% RULE) with respect to the following features:")
#st.markdown("The lower the ratio, the higher disparity")
#di, di_f = disparateI(df, feat_fair, target)