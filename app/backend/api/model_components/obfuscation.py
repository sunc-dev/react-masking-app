"""Docstring for the obfuscation.py module.

This module proposes several techniques to deal with masking different feature types. 
There are mainly 4 types of features (numeric, categorical, datetime, and PII), and obfuscation
function uses different techniques to make them: 

numerical features: greater than 1000 would be rounded to the nearest 100th
categoriacal features: are replaced with a list of alphabetical characters 
datetime features: the day portion is randomly replace with a value between 1 and 28
PII features: are masked using the unique_masking function in which a, b, and c are 
randomly generated coefficients 

"""

from datetime import datetime 
from random import choice
from string import ascii_uppercase
import random
import string
import numpy as np 
from os import path
import json
import pandas as pd 

#alphabet_string = string.ascii_lowercase
#alphabet_list = list(alphabet_string)
num2alpha = dict(zip(range(0, 26), string.ascii_uppercase))

def unique_masking(coeff, x: int):
  r"""unique_masking funcion is used to mask the value x using an arbitrary function
  where a, b, and c are randomly generated coefficients  

  Parameters
  -------
  a, b, c: int and randomly generated

  Returns
  -------
  an integer which is the masked version of x

  """
  if x == 0: 
    maskedvalue = 0
  else: 
    maskedvalue = int(coeff[1] * (int(x) ** coeff[0]) + coeff[2] * coeff[3])  
  return maskedvalue

def string_to_binary(string: str): 
  r"""string_to_binary function is used to mask categories/strings as binary    

  Parameters
  -------
  string: is a categorical values with string type

  Returns
  -------
  binary: is a stringified binary

  """
  a_byte_array = bytearray(string, "utf8")
  byte_list = []

  for byte in a_byte_array:
      binary_representation = bin(byte)
      byte_list.append(binary_representation[2:])

  return '-'.join(byte_list)

def obfuscation(data_frame, coeff: list): 
  data_frame.dropna(how = "all", inplace = True)
  r"""obfuscation function is used to mask raw files   

  Parameters
  -------
  data_frame: is a structured data frame with raw files 
  coeff: is the list of randomly generated coefficients  

  Returns
  -------
  data_frame_masked: is a structured data frame containing masked values with the same size as input 
  data frame

  """

  data_frame_masked = data_frame.copy()
  for i in data_frame_masked.columns:
    if data_frame_masked[i].dtypes == "object": 
      try: 
        data_frame_masked[i] = pd.to_datetime(data_frame_masked[i])
      except: 
        pass
    #PII, sensitive features, and unique identifiers 
    if "tel" in i.lower() or "phone" in i.lower(): 
      data_frame_masked[i] = data_frame_masked[i].apply(lambda x: str(int(str(x).replace(',', '')))[:3])
      data_frame_masked[i] = pd.to_numeric(data_frame_masked[i])

    elif ((data_frame_masked.dtypes[i] == 'int') and (len(data_frame_masked[i].unique()) == len(data_frame_masked)))\
      or (i == 'MembNo'): 
      data_frame_masked[i].fillna(0, inplace=True)
      data_frame_masked[i] = data_frame_masked[i].apply(lambda x: ' ' if x == 0 else ''.join([num2alpha[int(y)] for y in str(unique_masking(coeff, x))]))   
 
    #Datetime
    elif data_frame_masked.dtypes[i] == np.dtype('datetime64[ns]'):
      data_frame_masked[i] = data_frame_masked[i].apply(lambda dt: dt.replace(day=random.randint(1, 28)))

    #Categorical features
    elif data_frame_masked.dtypes[i] == 'object':   
      data_frame_masked[i].replace(0, '0', inplace = True)
      values = list(data_frame_masked[i].dropna().unique())
      binary_objects = [string_to_binary(x) for x in values]
      masked_obj_values = []
      for objs in binary_objects: 
        try: 
          objs = objs.split('-')
        except: 
          pass
        masked_temp = ['-'.join([''.join(z) for z in [[num2alpha[int(y)] for y in str(int(x, 2))] for x in objs]])]
        masked_obj_values.append('-'.join(masked_temp)+'-')

      data_frame_masked[i].replace(values, masked_obj_values, inplace = True)
      data_frame_masked[i].replace(np.nan, 'NaN-', inplace = True)

    #Numerics  
    elif ((data_frame_masked.dtypes[i] == 'float') or (data_frame_masked.dtypes[i] == 'int')) and \
         (data_frame_masked[i].max() > 1000):
      data_frame_masked[i].fillna(0, inplace=True)
      data_frame_masked[i] = data_frame_masked[i].apply(lambda x: (x//100)*100)
      
  return data_frame_masked  