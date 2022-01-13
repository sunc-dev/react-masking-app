# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:36:48 2021

@author: csun
"""

try:
    import os
    import pandas as pd

except (RuntimeError, Exception) as err_msg:
    print("Some Modules are Missing {}".format(err_msg))

class FetchData(object):
    '''class to define path'''
    def __init__(self):
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.folder = os.path.join(self.root, 'data')

    def path_items(self):
        '''function to return root and folder paths'''
        return self.root, self.folder
    
    def get_data(self, data):
        '''function to load data - replace with sql connection'''
        root, folder = FetchData().path_items()
        data = pd.read_csv(os.path.join(root, folder, data))

        return data, root, folder
    


if __name__ == '__main__':

    df = FetchData().get_data('data.csv')

