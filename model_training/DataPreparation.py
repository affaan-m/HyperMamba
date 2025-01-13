# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:10:00 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import pandas as pd


# %%
def prepare_data():
    
    data = pd.read_csv('btc.csv')
    data = data[365*2:-1]

    data['Date'] = pd.to_datetime(data['time'])
    data.set_index('Date', inplace=True)

    data['Difficulty'] = np.log(data['DiffLast'])
    data['Transaction Count'] = np.log(data['TxCnt'])
    data['Active Addresses Count'] = np.log(data['AdrActCnt'])

    data['30 Day Active Supply'] = np.log(data['SplyAct30d'])-10
    data['1 Year Active Supply'] = np.log(data['SplyAct1yr'])-10
    data['CurrentSupply'] = np.log(data['SplyCur'])-10
    data['LogPriceUSD'] = np.log(data['PriceUSD'])

    data = data[['Difficulty', 'Transaction Count', 'Active Addresses Count', '30 Day Active Supply', '1 Year Active Supply', 'CurrentSupply', 'LogPriceUSD']]
    data.fillna(0, inplace=True)
    
    return data

data = prepare_data()

